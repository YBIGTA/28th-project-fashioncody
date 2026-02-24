#!/usr/bin/env python3
"""Convert PyTorch models to ONNX format for torch-free deployment.

Outputs:
  model/text_encoder.onnx
  model/item_encoder.onnx
  model/artifacts_config.json
  model/item_embs.npy
  ml-server/app/efficientnet_kfashion.onnx
  ml-server/app/effnet_labels.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root so we can import ml-server modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml-server"))

from app.model_loader import TextEncoder, ItemEncoder, _safe_torch_load
from app.efficientnet_classifier import (
    FashionMultiTaskModel,
    SINGLE_LABEL_ATTRS,
    MULTI_LABEL_ATTRS,
)


# ── helpers ──────────────────────────────────────────────────────────

def _to_json_safe(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj


# ── EfficientNet ─────────────────────────────────────────────────────

class _EfficientNetONNXWrapper(nn.Module):
    """Wraps FashionMultiTaskModel to return a tuple (ONNX-friendly)."""

    def __init__(self, model, output_order):
        super().__init__()
        self.model = model
        self.output_order = output_order

    def forward(self, x):
        outputs = self.model(x)
        return tuple(outputs[name] for name in self.output_order)


def convert_efficientnet():
    print("=== Converting EfficientNet ===")
    ckpt = torch.load(
        str(ROOT / "ml-server/app/efficientnet_kfashion_best.pt"),
        map_location="cpu",
    )

    single_attrs = ckpt.get("single_label_attrs", SINGLE_LABEL_ATTRS)
    multi_attrs = ckpt.get("multi_label_attrs", MULTI_LABEL_ATTRS)

    model = FashionMultiTaskModel(single_attrs, multi_attrs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    output_order = list(single_attrs.keys()) + list(multi_attrs.keys())
    wrapper = _EfficientNetONNXWrapper(model, output_order)
    wrapper.eval()

    dummy = torch.randn(1, 3, 224, 224)

    out_path = ROOT / "ml-server/app/efficientnet_kfashion.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        opset_version=18,
        input_names=["image"],
        output_names=output_order,
        dynamic_axes={"image": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save label definitions
    labels = {
        "single_label_attrs": {k: list(v) for k, v in single_attrs.items()},
        "multi_label_attrs": {k: list(v) for k, v in multi_attrs.items()},
    }
    labels_path = ROOT / "ml-server/app/effnet_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {labels_path}")


# ── ItemEncoder ONNX wrapper ────────────────────────────────────────

class _ItemEncoderONNXWrapper(nn.Module):
    """Takes [N, num_features] int64 instead of Dict[str, Tensor]."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, features):
        feats = {}
        for i, col in enumerate(self.encoder.feature_cols):
            feats[col] = features[:, i]
        return self.encoder(feats)


# ── Artifacts (TextEncoder + ItemEncoder + config) ───────────────────

def convert_artifacts():
    print("=== Converting Artifacts ===")
    payload = _safe_torch_load(ROOT / "model/artifacts.pt")

    cfg = payload["cfg"]
    feature_cols = payload["FEATURE_COLS"]
    maps = payload["maps"]
    tv = payload["text_vocab"]

    text_stoi = dict(tv["stoi"])
    text_pad_idx = int(tv["pad_idx"])
    text_unk_idx = int(tv["unk_idx"])
    vocab_size = len(tv["itos"])

    # ── TextEncoder ──
    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        emb_dim=int(cfg["text_emb_dim"]),
        hidden_dim=int(cfg["text_hidden_dim"]),
        out_dim=int(cfg["embed_dim"]),
        pad_idx=text_pad_idx,
        use_attn=bool(cfg.get("text_use_attn", False)),
        nhead=int(cfg.get("text_attn_nhead", 4)),
        ff_dim=int(cfg.get("text_attn_ff", 256)),
    )
    text_encoder.load_state_dict(payload["text_enc_state"], strict=True)
    text_encoder.eval()

    max_len = int(cfg["max_len"])
    dummy_ids = torch.randint(0, vocab_size, (1, max_len))
    dummy_mask = torch.ones(1, max_len, dtype=torch.long)

    te_path = ROOT / "model/text_encoder.onnx"
    torch.onnx.export(
        text_encoder,
        (dummy_ids, dummy_mask),
        str(te_path),
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  TextEncoder saved: {te_path} ({te_path.stat().st_size / 1024:.1f} KB)")

    # ── ItemEncoder ──
    item_encoder = ItemEncoder(
        maps=maps,
        feature_cols=feature_cols,
        emb_dim=int(cfg["cat_emb_dim"]),
        hidden_dim=int(cfg["item_hidden_dim"]),
        out_dim=int(cfg["embed_dim"]),
    )
    item_encoder.load_state_dict(payload["item_enc_state"], strict=True)
    item_encoder.eval()

    wrapper = _ItemEncoderONNXWrapper(item_encoder)
    wrapper.eval()

    num_features = len(feature_cols)
    dummy_feats = torch.randint(0, 5, (1, num_features))

    ie_path = ROOT / "model/item_encoder.onnx"
    torch.onnx.export(
        wrapper,
        dummy_feats,
        str(ie_path),
        opset_version=18,
        input_names=["features"],
        output_names=["item_embedding"],
        dynamic_axes={"features": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  ItemEncoder saved: {ie_path} ({ie_path.stat().st_size / 1024:.1f} KB)")

    # ── item_embs numpy ──
    item_embs = payload.get("item_embs", torch.empty(0, int(cfg["embed_dim"])))
    embs_path = ROOT / "model/item_embs.npy"
    np.save(str(embs_path), item_embs.float().numpy())
    print(f"  item_embs saved: shape={item_embs.shape}")

    # ── Config JSON ──
    cfg_safe = {k: _to_json_safe(v) for k, v in cfg.items()}

    maps_safe = {}
    for col, mapping in maps.items():
        maps_safe[col] = {str(k): int(v) for k, v in mapping.items()}

    item_metas = payload.get("item_metas", [])
    item_metas_safe = []
    for meta in item_metas:
        meta_safe = {k: _to_json_safe(v) for k, v in meta.items()}
        item_metas_safe.append(meta_safe)

    item_table_min = payload.get("item_table_min", {})
    item_table_min_safe = {}
    for k, v in item_table_min.items():
        if isinstance(v, dict):
            item_table_min_safe[str(k)] = {kk: _to_json_safe(vv) for kk, vv in v.items()}
        else:
            item_table_min_safe[str(k)] = _to_json_safe(v)

    weather_ranges_raw = payload.get("WEATHER_LABEL_TO_TEMP_RANGE", {})
    weather_ranges = {}
    for label, rng in weather_ranges_raw.items():
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            weather_ranges[str(label)] = [int(rng[0]), int(rng[1])]

    config = {
        "cfg": cfg_safe,
        "feature_cols": feature_cols,
        "maps": maps_safe,
        "text_vocab": {
            "stoi": text_stoi,
            "pad_idx": text_pad_idx,
            "unk_idx": text_unk_idx,
            "vocab_size": vocab_size,
        },
        "item_metas": item_metas_safe,
        "item_table_min": item_table_min_safe,
        "weather_label_to_temp_range": weather_ranges,
    }

    config_path = ROOT / "model/artifacts_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    print(f"  Config saved: {config_path} ({config_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    convert_efficientnet()
    convert_artifacts()
    print("\n=== All conversions complete! ===")
