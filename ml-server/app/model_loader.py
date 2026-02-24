from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort


def _default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "model"


_TOKEN_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]+")


def basic_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def normalize_temp_range(
    value: Any, default: Tuple[int, int] = (-50, 50)
) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return default
    return default


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


@dataclass
class ArtifactsBundle:
    cfg: Dict[str, Any]
    feature_cols: List[str]
    maps: Dict[str, Dict[str, int]]
    text_stoi: Dict[str, int]
    text_pad_idx: int
    text_unk_idx: int
    text_session: ort.InferenceSession
    item_session: ort.InferenceSession
    item_embs: np.ndarray
    item_metas: List[Dict[str, Any]]
    item_table_min: Dict[str, Dict[str, Any]]
    weather_label_to_temp_range: Dict[str, Tuple[int, int]]

    @property
    def max_len(self) -> int:
        return int(self.cfg["max_len"])

    def encode_text(self, text: str) -> np.ndarray:
        tokens = basic_tokenize(text)
        ids = [
            self.text_stoi.get(token, self.text_unk_idx) for token in tokens
        ][: self.max_len]
        attn = [1] * len(ids)
        if not ids:
            ids = [self.text_unk_idx]
            attn = [1]
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids.extend([self.text_pad_idx] * pad_len)
            attn.extend([0] * pad_len)

        input_ids = np.array(ids, dtype=np.int64).reshape(1, -1)
        attention_mask = np.array(attn, dtype=np.int64).reshape(1, -1)
        outputs = self.text_session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        return outputs[0]  # already L2-normalized by the model

    def encode_items(self, item_features: Dict[str, List[int]]) -> np.ndarray:
        num_items = len(next(iter(item_features.values())))
        num_cols = len(self.feature_cols)
        features = np.zeros((num_items, num_cols), dtype=np.int64)
        for i, col in enumerate(self.feature_cols):
            features[:, i] = item_features[col]
        outputs = self.item_session.run(None, {"features": features})
        return outputs[0]  # already L2-normalized by the model


def load_artifacts(
    artifacts_path: str | None = None, device: str | None = None
) -> ArtifactsBundle:
    if artifacts_path:
        config_dir = Path(artifacts_path).parent
    else:
        config_dir = _default_config_dir()

    # Load JSON config
    config_path = config_dir / "artifacts_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"artifacts config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg = payload["cfg"]
    feature_cols = payload["feature_cols"]
    maps = payload["maps"]
    tv = payload["text_vocab"]

    text_stoi = tv["stoi"]
    text_pad_idx = int(tv["pad_idx"])
    text_unk_idx = int(tv["unk_idx"])

    # Load ONNX sessions
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    providers = ["CPUExecutionProvider"]

    te_path = config_dir / "text_encoder.onnx"
    ie_path = config_dir / "item_encoder.onnx"

    if not te_path.exists():
        raise FileNotFoundError(f"text_encoder.onnx not found: {te_path}")
    if not ie_path.exists():
        raise FileNotFoundError(f"item_encoder.onnx not found: {ie_path}")

    text_session = ort.InferenceSession(str(te_path), opts, providers=providers)
    item_session = ort.InferenceSession(str(ie_path), opts, providers=providers)

    # Load precomputed item embeddings
    embs_path = config_dir / "item_embs.npy"
    if embs_path.exists():
        item_embs = np.load(str(embs_path)).astype(np.float32)
    else:
        embed_dim = int(cfg["embed_dim"])
        item_embs = np.empty((0, embed_dim), dtype=np.float32)

    item_metas = payload.get("item_metas", [])
    item_table_min = payload.get("item_table_min", {})

    weather_ranges_raw = payload.get("weather_label_to_temp_range", {})
    weather_ranges = {
        str(label): normalize_temp_range(rng, default=(-50, 50))
        for label, rng in weather_ranges_raw.items()
    }

    return ArtifactsBundle(
        cfg=cfg,
        feature_cols=feature_cols,
        maps=maps,
        text_stoi=text_stoi,
        text_pad_idx=text_pad_idx,
        text_unk_idx=text_unk_idx,
        text_session=text_session,
        item_session=item_session,
        item_embs=item_embs,
        item_metas=item_metas,
        item_table_min=item_table_min,
        weather_label_to_temp_range=weather_ranges,
    )
