"""
Match harmony scoring for outfit recommendation.

대조학습(Contrastive Learning) 임베딩 유사도와 LAB 색상 조화 점수를 결합하여
최종 코디 세트를 추천합니다. MMR(Maximal Marginal Relevance)로 다양성을 확보합니다.
"""

import numpy as np
from typing import Dict, List, Optional, Set

from .color_harmony import (
    ItemColorInfo,
    TopBottomSet,
    InnerCandidate,
    FinalOutfit,
    harmony_score_lab,
)

# ---------------------------------------------------------------------------
# Constants & Hyperparameters
# ---------------------------------------------------------------------------
ALPHA_TB = 0.65
ALPHA_OI = 0.70
BETA_TB = 0.50
LAMBDA_TBSET = 0.15

MMR_LAMBDA = 0.75
MMR_MAX_CANDIDATES = 400

# ---------------------------------------------------------------------------
# Embedding Utility Functions
# ---------------------------------------------------------------------------

def build_emb_by_id(
    item_embs: np.ndarray,
    item_metas: Optional[List[dict]] = None,
    item_ids: Optional[List[str]] = None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    x = item_embs.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    x = x / norms

    if item_ids is None:
        if item_metas is None:
            raise ValueError("item_ids 또는 item_metas 중 하나는 필요합니다.")
        item_ids = [str(m.get("item_id", m.get("id", "unknown"))) for m in item_metas]

    if len(item_ids) != x.shape[0]:
        raise ValueError(
            f"길이 불일치: len(item_ids)={len(item_ids)} vs item_embs N={x.shape[0]}"
        )

    return {str(iid): x[i] for i, iid in enumerate(item_ids)}


def emb_sim_01(iid1: str, iid2: str, emb_by_id: Dict[str, np.ndarray]) -> float:
    if emb_by_id is None:
        return 0.5
    e1 = emb_by_id.get(str(iid1))
    e2 = emb_by_id.get(str(iid2))
    if e1 is None or e2 is None:
        return 0.5
    sim = float(np.dot(e1, e2))
    return 0.5 * (sim + 1.0)


def mix_score(color_s: float, emb_s01: float, alpha: float) -> float:
    return float(alpha * color_s + (1.0 - alpha) * emb_s01)


# ---------------------------------------------------------------------------
# Step 2: Top-Bottom set matching
# ---------------------------------------------------------------------------

def build_top_bottom_sets_with_emb(
    top_colors: List[ItemColorInfo],
    bottom_colors: List[ItemColorInfo],
    emb_by_id: Dict[str, np.ndarray],
    L: int = 7,
    alpha_tb: float = ALPHA_TB,
) -> List[TopBottomSet]:
    combos: List[TopBottomSet] = []
    for t in top_colors:
        for b in bottom_colors:
            c = harmony_score_lab(t.lab, b.lab)
            e = emb_sim_01(t.item_id, b.item_id, emb_by_id)
            s = mix_score(c, e, alpha_tb)
            combos.append(TopBottomSet(top_id=t.item_id, bottom_id=b.item_id, harmony=s))

    combos.sort(key=lambda x: x.harmony, reverse=True)
    return combos[:min(L, len(combos))]


# ---------------------------------------------------------------------------
# Step 4: Outer-Inner scoring & final outfit building
# ---------------------------------------------------------------------------

def _outer_inner_score_with_emb(
    outer: ItemColorInfo,
    inner: InnerCandidate,
    color_index: Dict[str, ItemColorInfo],
    emb_by_id: Dict[str, np.ndarray],
    alpha_oi: float = ALPHA_OI,
    beta_tb: float = BETA_TB,
    lambda_tbset: float = LAMBDA_TBSET,
) -> float:
    if inner.kind == "dress":
        op = color_index[inner.ids[0]]
        c = harmony_score_lab(outer.lab, op.lab)
        e = emb_sim_01(outer.item_id, op.item_id, emb_by_id)
        return mix_score(c, e, alpha_oi)

    elif inner.kind == "two_piece":
        t = color_index[inner.ids[0]]
        b = color_index[inner.ids[1]]

        c_ot = harmony_score_lab(outer.lab, t.lab)
        e_ot = emb_sim_01(outer.item_id, t.item_id, emb_by_id)
        s_ot = mix_score(c_ot, e_ot, alpha_oi)

        c_ob = harmony_score_lab(outer.lab, b.lab)
        e_ob = emb_sim_01(outer.item_id, b.item_id, emb_by_id)
        s_ob = mix_score(c_ob, e_ob, alpha_oi)

        return float(
            beta_tb * s_ot
            + (1.0 - beta_tb) * s_ob
            + lambda_tbset * inner.inner_harmony
        )

    else:
        raise ValueError(f"Unknown inner kind: {inner.kind}")


def build_final_outfits_with_match(
    outer_colors: List[ItemColorInfo],
    inner_candidates: List[InnerCandidate],
    color_index: Dict[str, ItemColorInfo],
    emb_by_id: Dict[str, np.ndarray],
    M: int = 10,
    alpha_oi: float = ALPHA_OI,
    beta_tb: float = BETA_TB,
    lambda_tbset: float = LAMBDA_TBSET,
) -> List[FinalOutfit]:
    all_combos: List[FinalOutfit] = []
    for o in outer_colors:
        for inner in inner_candidates:
            s = _outer_inner_score_with_emb(
                outer=o,
                inner=inner,
                color_index=color_index,
                emb_by_id=emb_by_id,
                alpha_oi=alpha_oi,
                beta_tb=beta_tb,
                lambda_tbset=lambda_tbset,
            )
            all_combos.append(FinalOutfit(outer_id=o.item_id, inner=inner, score=s))

    all_combos.sort(key=lambda x: x.score, reverse=True)
    return all_combos[: M * 2]


# ---------------------------------------------------------------------------
# MMR Diversity Re-ranking
# ---------------------------------------------------------------------------

def _outfit_item_set(outfit: FinalOutfit) -> Set[str]:
    s: Set[str] = set()
    if outfit.outer_id:
        s.add(str(outfit.outer_id))
    for iid in outfit.inner.ids:
        s.add(str(iid))
    return s


def _jaccard(a: Set[str], b: Set[str]) -> float:
    union = len(a | b)
    return 0.0 if union == 0 else len(a & b) / union


def _minmax_norm(xs: List[float], eps: float = 1e-12) -> List[float]:
    mn, mx = min(xs), max(xs)
    if mx - mn < eps:
        return [0.5] * len(xs)
    return [(x - mn) / (mx - mn) for x in xs]


def apply_mmr_reranking(
    outfits: List[FinalOutfit],
    M: int,
    lamb: float = MMR_LAMBDA,
    max_candidates: int = MMR_MAX_CANDIDATES,
    minmax_normalize: bool = True,
) -> List[FinalOutfit]:
    if not outfits:
        return []

    cand = outfits[: min(max_candidates, len(outfits))]
    cand_sets = [_outfit_item_set(o) for o in cand]

    raw_scores = [float(o.score) for o in cand]
    q_scores = _minmax_norm(raw_scores) if minmax_normalize else raw_scores

    selected: List[FinalOutfit] = []
    selected_sets: List[Set[str]] = []
    used = [False] * len(cand)

    while len(selected) < M:
        best_val = -1e18
        best_i = None

        for i, (cs, qs) in enumerate(zip(cand_sets, q_scores)):
            if used[i]:
                continue
            max_dup = max((_jaccard(cs, ss) for ss in selected_sets), default=0.0)
            val = lamb * qs - (1.0 - lamb) * max_dup
            if val > best_val:
                best_val = val
                best_i = i

        if best_i is None:
            break

        used[best_i] = True
        selected.append(cand[best_i])
        selected_sets.append(cand_sets[best_i])

    return selected
