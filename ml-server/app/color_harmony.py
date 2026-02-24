"""Color harmony scoring for outfit recommendation.

Ported from 색_기반_추천_매칭.ipynb.
Uses LAB color space and hue-based harmony rules (complementary, analogous, triadic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# LAB lookup table for named colors
# ---------------------------------------------------------------------------
# Representative CIE-LAB values for each color in COLOR_ALIASES (predictor.py).
# L in [0,100], a/b roughly [-128,127].

COLOR_NAME_TO_LAB: Dict[str, np.ndarray] = {
    "black": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "white": np.array([100.0, 0.0, 0.0], dtype=np.float32),
    "gray": np.array([53.6, 0.0, 0.0], dtype=np.float32),
    "grey": np.array([53.6, 0.0, 0.0], dtype=np.float32),
    "blue": np.array([32.3, 79.2, -107.9], dtype=np.float32),
    "skyblue": np.array([79.2, -14.8, -21.0], dtype=np.float32),
    "navy": np.array([12.0, 47.0, -64.0], dtype=np.float32),
    "red": np.array([53.2, 80.1, 67.2], dtype=np.float32),
    "pink": np.array([74.9, 23.1, 2.1], dtype=np.float32),
    "purple": np.array([29.8, 58.9, -36.5], dtype=np.float32),
    "lavender": np.array([78.4, 7.3, -15.4], dtype=np.float32),
    "green": np.array([46.2, -51.7, 49.9], dtype=np.float32),
    "mint": np.array([87.8, -22.5, -4.1], dtype=np.float32),
    "yellow": np.array([97.1, -21.6, 94.5], dtype=np.float32),
    "orange": np.array([67.0, 43.0, 74.1], dtype=np.float32),
    "brown": np.array([37.0, 15.0, 25.0], dtype=np.float32),
    "beige": np.array([86.9, 1.1, 17.4], dtype=np.float32),
    "khaki": np.array([65.7, -6.4, 28.8], dtype=np.float32),
    "gold": np.array([79.2, 2.5, 60.4], dtype=np.float32),
    "silver": np.array([77.7, 0.0, 0.0], dtype=np.float32),
}

# Korean color name -> English key (reverse of COLOR_ALIASES in predictor.py)
_COLOR_KR_TO_EN: Dict[str, str] = {
    "블랙": "black",
    "화이트": "white",
    "그레이": "gray",
    "블루": "blue",
    "스카이블루": "skyblue",
    "네이비": "navy",
    "레드": "red",
    "핑크": "pink",
    "퍼플": "purple",
    "라벤더": "lavender",
    "그린": "green",
    "민트": "mint",
    "옐로우": "yellow",
    "오렌지": "orange",
    "브라운": "brown",
    "베이지": "beige",
    "카키": "khaki",
    "골드": "gold",
    "실버": "silver",
    "없음": "gray",
}

# Neutral mid-gray fallback
_FALLBACK_LAB = np.array([53.6, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# LAB resolution
# ---------------------------------------------------------------------------

def resolve_item_lab(
    color: Optional[str],
    sub_color: Optional[str] = None,
) -> np.ndarray:
    """Resolve color name(s) to a LAB array.

    If both *color* and *sub_color* are available, blend 70/30.
    Accepts English ("navy") or Korean ("네이비") names.
    Falls back to mid-gray for unknown or missing colors.
    """
    primary = _lookup_lab(color)
    if sub_color is None:
        return primary

    secondary = _lookup_lab(sub_color)
    # Only blend if secondary is different from the fallback
    if np.array_equal(secondary, _FALLBACK_LAB) and not np.array_equal(primary, _FALLBACK_LAB):
        return primary
    return (0.7 * primary + 0.3 * secondary).astype(np.float32)


def _lookup_lab(name: Optional[str]) -> np.ndarray:
    if name is None:
        return _FALLBACK_LAB.copy()
    key = name.strip().lower()
    if key in COLOR_NAME_TO_LAB:
        return COLOR_NAME_TO_LAB[key].copy()
    # Try Korean -> English
    en = _COLOR_KR_TO_EN.get(key)
    if en and en in COLOR_NAME_TO_LAB:
        return COLOR_NAME_TO_LAB[en].copy()
    return _FALLBACK_LAB.copy()


# ---------------------------------------------------------------------------
# LAB -> LCh conversion & harmony scoring  (from notebook cell 3)
# ---------------------------------------------------------------------------

def _lab_to_lch(lab: np.ndarray) -> Tuple[float, float, float]:
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    C = math.sqrt(a * a + b * b)
    h = math.degrees(math.atan2(b, a))
    if h < 0:
        h += 360.0
    return L, C, h


def _angle_diff_deg(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def _gaussian_score(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def harmony_score_lab(
    lab1: np.ndarray,
    lab2: np.ndarray,
    w_comp: float = 0.50,
    w_anal: float = 0.30,
    w_tria: float = 0.20,
    sigma_comp: float = 25.0,
    sigma_anal: float = 18.0,
    sigma_tria: float = 22.0,
    w_light: float = 0.25,
    sigma_light: float = 25.0,
    w_chroma: float = 0.10,
) -> float:
    """Hue-based harmony score (higher = better).

    Uses complementary / analogous / triadic harmony on the LCh hue wheel,
    plus lightness contrast and chroma boost.  Directly ported from notebook.
    """
    L1, C1, h1 = _lab_to_lch(lab1)
    L2, C2, h2 = _lab_to_lch(lab2)

    dh = _angle_diff_deg(h1, h2)

    s_comp = _gaussian_score(dh, 180.0, sigma_comp)
    s_anal = _gaussian_score(dh, 30.0, sigma_anal) + _gaussian_score(dh, 0.0, sigma_anal)
    s_tria = _gaussian_score(dh, 120.0, sigma_tria)

    hue_harmony = w_comp * s_comp + w_anal * s_anal + w_tria * s_tria

    # Lightness contrast: some difference is preferred over identical lightness
    dL = abs(L1 - L2)
    light = 1.0 - _gaussian_score(dL, 0.0, sigma_light)

    # Chroma: very low chroma (achromatic) weakens hue meaning
    Cmean = (C1 + C2) / 2.0
    chroma_boost = 1.0 - math.exp(-Cmean / 25.0)

    return float(hue_harmony + w_light * light + w_chroma * chroma_boost)


def describe_harmony(lab1: np.ndarray, lab2: np.ndarray) -> str:
    """Return a short Korean description of the dominant harmony type."""
    _, C1, h1 = _lab_to_lch(lab1)
    _, C2, h2 = _lab_to_lch(lab2)
    Cmean = (C1 + C2) / 2.0

    # Achromatic items (black/white/gray) have very low chroma
    if Cmean < 5.0:
        return "무채색 톤 매치"

    dh = _angle_diff_deg(h1, h2)

    if dh <= 20:
        return "동일 색상 계열"
    if dh <= 45:
        return "유사색 조화"
    if 90 <= dh <= 150:
        return "삼각 조화"
    if dh >= 150:
        return "보색 대비"
    return "색상 밸런스"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemColorInfo:
    item_id: str
    part: str
    lab: np.ndarray
    similarity: float  # mood similarity from Step 1


@dataclass(frozen=True)
class TopBottomSet:
    top_id: str
    bottom_id: str
    harmony: float


@dataclass(frozen=True)
class InnerCandidate:
    kind: str  # "dress" or "two_piece"
    ids: Tuple[str, ...]  # (dress_id,) or (top_id, bottom_id)
    inner_harmony: float  # 0.0 for dresses, tb harmony for two_piece


@dataclass(frozen=True)
class FinalOutfit:
    outer_id: Optional[str]
    inner: InnerCandidate
    score: float


# ---------------------------------------------------------------------------
# Step 2: Top/Bottom set building  (from notebook cell 5)
# ---------------------------------------------------------------------------

def build_top_bottom_sets(
    tops: List[ItemColorInfo],
    bottoms: List[ItemColorInfo],
    L: int = 15,
) -> List[TopBottomSet]:
    """Score all top x bottom pairs by color harmony, return top-L."""
    combos: List[TopBottomSet] = []
    for t in tops:
        for b in bottoms:
            s = harmony_score_lab(t.lab, b.lab)
            combos.append(TopBottomSet(top_id=t.item_id, bottom_id=b.item_id, harmony=s))
    combos.sort(key=lambda x: x.harmony, reverse=True)
    return combos[: min(L, len(combos))]


# ---------------------------------------------------------------------------
# Step 3: Inner candidates  (from notebook cell 6)
# ---------------------------------------------------------------------------

def build_inner_candidates(
    dresses: List[ItemColorInfo],
    tb_sets: List[TopBottomSet],
) -> List[InnerCandidate]:
    """Combine dress items and top/bottom sets into a unified inner list."""
    inners: List[InnerCandidate] = []
    for d in dresses:
        inners.append(InnerCandidate(kind="dress", ids=(d.item_id,), inner_harmony=0.0))
    for s in tb_sets:
        inners.append(
            InnerCandidate(kind="two_piece", ids=(s.top_id, s.bottom_id), inner_harmony=s.harmony)
        )
    return inners


# ---------------------------------------------------------------------------
# Step 4: Final outfit building  (from notebook cell 6)
# ---------------------------------------------------------------------------

def _outer_inner_harmony(
    outer: ItemColorInfo,
    inner: InnerCandidate,
    color_index: Dict[str, ItemColorInfo],
    w_inner_hint: float = 0.10,
) -> float:
    """Color harmony between an outer and an inner candidate."""
    if inner.kind == "dress":
        ic = color_index.get(inner.ids[0])
        if ic is None:
            return 0.0
        base = harmony_score_lab(outer.lab, ic.lab)
    else:  # two_piece
        t = color_index.get(inner.ids[0])
        b = color_index.get(inner.ids[1])
        if t is None or b is None:
            return 0.0
        base = 0.5 * harmony_score_lab(outer.lab, t.lab) + 0.5 * harmony_score_lab(outer.lab, b.lab)

    return float(base + w_inner_hint * inner.inner_harmony)


def _inner_only_harmony(
    inner: InnerCandidate,
    color_index: Dict[str, ItemColorInfo],
) -> float:
    """Harmony score for an inner candidate without outer (uses internal harmony only)."""
    if inner.kind == "dress":
        return 0.0  # single item, no pair harmony
    return inner.inner_harmony


def _avg_mood_similarity(
    inner: InnerCandidate,
    outer: Optional[ItemColorInfo],
    mood_scores: Dict[str, float],
) -> float:
    """Weighted average mood similarity for the items in a combo."""
    sims: List[float] = []
    for item_id in inner.ids:
        sims.append(mood_scores.get(item_id, 0.0))
    if outer is not None:
        sims.append(mood_scores.get(outer.item_id, 0.0))
    return sum(sims) / len(sims) if sims else 0.0


def build_final_outfits(
    outers: List[ItemColorInfo],
    inner_candidates: List[InnerCandidate],
    color_index: Dict[str, ItemColorInfo],
    mood_scores: Dict[str, float],
    M: int = 10,
    temperature: float = 20.0,
    w_mood: float = 0.6,
    w_color: float = 0.4,
) -> List[FinalOutfit]:
    """Build outer x inner combinations, plus inner-only combos for warm weather.

    Final score = w_mood * avg_mood_similarity + w_color * color_harmony
    with weather adjustments matching the original predictor logic.
    """
    cold_weather = temperature <= 18.0
    results: List[FinalOutfit] = []

    # Outer x Inner combinations
    for outer in outers:
        for inner in inner_candidates:
            color_h = _outer_inner_harmony(outer, inner, color_index)
            mood_s = _avg_mood_similarity(inner, outer, mood_scores)
            raw = w_mood * mood_s + w_color * color_h
            if not cold_weather:
                raw -= 0.01  # warm-weather outer penalty
            results.append(FinalOutfit(outer_id=outer.item_id, inner=inner, score=raw))

    # Inner-only combinations (no outer)
    for inner in inner_candidates:
        color_h = _inner_only_harmony(inner, color_index)
        mood_s = _avg_mood_similarity(inner, None, mood_scores)
        raw = w_mood * mood_s + w_color * color_h
        if cold_weather:
            raw -= 0.03  # cold-weather no-outer penalty
        results.append(FinalOutfit(outer_id=None, inner=inner, score=raw))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[: min(M, len(results))]
