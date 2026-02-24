from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from .model_loader import ArtifactsBundle, load_artifacts
from .predictor import recommend_outfits
from .efficientnet_classifier import EfficientNetClassifier


app = FastAPI(title="OOTD Recommendation API")

artifacts: Optional[ArtifactsBundle] = None
classifier: Optional[EfficientNetClassifier] = None


class WeatherPayload(BaseModel):
    temperature: float = 0.0
    feels_like: float = 0.0
    precipitation: float = 0.0


class UserContextPayload(BaseModel):
    text: str
    comment: str = ""
    weather: WeatherPayload


class ClosetItemPayload(BaseModel):
    id: str
    vector: Optional[List[float]] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    season: Optional[List[str]] = None
    dominant_color_lab: Optional[List[float]] = None  # Phase 2: pre-computed LAB


class RecommendRequest(BaseModel):
    user_context: UserContextPayload
    closet_items: List[ClosetItemPayload]
    top_k: int = 10
    # 하이퍼파라미터 (기본값 = match_harmony.py 모듈 상수)
    alpha_tb: float = 0.65
    alpha_oi: float = 0.70
    mmr_lambda: float = 0.75
    beta_tb: float = 0.50
    lambda_tbset: float = 0.15


class RecommendationRow(BaseModel):
    outfit_type: Literal["two_piece", "dress"] = "two_piece"
    top_id: Optional[str] = None
    bottom_id: Optional[str] = None
    dress_id: Optional[str] = None
    outer_id: Optional[str] = None
    score: float
    reason: str


class RecommendResponse(BaseModel):
    selected_items: Dict[str, List[str]] = Field(default_factory=dict)
    recommendations: List[RecommendationRow]


@app.on_event("startup")
async def startup_event() -> None:
    global artifacts, classifier
    artifacts_path = (
        os.getenv("ARTIFACTS_PATH")
        or os.getenv("MODEL_ARTIFACTS_PATH")
        or None
    )
    artifacts = load_artifacts(artifacts_path=artifacts_path)

    # EfficientNet 이미지 분석 모델 로드 (ONNX)
    effnet_path = os.getenv("EFFNET_MODEL_PATH", "ml-server/app/efficientnet_kfashion.onnx")
    if not os.path.exists(effnet_path):
        effnet_path = "/app/app/efficientnet_kfashion.onnx"
    if os.path.exists(effnet_path):
        try:
            classifier = EfficientNetClassifier(effnet_path)
        except Exception as exc:
            print(f"EfficientNet load failed: {exc}")
    else:
        print(f"EfficientNet model not found: {effnet_path}")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    if artifacts is None:
        raise HTTPException(status_code=500, detail="model artifacts not loaded")

    mood = request.user_context.text.strip()
    if len(mood) < 2:
        raise HTTPException(status_code=400, detail="text must be at least 2 characters")

    try:
        result = recommend_outfits(
            bundle=artifacts,
            mood=mood,
            comment=request.user_context.comment or "",
            temperature=float(request.user_context.weather.temperature or 0.0),
            closet_items=[item.model_dump() for item in request.closet_items],
            top_k=int(request.top_k or 10),
            alpha_tb=request.alpha_tb,
            alpha_oi=request.alpha_oi,
            mmr_lambda=request.mmr_lambda,
            beta_tb=request.beta_tb,
            lambda_tbset=request.lambda_tbset,
        )
        return RecommendResponse(
            selected_items=result.get("selected_items", {}),
            recommendations=result.get("recommendations", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


PART_TO_CATEGORY = {
    "top": "top",
    "bottom": "bottom",
    "outer": "outer",
    "dress": "dress",
}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> Dict[str, Any]:
    """이미지 분석 → 의류 속성 반환 (upload/route.ts 호환)"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="EfficientNet model not loaded")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"이미지 읽기 실패: {exc}") from exc

    try:
        # 기본 yolo_category = 'top' (YOLO 미사용 시)
        result = classifier.classify(pil_image, yolo_category="top")

        # sub_type에서 카테고리 역추론 (카테고리→part 매핑)
        sub_type = result.get("sub_type", "")
        from .efficientnet_classifier import YOLO_TO_KR
        # sub_type 기반으로 part 추론
        category = _infer_category_from_sub_type(sub_type)

        return {
            "category": category,
            "detection_confidence": result.get("sub_type_confidence", 0.5),
            "sub_type": sub_type,
            "sub_type_confidence": result.get("sub_type_confidence"),
            "color": result.get("color"),
            "color_confidence": result.get("color_confidence"),
            "sub_color": result.get("sub_color"),
            "sub_color_confidence": result.get("sub_color_confidence"),
            "sleeve_length": result.get("sleeve_length"),
            "sleeve_length_confidence": result.get("sleeve_length_confidence"),
            "length": result.get("length"),
            "length_confidence": result.get("length_confidence"),
            "fit": result.get("fit"),
            "fit_confidence": result.get("fit_confidence"),
            "collar": result.get("collar"),
            "collar_confidence": result.get("collar_confidence"),
            "material": result.get("material"),
            "print": result.get("print"),
            "detail": result.get("detail"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# sub_type(카테고리) → part(top/bottom/outer/dress) 매핑
_SUB_TYPE_TO_PART = {
    "티셔츠": "top", "셔츠": "top", "블라우스": "top", "니트웨어": "top",
    "후드티": "top", "탑": "top", "브라탑": "top",
    "팬츠": "bottom", "청바지": "bottom", "스커트": "bottom",
    "조거팬츠": "bottom", "래깅스": "bottom",
    "재킷": "outer", "점퍼": "outer", "코트": "outer", "패딩": "outer",
    "가디건": "outer", "짚업": "outer", "베스트": "outer",
    "드레스": "dress", "점프수트": "dress",
}


def _infer_category_from_sub_type(sub_type: str) -> str:
    return _SUB_TYPE_TO_PART.get(sub_type, "top")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": artifacts is not None,
        "classifier_loaded": classifier is not None,
        "feature_cols": artifacts.feature_cols if artifacts else [],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
