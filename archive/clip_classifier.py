"""
CLIP Zero-shot 의류 세부 분류기

크롭된 옷 이미지를 입력받아:
  1) 세부 의류 타입 분류 (후드티, 니트, 청바지 등)
  2) 스타일 분류 (캐주얼, 포멀, 스포티)
  3) 온도 범위 자동 매핑

사용법:
    from clip_classifier import ClothingClassifier
    classifier = ClothingClassifier()
    result = classifier.classify("cropped_top.png", category="top")
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel



SUB_TYPE_CANDIDATES = {
    "top": [
        ("나시/민소매", "sleeveless top, tank top"),
        ("반팔", "short sleeve t-shirt"),
        ("얇은 셔츠", "thin shirt, light shirt"),
        ("얇은 긴팔", "thin long sleeve top"),
        ("긴팔티", "long sleeve t-shirt"),
        ("후드티", "hoodie, hooded sweatshirt"),
        ("맨투맨", "crewneck sweatshirt"),
        ("니트", "knit sweater, knitwear"),
        ("셔츠", "dress shirt, button-up shirt"),
    ],
    "bottom": [
        ("반바지", "shorts"),
        ("면바지", "cotton pants, chinos"),
        ("슬랙스", "slacks, dress pants"),
        ("스키니", "skinny jeans, skinny pants"),
        ("청바지", "blue jeans, denim jeans"),
        ("스타킹", "stockings, tights"),
    ],
    "outer": [
        ("가디건", "cardigan"),
        ("간절기 야상", "light field jacket, utility jacket"),
        ("자켓", "jacket, blazer"),
        ("트렌치코트", "trench coat"),
        ("가죽자켓", "leather jacket"),
        ("코트", "wool coat, long coat"),
        ("패딩", "puffer jacket, padded jacket, down jacket"),
        ("야상", "field jacket, military jacket"),
    ],
    "dress": [
        ("민소매 원피스", "sleeveless dress"),
        ("원피스", "long sleeve dress, dress"),
    ],
    "acc": [
        ("목도리", "scarf, muffler"),
    ],
}

# ── 세부 의류 → 온도 범위 매핑 ──
TEMP_RANGE_MAP = {
    "나시/민소매": [27, 40],
    "반팔": [23, 26],
    "얇은 셔츠": [23, 26],
    "얇은 긴팔": [23, 26],
    "긴팔티": [20, 22],
    "후드티": [17, 22],
    "맨투맨": [17, 19],
    "니트": [17, 19],
    "셔츠": [12, 16],
    "반바지": [23, 40],
    "면바지": [17, 26],
    "슬랙스": [17, 22],
    "스키니": [20, 22],
    "청바지": [17, 19],
    "스타킹": [12, 16],
    "가디건": [17, 22],
    "간절기 야상": [10, 16],
    "자켓": [12, 16],
    "트렌치코트": [10, 11],
    "가죽자켓": [6, 9],
    "코트": [6, 9],
    "패딩": [-20, 5],
    "야상": [-20, 5],
    "민소매 원피스": [27, 40],
    "원피스": [17, 19],
    "목도리": [-20, 5],
}


STYLE_CANDIDATES = [
    ("캐주얼", "casual style clothing"),
    ("포멀", "formal style clothing, business attire"),
    ("스포티", "sporty style clothing, athletic wear"),
]


class ClothingClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        CLIP 모델 로드

        Args:
            model_name: CLIP 모델 이름
            device: "cuda" 또는 "cpu" (None이면 자동 감지)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CLIP 모델 로딩 중... (device: {self.device})")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        print("모델 로딩 완료!")

    def _get_scores(self, image, text_candidates):
        """
        이미지와 텍스트 후보들 사이의 유사도 점수 계산

        Returns:
            list of (label, score) 내림차순 정렬
        """
        labels = [c[0] for c in text_candidates]
        prompts = [f"a photo of {c[1]}" for c in text_candidates]

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # (1, num_candidates)
            probs = logits.softmax(dim=1).squeeze().cpu().tolist()


        if isinstance(probs, float):
            probs = [probs]

        scored = list(zip(labels, probs))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def classify_sub_type(self, image, category):
        """
        세부 의류 타입 분류

        Args:
            image: PIL Image
            category: "top", "bottom", "outer", "dress", "acc"

        Returns:
            dict with "sub_type", "confidence", "all_scores"
        """
        candidates = SUB_TYPE_CANDIDATES.get(category)
        if not candidates:
            return {"sub_type": "unknown", "confidence": 0.0, "all_scores": []}

        scores = self._get_scores(image, candidates)

        return {
            "sub_type": scores[0][0],
            "confidence": round(scores[0][1], 4),
            "all_scores": [(name, round(s, 4)) for name, s in scores],
        }

    def classify_style(self, image):
        """
        스타일 분류 (캐주얼 / 포멀 / 스포티)

        Returns:
            dict with "style", "confidence", "all_scores"
        """
        scores = self._get_scores(image, STYLE_CANDIDATES)

        return {
            "style": scores[0][0],
            "confidence": round(scores[0][1], 4),
            "all_scores": [(name, round(s, 4)) for name, s in scores],
        }

    def classify(self, image_or_path, category):
        """
        전체 분류 실행 (세부 타입 + 스타일 + 온도 매핑)

        Args:
            image_or_path: PIL Image 또는 이미지 경로 (str/Path)
            category: YOLOv8이 탐지한 카테고리 ("top", "bottom", "outer", "dress", "acc")

        Returns:
            dict with full classification result
        """
        if isinstance(image_or_path, Image.Image):
            image = image_or_path.convert("RGB")
            image_path = "PIL_Image"
        else:
            image = Image.open(image_or_path).convert("RGB")
            image_path = str(image_or_path)

        sub_result = self.classify_sub_type(image, category)
        style_result = self.classify_style(image)

        temp_range = TEMP_RANGE_MAP.get(sub_result["sub_type"], [0, 40])

        return {
            "image": image_path,
            "category": category,
            "sub_type": sub_result["sub_type"],
            "sub_type_confidence": sub_result["confidence"],
            "sub_type_scores": sub_result["all_scores"],
            "temp_range": temp_range,
            "style": style_result["style"],
            "style_confidence": style_result["confidence"],
            "style_scores": style_result["all_scores"],
        }
