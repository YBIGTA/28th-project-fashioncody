"""
Vision Pipeline: YOLO + EfficientNet + 날씨 추론

rembg (배경 제거) → YOLOv8 (의류 탐지) → EfficientNet (12속성 분류) → 날씨 자동 추론

출력 형식은 dummy.json과 동일:
  image_name, category, detection_confidence,
  sub_type, color, sub_color, sleeve_length(하의 제외), length, fit,
  collar(하의 제외), style, sub_style,
  material[], print[], detail[],
  weather

사용법 (프로젝트 루트에서 실행):
    python -m src.recognition.vision_pipeline \\
        --image "옷사진.jpg" \\
        --yolo  src/recognition/models/best.pt \\
        --effnet src/recognition/models/efficientnet_kfashion_best.pt

    python -m src.recognition.vision_pipeline \\
        --image_dir "내옷들/" \\
        --yolo  src/recognition/models/best.pt \\
        --effnet src/recognition/models/efficientnet_kfashion_best.pt \\
        --output results.json
"""

import sys
import io
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import argparse
import logging
from pathlib import Path
from PIL import Image

RECOGNITION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT    = RECOGNITION_DIR.parent.parent
sys.path.insert(0, str(RECOGNITION_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from efficientnet_classifier import EfficientNetClassifier, YOLO_TO_KR

try:
    from src.image_preprocess.remove_background import remove_background
    from rembg import new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

YOLO_CLASSES = {0: "top", 1: "bottom", 2: "outer", 3: "dress", 4: "acc"}


class VisionPipeline:
    """
    YOLO + EfficientNet 기반 의류 분석 파이프라인.

    1) rembg: 배경 제거 (선택)
    2) YOLOv8: 의류 아이템 탐지 (top/bottom/outer/dress/acc)
    3) EfficientNet: 12개 속성 분류
    4) 날씨 범주 자동 추론 (규칙 기반, 7단계)
    """

    def __init__(self, yolo_model_path, effnet_model_path,
                 use_rembg=True, device=None):
        """
        Args:
            yolo_model_path:   YOLOv8 best.pt 경로
            effnet_model_path: efficientnet_kfashion_best.pt 경로
            use_rembg:         배경 제거 사용 여부
            device:            'cuda' / 'cpu' (None이면 자동)
        """
        logging.info("VisionPipeline 초기화 중...")

        logging.info(f"  YOLOv8 로딩: {yolo_model_path}")
        self.yolo = YOLO(str(yolo_model_path))

        logging.info(f"  EfficientNet 로딩: {effnet_model_path}")
        self.classifier = EfficientNetClassifier(str(effnet_model_path), device=device)

        self.use_rembg = use_rembg and REMBG_AVAILABLE
        self.rembg_session = None
        if self.use_rembg:
            logging.info("  rembg 세션 생성 중...")
            self.rembg_session = new_session()
            logging.info("  rembg 준비 완료!")
        elif use_rembg and not REMBG_AVAILABLE:
            logging.warning("rembg 미설치 → 배경 제거 건너뜀 (pip install rembg[cpu])")

        logging.info("VisionPipeline 준비 완료!\n")

    def process(self, image_path, conf=0.3):
        """
        단일 이미지 처리.

        Args:
            image_path: 옷 사진 경로
            conf:       YOLOv8 confidence 임계값 (기본 0.3)

        Returns:
            list[dict]: dummy.json 형식의 아이템 리스트
                각 아이템:
                  image_name, category, detection_confidence,
                  sub_type, sub_type_confidence,
                  color, color_confidence,
                  sub_color, sub_color_confidence,
                  sleeve_length, sleeve_length_confidence,  (하의 제외)
                  length, length_confidence,
                  fit, fit_confidence,
                  collar, collar_confidence,                (하의 제외)
                  style, style_confidence,
                  sub_style, sub_style_confidence,
                  material[{value, confidence}],
                  print[{value, confidence}],
                  detail[{value, confidence}],
                  weather
        """
        image_path = Path(image_path)
        logging.info(f"처리 중: {image_path.name}")

        # 1) 이미지 로드 + 배경 제거
        img = Image.open(image_path).convert("RGB")
        if self.use_rembg:
            logging.info("  배경 제거 중...")
            img_rgba     = remove_background(img, self.rembg_session)
            img_processed = img_rgba.convert("RGB")
        else:
            img_processed = img

        # 2) YOLOv8 탐지
        results = self.yolo.predict(img_processed, conf=conf, verbose=False)
        boxes   = results[0].boxes

        if len(boxes) == 0:
            logging.info("  탐지된 의류 없음")
            return []

        logging.info(f"  YOLOv8 탐지: {len(boxes)}개 아이템")

        # 3) 아이템별 EfficientNet 분류
        items = []
        for i, box in enumerate(boxes):
            cls_id   = int(box.cls)
            category = YOLO_CLASSES.get(cls_id, "unknown")
            det_conf = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            cropped    = img_processed.crop((x1, y1, x2, y2))
            attr_result = self.classifier.classify(cropped, yolo_category=category)

            # dummy.json 형식으로 조립
            item = {
                "image_name":            image_path.name,
                "category":              YOLO_TO_KR.get(category, category),
                "detection_confidence":  round(det_conf, 4),
            }
            item.update(attr_result)  # EfficientNet 속성 + weather 전부 포함

            items.append(item)

            logging.info(
                f"  [{i+1}] {item['category']} ({det_conf:.0%})"
                f" → {item.get('sub_type','?')} | {item.get('style','?')}"
                f" | 날씨:{item.get('weather','?')}"
            )

        return items

    def process_directory(self, image_dir, output_json=None, conf=0.3):
        """
        폴더 내 모든 이미지 일괄 처리.

        Args:
            image_dir:   이미지 폴더 경로
            output_json: 결과 저장할 JSON 경로 (None이면 저장 안 함)
            conf:        YOLOv8 confidence 임계값

        Returns:
            list[dict]: 전체 아이템 리스트 (모든 이미지 합산)
        """
        image_dir  = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png"}
        image_files = [
            f for f in image_dir.rglob("*")
            if f.suffix.lower() in extensions
        ]

        if not image_files:
            logging.warning(f"이미지를 찾을 수 없습니다: {image_dir}")
            return []

        logging.info(f"총 {len(image_files)}장 처리 시작\n")

        all_items = []
        for i, img_path in enumerate(image_files, 1):
            logging.info(f"[{i}/{len(image_files)}] {img_path.name}")
            items = self.process(img_path, conf=conf)
            all_items.extend(items)
            print()

        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_items, f, ensure_ascii=False, indent=2)
            logging.info(f"결과 저장: {output_path}")

        logging.info(
            f"\n처리 완료: {len(image_files)}장 이미지, {len(all_items)}개 아이템"
        )
        return all_items


def _auto_numbered_path(output_path):
    """
    파일이 이미 존재하면 _1, _2, _3 ... 을 붙여 새 경로 반환.
    예: result.json → result_1.json → result_2.json
    """
    p = Path(output_path)
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    i = 1
    while True:
        new_p = p.parent / f"{stem}_{i}{suffix}"
        if not new_p.exists():
            return new_p
        i += 1


def main():
    parser = argparse.ArgumentParser(description="Vision Pipeline: YOLO + EfficientNet")
    parser.add_argument("--image",     type=str, help="단일 이미지 경로")
    parser.add_argument("--image_dir", type=str, help="이미지 폴더 경로")
    parser.add_argument("--yolo",      type=str, required=True,
                        help="YOLOv8 모델 경로 (best.pt)")
    parser.add_argument("--effnet",    type=str, required=True,
                        help="EfficientNet 모델 경로 (efficientnet_kfashion_best.pt)")
    parser.add_argument("--output",    type=str, default="result.json",
                        help="결과 JSON 저장 경로 (기본: result.json, 자동 넘버링)")
    parser.add_argument("--conf",      type=float, default=0.3,
                        help="탐지 confidence 임계값 (기본: 0.3)")
    parser.add_argument("--no_rembg",  action="store_true",
                        help="배경 제거 비활성화")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("--image 또는 --image_dir 중 하나를 지정해주세요")

    pipeline = VisionPipeline(
        yolo_model_path=args.yolo,
        effnet_model_path=args.effnet,
        use_rembg=not args.no_rembg,
    )

    # 자동 넘버링: 기존 파일 덮어쓰지 않음
    output_path = _auto_numbered_path(args.output)

    if args.image:
        items = pipeline.process(args.image, conf=args.conf)
        print("\n" + "=" * 60)
        print(json.dumps(items, ensure_ascii=False, indent=2))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {output_path}")

    elif args.image_dir:
        items = pipeline.process_directory(
            args.image_dir,
            output_json=str(output_path),
            conf=args.conf,
        )
        print("\n" + "=" * 60)
        print(json.dumps(items, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
