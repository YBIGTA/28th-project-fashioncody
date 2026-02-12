"""
패션 코디네이터 전체 파이프라인

rembg (배경 제거) → YOLOv8 (의류 탐지) → CLIP (세부 분류 + 스타일)

사용법:
    python pipeline.py --image "옷사진.jpg" --model "best.pt"
    python pipeline.py --image_dir "내옷들/" --model "best.pt"
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


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / "rec_scripts"
REMBG_PROJECT_SRC = PROJECT_DIR / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REMBG_PROJECT_SRC))

from ultralytics import YOLO
from clip_classifier import ClothingClassifier, TEMP_RANGE_MAP


try:
    from process_clothing_images import remove_background, center_and_resize
    from rembg import new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

YOLO_CLASSES = {0: "top", 1: "bottom", 2: "outer", 3: "dress", 4: "acc"}


class FashionPipeline:
    """
    전체 패션 분석 파이프라인

    1) rembg: 전체 이미지 배경 제거
    2) YOLOv8: 배경 제거된 이미지에서 의류 아이템 탐지 (top/bottom/outer/dress/acc)
    3) CLIP: 세부 의류 타입 + 스타일 분류 + 온도 범위 매핑
    """

    def __init__(self, yolo_model_path, use_rembg=True, device=None):
        """
        Args:
            yolo_model_path: 학습된 YOLOv8 모델 경로 (best.pt)
            use_rembg: 배경 제거 사용 여부
            device: "cuda" 또는 "cpu" (None이면 자동 감지)
        """
        logging.info("파이프라인 초기화 중...")

     
        logging.info(f"YOLOv8 모델 로딩: {yolo_model_path}")
        self.yolo = YOLO(str(yolo_model_path))

       
        self.classifier = ClothingClassifier(device=device)


        self.use_rembg = use_rembg and REMBG_AVAILABLE
        self.rembg_session = None
        if self.use_rembg:
            logging.info("rembg 세션 생성 중...")
            self.rembg_session = new_session()
            logging.info("rembg 준비 완료!")
        elif use_rembg and not REMBG_AVAILABLE:
            logging.warning(
                "rembg를 사용할 수 없습니다. pip install rembg[cpu] 를 실행하세요.\n"
                f"코드 경로 확인: {REMBG_PROJECT_SRC}"
            )

        logging.info("파이프라인 준비 완료!\n")

    def process(self, image_path, conf=0.3):
        """
        단일 이미지 전체 파이프라인 실행

        Args:
            image_path: 옷 사진 경로
            conf: YOLOv8 confidence 임계값 (기본 0.3)

        Returns:
            dict: {
                "image": 파일명,
                "num_items": 탐지된 아이템 수,
                "items": [
                    {
                        "category": "top",
                        "detection_confidence": 0.85,
                        "bbox": [x1, y1, x2, y2],
                        "sub_type": "후드티",
                        "sub_type_confidence": 0.65,
                        "temp_range": [17, 22],
                        "style": "캐주얼",
                        "style_confidence": 0.72,
                    },
                    ...
                ]
            }
        """
        image_path = Path(image_path)
        logging.info(f"처리 중: {image_path.name}")

        
        img = Image.open(image_path).convert("RGB")
        if self.use_rembg:
            logging.info("  배경 제거 중...")
            img_rgba = remove_background(img, self.rembg_session)
            img_processed = img_rgba.convert("RGB")
        else:
            img_processed = img

        
        results = self.yolo.predict(img_processed, conf=conf, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            logging.info("  탐지된 의류 없음")
            return {
                "image": image_path.name,
                "num_items": 0,
                "items": [],
            }

        logging.info(f"  YOLOv8 탐지: {len(boxes)}개 아이템")

        
        items = []
        for i, box in enumerate(boxes):
            cls_id = int(box.cls)
            category = YOLO_CLASSES.get(cls_id, "unknown")
            det_conf = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            cropped = img_processed.crop((x1, y1, x2, y2))
            clip_result = self.classifier.classify(cropped, category)

            item = {
                "category": category,
                "detection_confidence": round(det_conf, 4),
                "bbox": [x1, y1, x2, y2],
                "sub_type": clip_result["sub_type"],
                "sub_type_confidence": clip_result["sub_type_confidence"],
                "sub_type_top3": clip_result["sub_type_scores"][:3],
                "temp_range": clip_result["temp_range"],
                "style": clip_result["style"],
                "style_confidence": clip_result["style_confidence"],
            }
            items.append(item)

            logging.info(
                f"  [{i+1}] {category} ({det_conf:.0%}) "
                f"-> {item['sub_type']} | {item['style']} | "
                f"{item['temp_range'][0]}~{item['temp_range'][1]}도"
            )

        return {
            "image": image_path.name,
            "num_items": len(items),
            "items": items,
        }

    def process_directory(self, image_dir, output_json=None, conf=0.3):
        """
        폴더 내 모든 이미지 일괄 처리

        Args:
            image_dir: 이미지 폴더 경로
            output_json: 결과 저장할 JSON 파일 경로 (None이면 저장 안 함)
            conf: YOLOv8 confidence 임계값

        Returns:
            list of dict (각 이미지 처리 결과)
        """
        image_dir = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png"}
        image_files = [
            f for f in image_dir.rglob("*")
            if f.suffix.lower() in extensions
        ]

        if not image_files:
            logging.warning(f"이미지를 찾을 수 없습니다: {image_dir}")
            return []

        logging.info(f"총 {len(image_files)}장 처리 시작\n")

        all_results = []
        for i, img_path in enumerate(image_files, 1):
            logging.info(f"[{i}/{len(image_files)}] {img_path.name}")
            result = self.process(img_path, conf=conf)
            all_results.append(result)
            print()


        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logging.info(f"결과 저장: {output_path}")

    
        total_items = sum(r["num_items"] for r in all_results)
        logging.info(f"\n처리 완료: {len(all_results)}장 이미지, {total_items}개 아이템 탐지")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="패션 코디네이터 파이프라인")
    parser.add_argument("--image", type=str, help="단일 이미지 경로")
    parser.add_argument("--image_dir", type=str, help="이미지 폴더 경로")
    parser.add_argument("--model", type=str, required=True, help="YOLOv8 모델 경로 (best.pt)")
    parser.add_argument("--output", type=str, default=None, help="결과 JSON 저장 경로")
    parser.add_argument("--conf", type=float, default=0.3, help="탐지 confidence 임계값 (기본: 0.3)")
    parser.add_argument("--no_rembg", action="store_true", help="배경 제거 비활성화")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("--image 또는 --image_dir 중 하나를 지정해주세요")

    pipeline = FashionPipeline(
        yolo_model_path=args.model,
        use_rembg=not args.no_rembg,
    )

    if args.image:
        result = pipeline.process(args.image, conf=args.conf)
        print("\n" + "=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n결과 저장: {args.output}")

    elif args.image_dir:
        results = pipeline.process_directory(
            args.image_dir,
            output_json=args.output,
            conf=args.conf,
        )
        print("\n" + "=" * 60)
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            print()


if __name__ == "__main__":
    main()
