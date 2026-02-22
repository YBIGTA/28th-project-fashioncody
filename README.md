# FashionCody - 멀티모달 패션 코디네이터

> **YBIGTA 28기 신입기수 프로젝트**
> 옷 사진 한 장으로 의류 속성을 자동 분석하고, 날씨에 맞는 코디를 추천하는 멀티모달 AI 시스템

---

## 개요

사용자가 옷 사진을 업로드하면 다음 과정을 자동으로 수행합니다.

1. **배경 제거** (`rembg`) - 옷만 깔끔하게 분리
2. **의류 탐지** (`YOLOv8`) - 상의/하의/아우터/원피스/액세서리 감지
3. **속성 분류** (`EfficientNet`) - 12개 세부 속성 예측
4. **날씨 추론** (규칙 기반) - 의류 속성 기반 7단계 날씨 범주 출력

---

## 프로젝트 구조

```text
28th-project-fashioncody/
├── src/
│   ├── image_preprocess/          # 이미지 전처리
│   │   ├── remove_background.py   # rembg 배경 제거
│   │   ├── center_and_resize.py   # 이미지 정규화
│   │   ├── batch.py               # 배치 처리
│   │   └── cil.py                 # CLI 전체 파이프라인
│   ├── recognition/               # 의류 인식 및 분류
│   │   ├── models/
│   │   │   ├── best.pt                        # YOLOv8 학습 모델
│   │   │   └── efficientnet_kfashion_best.pt  # EfficientNet 학습 모델
│   │   ├── notebooks/
│   │   │   └── colab_train_yolov8.ipynb       # YOLOv8 학습 노트북
│   │   ├── vision_data/                       # 데이터 정의 파일
│   │   ├── efficientnet_classifier.py         # EfficientNet 분류기 (12속성)
│   │   └── vision_pipeline.py                 # 전체 비전 파이프라인
│   ├── pipeline.py                # 메인 실행 진입점
│   └── __init__.py
├── archive/                       # 이전 CLIP 기반 접근법 (참고용)
│   ├── clip_classifier.py
│   ├── pipeline.py
│   └── rec_scripts/
├── data/
│   ├── processed/                 # 전처리 완료 데이터
│   │   ├── top/
│   │   ├── bottom/
│   │   └── outer/
│   └── raw/                       # 크롤링 원본 데이터
├── requirements.txt
└── README.md
```

---

## 모델 파이프라인

```
[입력 이미지]
      ↓
 rembg 배경제거
      ↓
 YOLOv8 탐지
 (top / bottom / outer / dress / acc)
      ↓
 EfficientNet 속성분류 (아이템별)
  ├─ 카테고리(서브타입)
  ├─ 색상 / 서브색상
  ├─ 소매기장 / 기장 / 핏
  ├─ 옷깃 / 스타일 / 서브스타일
  ├─ 소재 (다중라벨)
  ├─ 프린트 (다중라벨)
  └─ 디테일 (다중라벨)
      ↓
 날씨 자동 추론
 (한파 / 한겨울 / 쌀쌀 / 선선 / 따뜻 / 더움 / 폭염)
      ↓
[JSON 결과 출력]
```

---

## 설치 및 실행

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 파이프라인 실행

**단일 이미지 분석:**
```bash
python -m src.recognition.vision_pipeline \
    --image "옷사진.jpg" \
    --yolo  src/recognition/models/best.pt \
    --effnet src/recognition/models/efficientnet_kfashion_best.pt
```

**폴더 일괄 처리:**
```bash
python -m src.recognition.vision_pipeline \
    --image_dir "이미지폴더/" \
    --yolo  src/recognition/models/best.pt \
    --effnet src/recognition/models/efficientnet_kfashion_best.pt \
    --output results.json
```

**배경 제거 없이 실행:**
```bash
python -m src.recognition.vision_pipeline \
    --image "옷사진.jpg" \
    --yolo  src/recognition/models/best.pt \
    --effnet src/recognition/models/efficientnet_kfashion_best.pt \
    --no_rembg
```

### 3. 이미지 배경 제거만 실행

```bash
python -m src.image_preprocess.cil --input_dir data/raw --output_dir data/processed
```

---



---

## 기술 스택

| 구성 요소 | 라이브러리/모델 |
|-----------|----------------|
| 배경 제거 | `rembg[cpu]` |
| 객체 탐지 | `YOLOv8` (ultralytics) |
| 속성 분류 | `EfficientNet-B0` (timm, multi-task) |
| 이미지 처리 | `Pillow` |
| 딥러닝 프레임워크 | `PyTorch`, `torchvision` |
| UI | `Streamlit` |
| 학습 데이터 | K-Fashion 데이터셋 |

---

## 팀 구성

**YBIGTA 28기 신입기수**

- **비전팀**: 이미지 전처리, YOLO 학습, EfficientNet 멀티태스크 모델 개발
