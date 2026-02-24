# FashionCody — AI 기반 코디 추천 서비스

> **YBIGTA 28기 신입기수 프로젝트**
> 옷장 사진을 업로드하면 추구미·날씨·상황에 맞는 코디를 자동 추천하는 AI 웹 애플리케이션

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [기술 스택](#2-기술-스택)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [프로젝트 구조](#4-프로젝트-구조)
5. [ML 파이프라인](#5-ml-파이프라인)
6. [추천 알고리즘](#6-추천-알고리즘)
7. [데이터베이스](#7-데이터베이스)
8. [배포 구조](#8-배포-구조)
9. [로컬 실행 방법](#9-로컬-실행-방법)
10. [환경 변수](#10-환경-변수)
11. [팀 구성](#11-팀-구성)

---

## 1. 프로젝트 개요

**FashionCody**는 사용자의 옷장 데이터를 AI로 분석하고, 실시간 날씨와 사용자 추구미(상황)를 결합해 최적의 코디를 추천하는 서비스입니다.

### 핵심 기능

| 기능 | 설명 |
|------|------|
| **의류 이미지 분석** | 업로드된 옷 사진에서 12개 속성 자동 추출 (EfficientNet-B0) |
| **AI 코디 추천** | 무드 텍스트 + 날씨를 반영한 맞춤 코디 10벌 추천 |
| **색상 조화 매칭** | CIE LAB 색공간 기반 색상 조화 분석 (유사색·보색 조합) |
| **피드백 학습** | 좋아요/싫어요 피드백으로 추천 알고리즘 개인화 |
| **실시간 날씨** | WeatherAPI.com 연동, 기온·체감온도·강수량 반영 |
| **옷장 관리** | 의류 이미지 업로드, 속성 자동 분석, CRUD |

### 사용자 플로우

```
1. /ootd 진입       → 실시간 날씨 + 옷장 자동 로드
2. 이미지 업로드    → EfficientNet으로 12속성 자동 분석 → 옷장 등록
3. 무드 텍스트 입력 → 예: "오늘은 캐주얼하게 입고 싶어"
4. 추천받기 클릭    → AI 선정 후보 아이템 표시 (카테고리별 7개)
5. 최종 코디 추천   → 점수 + 추천 이유 포함 코디 10벌 출력
6. 피드백 입력      → 좋아요/싫어요 → 다음 추천에 자동 반영
```

---

## 2. 기술 스택

| 영역 | 기술 |
|------|------|
| **프론트엔드** | Next.js 16 (App Router) + React 19 + TypeScript |
| **스타일링** | Tailwind CSS 4 + Shadcn/ui (Radix UI) |
| **데이터베이스** | Neon PostgreSQL + pgvector |
| **ML 서버** | FastAPI + ONNX Runtime (Python) |
| **이미지 저장** | Cloudinary CDN |
| **날씨 API** | WeatherAPI.com |
| **프론트 배포** | Vercel |
| **ML 배포** | Railway (Docker) |
| **의류 탐지** | YOLOv8 (`best.pt`) |
| **속성 분류** | EfficientNet-B0 Multi-Task (`efficientnet_kfashion.onnx`) |
| **추천 모델** | MLP 기반 텍스트/아이템 인코더 (ONNX) |
| **학습 데이터** | K-Fashion 데이터셋 |

---

## 3. 시스템 아키텍처

```
┌─────────────────────────────────────────────────┐
│  사용자 브라우저 (/ootd)                          │
│  무드 입력 / 옷장 조회 / 추천 요청 / 피드백        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Vercel (Next.js App Router)                    │
│                                                  │
│  /api/closet          — 옷장 CRUD                │
│  /api/closet/upload   — 이미지 업로드 + 분석      │
│  /api/recommend       — ML 추천 요청              │
│  /api/feedback        — 피드백 → θ 갱신           │
│  /api/weather         — 날씨 API 프록시           │
└───┬────────────────┬──────────────────┬──────────┘
    │                │                  │
    ▼                ▼                  ▼
┌────────┐    ┌──────────┐    ┌─────────────────────┐
│  Neon  │    │Cloudinary│    │  Railway (ML Server) │
│  Post  │    │  CDN     │    │                      │
│  gres  │    │  이미지   │    │  FastAPI + ONNX RT   │
│        │    │  서빙     │    │  /recommend  코디추천 │
│ closet │    └──────────┘    │  /analyze    속성분석 │
│ rec_   │                    │  /health     헬스체크 │
│ hist   │                    │                      │
│ user_  │                    │  text_encoder.onnx   │
│ hp     │                    │  item_encoder.onnx   │
│        │                    │  efficientnet.onnx   │
└────────┘                    └─────────────────────┘
```

---

## 4. 프로젝트 구조

```
28th-project-fashioncody/
├── src/                              # Next.js 프론트엔드 + API
│   ├── app/
│   │   ├── page.tsx                  # 랜딩 페이지
│   │   ├── ootd/page.tsx             # 메인 추천 페이지
│   │   └── api/
│   │       ├── closet/               # 옷장 CRUD API
│   │       ├── recommend/            # 추천 API
│   │       ├── feedback/             # 피드백 API
│   │       └── weather/              # 날씨 프록시 API
│   ├── components/
│   │   ├── landing/                  # 랜딩 페이지 컴포넌트
│   │   ├── ootd/                     # 추천 페이지 컴포넌트
│   │   └── ui/                       # Shadcn/ui 기본 컴포넌트
│   └── lib/
│       ├── db/                       # 데이터베이스 레포지토리
│       └── types/                    # TypeScript 타입 정의
│
├── ml-server/                        # FastAPI ML 추론 서버
│   ├── app/
│   │   ├── main.py                   # FastAPI 엔트리포인트
│   │   ├── predictor.py              # 추천 파이프라인 (4단계)
│   │   ├── color_harmony.py          # LAB 색상 조화
│   │   ├── match_harmony.py          # 임베딩 매칭 + MMR
│   │   └── efficientnet_classifier.py # EfficientNet 속성 분류
│   └── requirements.txt
│
├── model/                            # 학습된 모델 아티팩트
│   ├── text_encoder.onnx             # 텍스트 임베딩 (1.1 MB)
│   ├── item_encoder.onnx             # 아이템 임베딩 (0.6 MB)
│   ├── item_embs.npy                 # 사전 계산 임베딩 (83 MB)
│   └── artifacts_config.json         # 어휘/매핑 설정 (8.2 MB)
│
├── ml-recommendation/                # 추천 모델 학습 코드
│   └── train/
│       ├── train_model.py            # MLP 인코더 학습
│       ├── preprocess.py             # 데이터 전처리
│       └── tokenizer.py              # 토크나이저 설정
│
├── recognition/ (src/recognition/)   # 이미지 인식 파이프라인 (학습/개발용)
│   ├── models/
│   │   ├── best.pt                   # YOLOv8 학습 모델
│   │   └── efficientnet_kfashion_best.pt
│   ├── efficientnet_classifier.py    # 속성 분류기 (12속성)
│   └── vision_pipeline.py            # 전체 비전 파이프라인
│
├── data/
│   └── items.csv                     # 의류 데이터셋 (1,427개)
│
├── database/
│   └── schema.sql                    # PostgreSQL 전체 스키마
│
├── scripts/
│   ├── seed.ts                       # DB 시딩
│   └── convert_to_onnx.py            # PyTorch → ONNX 변환
│
├── Dockerfile.ml                     # ML 서버 Docker 빌드 파일
├── package.json
└── requirements.txt
```

---

## 5. ML 파이프라인

### 이미지 분석 파이프라인

```
[입력 이미지]
     ↓
rembg — 배경 제거
     ↓
YOLOv8 — 의류 탐지 (top / bottom / outer / dress / acc)
     ↓
EfficientNet-B0 Multi-Task — 12개 속성 분류
  ├─ 단일 라벨 Softmax (9개)
  │   ├─ 카테고리  (21종): 탑, 후드티, 패딩, 코트, 청바지 ...
  │   ├─ 색상      (21종): 블랙, 화이트, 네이비, 베이지 ...
  │   ├─ 서브색상  (21종)
  │   ├─ 소매기장  (6종): 긴팔, 반팔, 민소매, 7부소매 ...
  │   ├─ 기장      (9종): 롱, 미디, 미니, 크롭, 맥시 ...
  │   ├─ 핏        (7종): 오버사이즈, 루즈, 타이트 ...
  │   ├─ 옷깃      (9종): 셔츠칼라, 폴로칼라, 밴드칼라 ...
  │   ├─ 스타일    (23종): 캐주얼, 스트리트, 스포티 ...
  │   └─ 서브스타일(23종)
  └─ 다중 라벨 Sigmoid (3개)
      ├─ 소재  (25종): 니트, 데님, 린넨, 울/캐시미어 ...
      ├─ 프린트(21종): 무지, 스트라이프, 체크, 플로럴 ...
      └─ 디테일(40종): 지퍼, 포켓, 리본, 러플 ...
     ↓
날씨 자동 추론 (score = CATEGORY + SLEEVE + MATERIAL + LENGTH)
     ↓
[JSON 결과 출력]
```

### 날씨 점수 시스템

| 범주 | 온도 | score 기준 |
|------|------|------------|
| 한파 | -20 ~ -5°C | ≥ 7 |
| 한겨울 | -5 ~ 5°C | ≥ 5 |
| 쌀쌀 | 5 ~ 15°C | ≥ 3 |
| 선선 | 15 ~ 20°C | ≥ 1 |
| 따뜻 | 20 ~ 25°C | ≥ -1 |
| 더움 | 25 ~ 33°C | ≥ -3 |
| 폭염 | 33 ~ 40°C | < -3 |

---

## 6. 추천 알고리즘

### 4단계 추천 파이프라인

```
Step 1  후보 선정
  ├─ 텍스트 임베딩으로 무드 유사 아이템 필터링
  ├─ 기온 기반 시즌 필터링
  └─ 카테고리별 상위 7개 선정 (상의 / 하의 / 아우터 / 원피스)

Step 2  상의-하의 조합
  ├─ 색상 조화(LAB) + 임베딩 유사도 가중합
  └─ alpha_tb 비중으로 혼합 → 상위 L개 선정

Step 3  이너 후보 생성
  └─ Step 2 세트 + 원피스 통합

Step 4  아우터 매칭 + MMR 재랭킹
  ├─ 아우터 × 이너 조합별 점수 계산
  └─ MMR(Maximal Marginal Relevance)로 시각적/스타일적 유사 중복 제거 및 추천 다양성 최적화
```

### 피드백 학습 시스템 (Bandit 방식)

```
추천 시 (Explore):  θ_used = clip(θ_baseline + N(0, σ²), bounds)
피드백 시 (Exploit):
  좋아요 → θ_new = clip(θ + η · diff, bounds)
  싫어요 → θ_new = clip(θ - η · diff, bounds)
```

| 하이퍼파라미터 | 기본값 | 설명 |
|----------------|--------|------|
| alpha_tb | 0.65 | 상의-하의: 색상 vs 임베딩 비중 |
| alpha_oi | 0.70 | 아우터-이너: 색상 vs 임베딩 비중 |
| mmr_lambda | 0.75 | MMR: 품질 vs 다양성 균형 |
| eta | 0.10 | 피드백 학습률 |

---

## 7. 데이터베이스

**Neon PostgreSQL + pgvector**

| 테이블 | 설명 |
|--------|------|
| `closet_items` | 옷장 아이템 (속성 + CLIP 임베딩 vector(512)) |
| `recommendation_history` | 추천 이력 (피드백 학습용) |
| `user_hyperparams` | 사용자별 하이퍼파라미터 기준점 |

**데이터 현황**: 총 1,427개 (상의 418 / 아우터 394 / 하의 314 / 원피스 301)

---

## 8. 배포 구조

| 서비스 | 플랫폼 | 설명 |
|--------|--------|------|
| 프론트엔드 + API Routes | Vercel | main 브랜치 push 시 자동 배포 |
| ML 서버 | Railway (Docker) | `Dockerfile.ml` 멀티스테이지 빌드 |

ONNX Runtime 사용으로 Docker 이미지 **~0.8 GB** (PyTorch 대비 1/8 수준)

---

## 9. 로컬 실행 방법

### 프론트엔드

```bash
npm install
npm run dev        # http://localhost:3000
```

### ML 서버

```bash
cd ml-server
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Docker로 ML 서버 실행

```bash
docker build -f Dockerfile.ml -t fashioncody-ml:latest .
docker run -p 8000:8000 fashioncody-ml:latest
```

### 비전 파이프라인 단독 실행 (개발/테스트용)

```bash
python -m src.recognition.vision_pipeline \
    --image "옷사진.jpg" \
    --yolo  src/recognition/models/best.pt \
    --effnet src/recognition/models/efficientnet_kfashion_best.pt
```

---

## 10. 환경 변수

`.env.local` 파일 생성 후 설정:

```env
DATABASE_URL=postgresql://...

CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...

WEATHERAPI_KEY=...

ML_SERVER_URL=https://...railway.app
IMAGE_ANALYSIS_MODEL_URL=https://...railway.app
```

---

## 11. 팀 구성

**YBIGTA 28기 신입기수**

| 역할 | 담당자 |
|------|----------|
| **비전팀** | 변민주, 이근하 |
| **인텔리전스팀** | 박정현, 문형서, 황소현

---

<div align="center">

**YBIGTA 28th Project · FashionCody Team**

</div>
