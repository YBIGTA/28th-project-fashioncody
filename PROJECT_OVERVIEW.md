# OOTD AI — 프로젝트 전체 문서

AI 기반 의류 코디 추천 서비스. 사용자의 옷장 데이터와 무드/날씨를 기반으로 코디를 추천하고, 피드백을 통해 추천 알고리즘을 개인화한다.

---

## 1. 기술 스택 요약

| 영역 | 기술 | 역할 |
|------|------|------|
| 프론트엔드 | Next.js 16 (App Router) + React 19 + TypeScript | UI, SSR, API Routes |
| 스타일링 | Tailwind CSS 4 + Shadcn/ui (Radix) | 컴포넌트 라이브러리 |
| 데이터베이스 | Neon PostgreSQL + pgvector | 옷장 데이터, 벡터 유사도 검색 |
| ML 서버 | FastAPI + ONNX Runtime (Python) | 추천 알고리즘, 이미지 분석 |
| 이미지 저장 | Cloudinary CDN | 의류 이미지 업로드/서빙 |
| 날씨 API | WeatherAPI.com | 실시간 날씨 데이터 |
| 프론트 배포 | Vercel | Next.js 호스팅 |
| ML 배포 | Railway (Docker) | FastAPI 컨테이너 |
| 컨테이너화 | Docker + Dockerfile.ml | 멀티스테이지 빌드, 이미지 최적화 |

---

## 2. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│  사용자 브라우저 (/ootd)                                  │
│  - 무드 입력, 옷장 조회, 추천 요청, 좋아요/싫어요 피드백     │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│  Vercel (Next.js App Router)                            │
│                                                         │
│  API Routes:                                            │
│  ├─ /api/closet          GET     옷장 목록 조회           │
│  ├─ /api/closet/upload   POST    이미지 업로드 + DB 저장   │
│  ├─ /api/closet/[id]     GET/PUT/DELETE  개별 아이템 관리  │
│  ├─ /api/recommend       POST    추천 요청 (ML → fallback) │
│  ├─ /api/feedback        POST    좋아요/싫어요 → θ 갱신    │
│  └─ /api/weather         GET     WeatherAPI.com 프록시    │
└──┬───────────┬───────────┬──────────────────────────────┘
   │           │           │
   ▼           ▼           ▼
┌──────┐  ┌────────┐  ┌─────────────────────────────────┐
│ Neon │  │Cloudi- │  │  Railway (ML Server)             │
│ Post │  │nary    │  │                                   │
│ gres │  │CDN     │  │  FastAPI + ONNX Runtime          │
│      │  │        │  │  ├─ /recommend  코디 추천          │
│ 테이블│  │ 이미지 │  │  ├─ /analyze   이미지 속성 분석    │
│ ├ closet_items   │  │  └─ /health    헬스체크            │
│ ├ recommendation │  │                                   │
│ │ _history       │  │  모델:                             │
│ └ user_          │  │  ├ text_encoder.onnx (텍스트 임베딩)│
│   hyperparams    │  │  ├ item_encoder.onnx (아이템 임베딩)│
│                  │  │  └ efficientnet_kfashion.onnx     │
│ pgvector 확장    │  │    (12속성 분류)                    │
└──────┘  └────────┘  └─────────────────────────────────┘
```

---

## 3. 데이터베이스 (Neon PostgreSQL)

### 3.1 closet_items — 옷장 아이템

사용자의 옷장에 등록된 의류 아이템.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID (PK) | 자동 생성 |
| image_url | TEXT | Cloudinary 이미지 URL |
| image_id | VARCHAR(100) | CSV 원본 파일명 매핑 |
| image_vector | vector(512) | CLIP 이미지 임베딩 (pgvector) |
| category | VARCHAR(50) | top, bottom, outer, dress |
| sub_type | VARCHAR(100) | 니트웨어, 청바지, 코트 등 |
| color / sub_color | VARCHAR(50) | 주색상 / 보조색상 (한국어) |
| sleeve_length, length, fit, collar | VARCHAR(50) | 디자인 속성 |
| material, print, detail | JSONB | 다중라벨 속성 (배열) |
| season | VARCHAR(10)[] | spring, summer, fall, winter |
| created_at | TIMESTAMP | 생성일 |

**인덱스**: category, created_at, image_id, pgvector IVFFlat (코사인 유사도)

### 3.2 recommendation_history — 추천 이력

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID (PK) | 자동 생성 |
| mood | TEXT | 사용자 무드 텍스트 |
| weather_data | JSONB | 기온, 체감온도, 강수량 |
| recommended_items | JSONB | 추천 결과 |
| hyperparams_used | JSONB | 해당 추천에 사용된 θ_used |
| created_at | TIMESTAMP | 추천 시각 |

### 3.3 user_hyperparams — 하이퍼파라미터 기준점 (싱글톤)

| 컬럼 | 기본값 | 범위 | 설명 |
|------|--------|------|------|
| alpha_tb | 0.65 | [0.20, 0.95] | 상의-하의 색상 vs 임베딩 비중 |
| alpha_oi | 0.70 | [0.20, 0.95] | 아우터-이너 색상 vs 임베딩 비중 |
| mmr_lambda | 0.75 | [0.30, 0.95] | MMR 품질 vs 다양성 |
| beta_tb | 0.50 | [0.20, 0.80] | 아우터-상의 vs 아우터-하의 균형 |
| lambda_tbset | 0.15 | [0.00, 0.30] | 이너 세트 응집력 비중 |
| sigma | 0.05 | — | 탐색 노이즈 표준편차 |
| eta | 0.10 | — | 학습률 (피드백 시 갱신 폭) |

---

## 4. ML 서버 (FastAPI + ONNX Runtime)

### 4.1 개요

PyTorch 모델을 ONNX로 변환하여 경량 추론 서버 구축. Docker 이미지 약 0.8GB (PyTorch 사용 시 6.6GB).

**의존성**: fastapi, uvicorn, onnxruntime, numpy, Pillow, pydantic

### 4.2 모델 파일

| 파일 | 크기 | 역할 |
|------|------|------|
| text_encoder.onnx | 1.1 MB | 무드 텍스트 → 128차원 임베딩 |
| item_encoder.onnx | 0.6 MB | 아이템 속성(10개 feature) → 256차원 임베딩 |
| item_embs.npy | 83 MB | 사전 계산된 아이템 임베딩 (85,268 × 256) |
| artifacts_config.json | 8.2 MB | 모델 설정, 어휘 사전, 매핑 테이블 |
| efficientnet_kfashion.onnx | 16.4 MB | EfficientNet-B0 12속성 분류 |

### 4.3 엔드포인트

#### POST /recommend — 코디 추천

**요청**:
```json
{
  "user_context": {
    "text": "캐주얼한 데일리룩",
    "comment": "",
    "weather": { "temperature": 5, "feels_like": 2, "precipitation": 0 }
  },
  "closet_items": [{ "id": "uuid", "attributes": {...}, "season": [...] }],
  "top_k": 10,
  "alpha_tb": 0.65,
  "alpha_oi": 0.70,
  "mmr_lambda": 0.75,
  "beta_tb": 0.50,
  "lambda_tbset": 0.15
}
```

**응답**:
```json
{
  "selected_items": { "상의": [...], "하의": [...], "아우터": [...], "원피스": [...] },
  "recommendations": [
    { "outfit_type": "two_piece", "top_id": "...", "bottom_id": "...", "outer_id": "...", "score": 0.85, "reason": "..." }
  ]
}
```

#### POST /analyze — 이미지 속성 분석

의류 이미지를 입력하면 12개 속성을 예측:
- **단일라벨 (9개)**: 카테고리, 색상, 서브색상, 소매기장, 기장, 핏, 옷깃, 스타일, 서브스타일
- **다중라벨 (3개)**: 소재, 프린트, 디테일

### 4.4 추천 파이프라인 (4단계)

```
Step 1: 후보 선정
  - 텍스트 임베딩으로 무드 유사 아이템 필터링
  - 기온 기반 시즌 필터링
  - 카테고리별 상위 K개 선정 (상의/하의/아우터/원피스 각 7개)

Step 2: 상의-하의 조합 (build_top_bottom_sets_with_emb)
  - 모든 상의×하의 조합의 색상 조화(LAB) + 임베딩 유사도 계산
  - alpha_tb 비중으로 혼합 → 상위 L개 선정

Step 3: 이너 후보 생성 (build_inner_candidates)
  - Step 2의 상의-하의 세트 + 원피스를 이너 후보로 통합

Step 4: 아우터 매칭 + MMR (build_final_outfits_with_match → apply_mmr_reranking)
  - 아우터 × 이너 후보 조합별 점수 계산
  - alpha_oi, beta_tb, lambda_tbset 적용
  - MMR(Maximal Marginal Relevance)로 다양성 확보 (mmr_lambda)
```

### 4.5 색상 조화 점수 (color_harmony.py)

한국어 색상명 → CIE LAB 색공간 변환 후 조화 점수 산출:
- **유사색 조화**: 색상 거리 가까운 조합 (ΔE < 30)
- **보색 조화**: 색상각 차이 120°~180°
- **무채색 활용**: 블랙/화이트/그레이는 범용 매칭

---

## 5. 프론트엔드 (Next.js App Router)

### 5.1 페이지 구조

| 경로 | 컴포넌트 | 역할 |
|------|----------|------|
| `/` | page.tsx | 랜딩 페이지 (Hero, Features, Gallery, CTA) |
| `/ootd` | page.tsx | 메인 추천 인터페이스 |

### 5.2 /ootd 페이지 컴포넌트 구성

```
OotdPage
├── OotdHeader              (네비게이션)
├── DateWeatherHeader        (오늘 날짜 + 실시간 날씨)
├── OotdUploadCard           (이미지 업로드 드래그앤드롭)
├── UploadFormDialog         (옷장 추가: 카테고리/색상/유형 입력)
├── MoodInput                (무드 텍스트 입력)
├── ClosetGrid               (옷장 그리드, 카테고리 필터)
│   └── ClosetItemCard       (개별 아이템 카드)
├── SelectedItemsGrid        (Step 1: AI 선정 후보 아이템)
└── RecommendationResult     (Step 4: 최종 코디 추천 + 좋아요/싫어요)
```

### 5.3 사용자 플로우

```
1. /ootd 진입 → 날씨 자동 로드 + 옷장 목록 로드
2. (선택) 이미지 업로드 → "옷장에 추가하기" → 카테고리/색상 입력 → 등록
3. 무드 텍스트 입력 (예: "캐주얼한 데일리룩")
4. "추천받기" 클릭
5. AI 선정 아이템 표시 (상의/하의/아우터/원피스 각 7개)
6. 최종 코디 추천 10개 표시 (점수 + 사유)
7. 좋아요/싫어요 피드백 → 다음 추천에 반영
```

---

## 6. 하이퍼파라미터 피드백 학습 시스템

### 6.1 개요

Bandit 스타일 explore/exploit 패턴. 매 추천마다 하이퍼파라미터에 노이즈를 추가해 탐색하고, 사용자 피드백으로 기준점을 갱신.

### 6.2 추천 시 (explore)

```
θ_baseline ← DB(user_hyperparams)
θ_used = clip(θ_baseline + N(0, σ²), bounds)   // 가우시안 노이즈
→ ML 서버에 θ_used 전달
→ 추천 결과 + θ_used를 recommendation_history에 저장
→ recommendation_id 반환
```

### 6.3 피드백 시 (exploit)

```
POST /api/feedback { recommendation_id, liked: true/false }

θ_used ← recommendation_history[id].hyperparams_used
θ_current ← user_hyperparams
diff = θ_used - θ_current

좋아요:  θ_new = clip(θ + η · diff, bounds)   // θ_used 방향으로 이동
싫어요:  θ_new = clip(θ - η · diff, bounds)   // 반대 방향으로 이동

→ user_hyperparams 갱신
```

---

## 7. 이미지 업로드 파이프라인

```
사용자 이미지 선택
  ↓
UploadFormDialog (카테고리/색상/유형 입력)
  ↓
POST /api/closet/upload (FormData)
  ↓
┌─ Cloudinary 업로드 → image_url 획득
├─ 사용자 입력 속성 우선 적용 (ML fallback 대비)
├─ ML /analyze 호출 (EfficientNet) → 12속성 예측
├─ closet_items INSERT (속성 + 이미지 URL)
└─ (비동기) CLIP 서버 → image_vector 업데이트
```

---

## 8. 환경 변수

| 변수 | 용도 |
|------|------|
| DATABASE_URL | Neon PostgreSQL 연결 |
| CLOUDINARY_CLOUD_NAME / API_KEY / API_SECRET | 이미지 업로드 |
| WEATHERAPI_KEY | 날씨 API 인증 |
| ML_SERVER_URL | Railway ML 서버 주소 |
| IMAGE_ANALYSIS_MODEL_URL | /analyze 엔드포인트 주소 |
| CLIP_MODEL_URL | CLIP 텍스트 인코딩 서버 |

---

## 9. 배포

### 9.1 Vercel (프론트엔드 + API Routes)

- GitHub 연동 자동 배포 (main 브랜치 push 시)
- 환경 변수는 Vercel Dashboard에서 설정
- URL: `https://ootdai.vercel.app`

### 9.2 Railway (ML 서버)

- GitHub 연동, `Dockerfile.ml` 기반 빌드
- 멀티스테이지 Docker: builder(pip install) → runtime(venv 복사)
- ONNX Runtime 사용으로 이미지 ~0.8GB (PyTorch 시 6.6GB로 무료 4GB 제한 초과)
- URL: `https://ootdai-production.up.railway.app`

### 9.3 Dockerfile.ml 구조

```dockerfile
FROM python:3.10-slim AS builder
  → venv 생성 + pip install requirements.txt

FROM python:3.10-slim
  → venv 복사
  → app 코드 + ONNX 모델 파일 COPY
  → uvicorn 실행 (PORT 8000)
```

### 9.4 Docker 활용

#### 9.4.1 개요

ML 서버는 Docker 컨테이너로 패키징되어 Railway에 배포됩니다. 멀티스테이지 빌드를 통해 이미지 크기를 최소화하고, ONNX Runtime을 사용하여 PyTorch 대비 약 8배 작은 이미지(~0.8GB)를 생성합니다.

**Docker의 역할**:
- ML 서버의 일관된 실행 환경 보장
- 의존성 격리 및 버전 관리
- Railway 배포 자동화
- 로컬 개발 환경과 프로덕션 환경의 일치

#### 9.4.2 Dockerfile.ml 상세

**멀티스테이지 빌드 구조**:

```dockerfile
# Stage 1: Builder
FROM python:3.10-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY ml-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# 애플리케이션 코드 복사
COPY ml-server/app/ ./app/

# ONNX 모델 파일 복사
COPY model/artifacts_config.json ./model/artifacts_config.json
COPY model/text_encoder.onnx     ./model/text_encoder.onnx
COPY model/item_encoder.onnx     ./model/item_encoder.onnx
COPY model/item_embs.npy         ./model/item_embs.npy

# 환경 변수 설정
ENV ARTIFACTS_PATH=/app/model/artifacts_config.json
ENV EFFNET_MODEL_PATH=/app/app/efficientnet_kfashion.onnx

EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

**최적화 포인트**:
- **멀티스테이지 빌드**: 빌드 의존성과 런타임 분리로 최종 이미지 크기 감소
- **venv 활용**: 시스템 Python과 격리된 가상환경 사용
- **--no-cache-dir**: pip 캐시 제외로 이미지 크기 감소
- **python:3.10-slim**: 최소한의 베이스 이미지 사용

#### 9.4.3 .dockerignore

빌드 컨텍스트에서 불필요한 파일 제외:

```
node_modules
.next
.git
__pycache__
*.pyc
data/
scripts/
src/
public/
*.md
.env*
.claude/
```

**효과**: Docker 빌드 속도 향상 및 컨텍스트 크기 감소

#### 9.4.4 로컬 개발에서 Docker 사용

**이미지 빌드**:
```bash
docker build -f Dockerfile.ml -t ootd-ml-server:latest .
```

**컨테이너 실행**:
```bash
docker run -p 8000:8000 \
  -e PORT=8000 \
  -e ARTIFACTS_PATH=/app/model/artifacts_config.json \
  -e EFFNET_MODEL_PATH=/app/app/efficientnet_kfashion.onnx \
  ootd-ml-server:latest
```

**개발 모드 (볼륨 마운트)**:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/ml-server/app:/app/app \
  -v $(pwd)/model:/app/model \
  ootd-ml-server:latest
```

#### 9.4.5 Railway 배포

Railway는 GitHub 저장소와 연동되어 자동으로 Docker 이미지를 빌드하고 배포합니다:

1. **자동 빌드**: `main` 브랜치에 push 시 자동 빌드 트리거
2. **Dockerfile 인식**: 프로젝트 루트의 `Dockerfile.ml` 자동 감지
3. **환경 변수**: Railway Dashboard에서 설정한 환경 변수 자동 주입
4. **포트 바인딩**: Railway가 자동으로 `PORT` 환경 변수 설정

**배포 플로우**:
```
GitHub Push → Railway Webhook → Docker Build → Container Deploy → Health Check
```

#### 9.4.6 이미지 크기 최적화

| 구성 요소 | PyTorch 사용 시 | ONNX Runtime 사용 시 |
|----------|----------------|---------------------|
| 베이스 이미지 | python:3.10-slim (~150MB) | python:3.10-slim (~150MB) |
| ML 라이브러리 | PyTorch (~6.5GB) | ONNX Runtime (~200MB) |
| 애플리케이션 코드 | ~10MB | ~10MB |
| 모델 파일 | ~110MB | ~110MB |
| **총합** | **~6.6GB** | **~0.8GB** |

**최적화 전략**:
- ONNX Runtime 사용으로 PyTorch 의존성 제거
- 멀티스테이지 빌드로 빌드 도구 제외
- 불필요한 파일 `.dockerignore`로 제외
- slim 베이스 이미지 사용

---

## 10. 프로젝트 디렉토리 구조

```
ootd_ai/
├── src/
│   ├── app/
│   │   ├── page.tsx                  # 랜딩 페이지
│   │   ├── ootd/page.tsx             # 메인 추천 페이지
│   │   └── api/
│   │       ├── closet/               # 옷장 CRUD
│   │       │   ├── route.ts
│   │       │   ├── [id]/route.ts
│   │       │   └── upload/route.ts
│   │       ├── recommend/route.ts    # 추천 API
│   │       ├── feedback/route.ts     # 피드백 API
│   │       └── weather/route.ts      # 날씨 프록시
│   ├── components/
│   │   ├── ui/                       # Shadcn 기본 컴포넌트
│   │   ├── landing/                  # 랜딩 페이지 컴포넌트
│   │   └── ootd/                     # 추천 페이지 컴포넌트
│   └── lib/
│       ├── db/
│       │   ├── neon-client.ts        # DB 연결
│       │   ├── repository.ts         # Repository 팩토리
│       │   ├── postgres-repository.ts # PostgreSQL 구현
│       │   └── hyperparams-repository.ts # 하이퍼파라미터 DB
│       ├── types/
│       │   ├── closet.ts             # 도메인 타입
│       │   └── closet-view.ts        # 뷰 타입
│       └── utils.ts                  # Tailwind 유틸리티
├── ml-server/
│   ├── app/
│   │   ├── main.py                   # FastAPI 엔트리포인트
│   │   ├── model_loader.py           # ONNX 모델 로딩
│   │   ├── predictor.py              # 추천 파이프라인
│   │   ├── color_harmony.py          # LAB 색상 조화
│   │   ├── match_harmony.py          # 임베딩 매칭 + MMR
│   │   ├── efficientnet_classifier.py # 이미지 분류
│   │   └── efficientnet_kfashion.onnx # EfficientNet 모델
│   └── requirements.txt
├── model/                            # 추천 모델 아티팩트
│   ├── text_encoder.onnx
│   ├── item_encoder.onnx
│   ├── item_embs.npy
│   └── artifacts_config.json
├── database/
│   └── schema.sql                    # 전체 DB 스키마
├── data/
│   ├── items.csv                     # 의류 데이터셋
│   └── images/                       # 의류 이미지 (1,425개)
├── scripts/
│   ├── seed.ts                       # DB 시딩
│   ├── bulk-seed.ts                  # 대량 시딩
│   └── convert_to_onnx.py           # PyTorch → ONNX 변환
├── Dockerfile.ml                     # ML 서버 Docker 빌드 파일
├── .dockerignore                     # Docker 빌드 제외 파일
├── package.json
├── tsconfig.json
└── next.config.ts
```

---

## 11. 데이터 현황

| 항목 | 수치 |
|------|------|
| 전체 아이템 | 1,427개 |
| 카테고리 분포 | 상의 418, 아우터 394, 하의 314, 원피스 301 |
| 시즌 분포 | 여름 733, 봄 612, 가을 596, 겨울 454 |
| 이미지 소스 | CSV import 1,425 + 직접 업로드 2 |
| 색상 NULL | 0개 (EfficientNet /analyze로 backfill 완료) |
