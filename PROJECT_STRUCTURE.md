# 프로젝트 전체 구성 정리

## 📋 프로젝트 개요

**OOTD AI (Outfit Of The Day AI)** - AI 기반 코디 추천 시스템

사용자의 옷장 이미지를 분석하고, 날씨와 상황(분위기)을 고려하여 최적의 코디를 추천하는 웹 애플리케이션입니다.

---

## 🏗️ 프로젝트 구조

```
ybigta_newbieproject/
└── ootd_ai/                          # 메인 프로젝트 디렉토리
    ├── src/                          # Next.js 프론트엔드 & 백엔드
    │   ├── app/                      # Next.js App Router
    │   │   ├── api/                  # API 라우트
    │   │   │   ├── closet/          # 옷장 관리 API
    │   │   │   │   ├── route.ts     # GET, POST /api/closet
    │   │   │   │   ├── [id]/        # 개별 아이템 조회/수정/삭제
    │   │   │   │   └── upload/      # 이미지 업로드
    │   │   │   ├── recommend/        # 추천 API
    │   │   │   │   └── route.ts     # POST /api/recommend
    │   │   │   └── weather/         # 날씨 API
    │   │   ├── ootd/                 # OOTD 페이지
    │   │   │   └── page.tsx
    │   │   ├── page.tsx              # 랜딩 페이지
    │   │   ├── layout.tsx            # 레이아웃
    │   │   └── globals.css           # 전역 스타일
    │   ├── components/               # React 컴포넌트
    │   │   ├── landing/             # 랜딩 페이지 컴포넌트
    │   │   │   ├── Header.tsx
    │   │   │   ├── HeroSection.tsx
    │   │   │   ├── FeaturesSection.tsx
    │   │   │   ├── GallerySection.tsx
    │   │   │   ├── CTASection.tsx
    │   │   │   └── Footer.tsx
    │   │   ├── ootd/                 # OOTD 페이지 컴포넌트
    │   │   │   ├── OotdHeader.tsx
    │   │   │   ├── DateWeatherHeader.tsx
    │   │   │   ├── OotdUploadCard.tsx
    │   │   │   ├── MoodInput.tsx
    │   │   │   ├── ClosetGrid.tsx
    │   │   │   ├── ClosetItemCard.tsx
    │   │   │   ├── ClosetItemDialog.tsx
    │   │   │   └── RecommendationResult.tsx
    │   │   └── ui/                   # Shadcn/ui 컴포넌트
    │   │       ├── button.tsx
    │   │       ├── card.tsx
    │   │       ├── input.tsx
    │   │       ├── dialog.tsx
    │   │       └── ...
    │   └── lib/                      # 유틸리티 & 타입
    │       ├── db/                   # 데이터베이스 레포지토리
    │       │   ├── repository.ts    # 레포지토리 팩토리
    │       │   ├── closet-repository.ts  # 인터페이스
    │       │   ├── postgres-repository.ts # PostgreSQL 구현
    │       │   └── neon-client.ts    # Neon DB 클라이언트
    │       ├── types/                # TypeScript 타입 정의
    │       │   ├── closet.ts        # ClosetItem 타입
    │       │   └── closet-view.ts    # UI용 뷰 타입
    │       └── utils.ts              # 유틸리티 함수
    │
    ├── ml-server/                    # ML 추론 서버 (FastAPI)
    │   ├── app/
    │   │   ├── __init__.py
    │   │   ├── main.py               # FastAPI 앱 & 엔드포인트
    │   │   ├── model_loader.py       # artifacts.pt 로더
    │   │   └── predictor.py          # 추론 로직
    │   └── requirements.txt           # Python 의존성
    │
    ├── ml-recommendation/            # ML 모델 학습 코드
    │   ├── train/
    │   │   ├── preprocess.py        # 데이터 전처리
    │   │   ├── tokenizer.py         # 토크나이저 설정
    │   │   └── train_model.py       # 모델 학습
    │   ├── README.md
    │   ├── EXPLANATION.md
    │   ├── MODEL_USAGE.md
    │   ├── PERFORMANCE_ANALYSIS.md
    │   └── requirements.txt
    │
    ├── database/
    │   └── schema.sql                # PostgreSQL 스키마
    │
    ├── model/                        # 학습된 모델 파일
    │   ├── artifacts.pt              # 모델 아티팩트 (인코더 + vocab)
    │   └── train_model_i.ipynb       # 학습 노트북
    │
    ├── scripts/
    │   └── seed.ts                   # 시드 데이터 스크립트
    │
    ├── public/                       # 정적 파일
    │   └── *.svg
    │
    ├── package.json                  # Node.js 의존성
    ├── tsconfig.json                 # TypeScript 설정
    ├── next.config.ts                # Next.js 설정
    ├── components.json               # Shadcn/ui 설정
    ├── eslint.config.mjs             # ESLint 설정
    ├── postcss.config.mjs            # PostCSS 설정
    │
    └── 문서 파일들/
        ├── README.md                 # 기본 README
        ├── PROJECT_STATUS.md         # 프로젝트 진행 상황
        ├── PROJECT_PIPELINE.md       # 파이프라인 설명
        ├── PIPELINE_STATUS.md        # 파이프라인 상태
        ├── TODO.md                   # 할 일 목록
        ├── IMPLEMENTATION_GUIDE.md   # 구현 가이드
        ├── VECTOR_DB_FLOW.md         # Vector DB 흐름
        ├── MY_TASKS.md               # 내 작업 목록
        └── 수정_문형서.md            # 최근 수정 내역
```

---

## 🛠️ 기술 스택

### 프론트엔드
- **Next.js 16.1.6** (App Router)
- **React 19.2.3**
- **TypeScript 5**
- **Tailwind CSS 4**
- **Shadcn/ui** (UI 컴포넌트 라이브러리)
- **Lucide React** (아이콘)
- **Sonner** (토스트 알림)

### 백엔드
- **Next.js API Routes** (프론트엔드와 통합)
- **FastAPI** (ML 서버, Python)
- **PostgreSQL** (Neon Database)
- **pgvector** (벡터 검색 확장)

### ML/AI
- **PyTorch 2.0+** (모델 학습 및 추론)
- **Transformers** (한국어 토크나이저)
- **CLIP** (이미지/텍스트 임베딩, 팀원 담당)

### 스토리지
- **Vercel Blob** (이미지 스토리지)
- **Vector DB** (옵션: Pinecone/Weaviate/Chroma, 현재는 pgvector 사용)

### 외부 API
- **WeatherAPI.com** (날씨 정보)

---

## 📊 시스템 아키텍처

### 1. 옷장 아이템 등록 파이프라인

```
[사용자] 이미지 업로드
  ↓
[프론트엔드] POST /api/closet/upload
  ↓
[이미지 스토리지] Vercel Blob
  ↓ 이미지 URL 반환
[이미지 분석 모델 서버] (팀원)
  → 속성 추출 (category, sub_type, color, material 등)
  ↓
[CLIP Image Encoder] (팀원)
  → 이미지 벡터 생성 (512차원)
  ↓
[PostgreSQL + pgvector]
  → 메타데이터 + 벡터 저장
  ↓
[완료] 옷장에 아이템 추가됨
```

### 2. 코디 추천 파이프라인

```
[사용자] 입력: "미니멀 데이트" + 날씨 정보
  ↓
[프론트엔드] POST /api/recommend
  ↓
[CLIP Text Encoder] (팀원, 선택)
  → 텍스트 벡터 생성
  ↓
[PostgreSQL pgvector] 유사도 검색
  → Cosine Similarity로 유사한 옷장 아이템 찾기
  → Top-K 아이템 ID 반환 (선택적 narrowing)
  ↓
[ML 서버] FastAPI /recommend
  → 최종 점수 계산 (MLP)
  → 조합 추천 (상의+하의+아우터 또는 원피스)
  ↓
[프론트엔드]
  → 추천 결과 표시
```

---

## 🗄️ 데이터베이스 스키마

### `closet_items` 테이블
- **id**: UUID (Primary Key)
- **image_url**: TEXT (이미지 URL)
- **image_vector**: vector(512) (CLIP 이미지 벡터, pgvector)
- **category**: VARCHAR(50) (top, bottom, outer, shoes, bag, accessory, dress)
- **속성 정보**: sub_type, color, material, print, detail 등
- **사용자 정보**: name, tags, season
- **메타데이터**: created_at, updated_at

### `recommendation_history` 테이블
- **id**: UUID (Primary Key)
- **입력 정보**: mood, comment, weather_data
- **추천 결과**: recommended_items (JSONB)
- **피드백**: feedback (JSONB)
- **메타데이터**: created_at

---

## 🔌 API 엔드포인트

### 옷장 관리 API (`/api/closet`)

#### `GET /api/closet`
- 옷장 아이템 목록 조회
- Query: `?category=top` (선택)

#### `POST /api/closet`
- 옷장 아이템 생성
- Body: `{ imageUrl, attributes, name, tags, season }`

#### `GET /api/closet/[id]`
- 개별 아이템 조회

#### `POST /api/closet/upload`
- 이미지 업로드 및 분석
- FormData: `image` 파일

### 추천 API (`/api/recommend`)

#### `POST /api/recommend`
- 코디 추천 요청
- Body: `{ mood, comment, temperature, feelsLike, precipitation }`
- Response: `{ recommendations: [...] }`

### 날씨 API (`/api/weather`)

#### `GET /api/weather`
- 현재 날씨 정보 조회
- Response: `{ temperature, feelsLike, precipitation }`

### ML 서버 API (`http://localhost:8000`)

#### `POST /recommend`
- ML 추론 엔드포인트
- Body: `{ user_context, closet_items, top_k }`
- Response: `{ recommendations: [...] }`

#### `GET /health`
- 서버 상태 확인

---

## 📁 주요 파일 설명

### 프론트엔드

#### `src/app/page.tsx`
- 랜딩 페이지 (Pinterest 스타일)
- Hero, Features, Gallery, CTA 섹션

#### `src/app/ootd/page.tsx`
- 메인 OOTD 추천 페이지
- 날씨 정보, 옷장 그리드, 추천 결과 표시
- 상태 관리: mood, comment, closetItems, recommendations

#### `src/app/api/recommend/route.ts`
- 추천 API 핵심 로직
- ML 서버 호출 + Fallback 규칙 기반 추천
- Vector 검색 (선택적)

#### `src/app/api/closet/route.ts`
- 옷장 CRUD API
- PostgreSQL 레포지토리 사용

### 백엔드/ML

#### `ml-server/app/main.py`
- FastAPI 애플리케이션
- `/recommend` 엔드포인트
- artifacts.pt 로딩 (startup)

#### `ml-server/app/model_loader.py`
- `artifacts.pt` 파일 로딩
- TextEncoder, ItemEncoder 재구성
- Vocab/maps 로딩

#### `ml-server/app/predictor.py`
- 추론 로직
- 옷장 아이템 → 학습 feature map 인덱싱
- 스타일 매핑 우선순위: `스타일` > `style` > `서브스타일` > `sub_style`
- 추천 타입: `two_piece` (상+하+아우터) / `dress` (원피스+아우터)

### 데이터베이스

#### `src/lib/db/repository.ts`
- 레포지토리 팩토리
- 현재: `PostgresClosetRepository` 사용

#### `src/lib/db/postgres-repository.ts`
- PostgreSQL 실제 구현
- CRUD + 벡터 검색 (pgvector)

#### `database/schema.sql`
- PostgreSQL 스키마 정의
- pgvector 확장 활성화
- 인덱스 설정

---

## 🎯 주요 기능

### ✅ 완료된 기능

1. **랜딩 페이지**
   - Pinterest 스타일 UI
   - 반응형 디자인

2. **OOTD 페이지**
   - 날씨 정보 표시 (WeatherAPI)
   - 옷장 그리드 (이미지 카드)
   - Mood 입력 (텍스트)
   - 추천 결과 표시 (상+하+아우터 또는 원피스)

3. **옷장 관리**
   - 이미지 업로드 UI
   - 옷장 아이템 조회
   - 카테고리별 필터링

4. **ML 추천 시스템**
   - FastAPI ML 서버
   - artifacts.pt 기반 추론
   - 원피스/투피스 타입 분리
   - Fallback 규칙 기반 추천

5. **데이터베이스**
   - PostgreSQL 스키마 설계
   - pgvector 벡터 검색
   - 레포지토리 패턴 구현

### 🚧 진행 중 / 예정 기능

1. **이미지 분석 통합**
   - 이미지 분석 모델 서버 연동 (팀원)
   - CLIP Image Encoder 연동 (팀원)

2. **Vector DB 검색 강화**
   - CLIP Text Encoder 연동 (팀원)
   - 벡터 검색 최적화

3. **옷장 관리 완성**
   - 아이템 수정/삭제
   - 상세 정보 편집

4. **추천 기능 개선**
   - 피드백 시스템
   - 추천 히스토리
   - 개인화 추천

---

## 🔧 환경 변수 설정

### `.env.local` (필수)

```env
# 데이터베이스
DATABASE_URL=postgresql://user:password@host:port/database

# ML 서버
ML_SERVER_URL=http://localhost:8000
CLIP_MODEL_URL=http://localhost:8002

# 이미지 스토리지
BLOB_READ_WRITE_TOKEN=...

# 날씨 API
WEATHERAPI_KEY=...

# 모델 경로 (ML 서버)
ARTIFACTS_PATH=../model/artifacts.pt
```

---

## 🚀 실행 방법

### 1. 프론트엔드 (Next.js)

```bash
cd ootd_ai
npm install
npm run dev
```

서버: `http://localhost:3000`

### 2. ML 서버 (FastAPI)

```bash
cd ootd_ai/ml-server
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

서버: `http://localhost:8000`

### 3. 데이터베이스 마이그레이션

```bash
# PostgreSQL에 스키마 적용
psql $DATABASE_URL -f database/schema.sql
```

---

## 📝 최근 주요 수정 사항

### 수정_문형서.md 참고

1. **모델 포맷 불일치 해결**
   - `artifacts.pt` 기반 로더 구현
   - TextEncoder, ItemEncoder 재구성

2. **API 스키마 정렬**
   - Next API ↔ ML 서버 스키마 일치
   - 요청/응답 형식 통일

3. **원피스 처리 분리**
   - `dress` 전용 추천 타입 도입
   - UI 렌더링 분리

4. **벡터 검색 Optional화**
   - CLIP 실패 시에도 ML 추론 진행
   - Fallback 강화

---

## 🤝 팀원 협업 구조

### 내 역할
- 프론트엔드 개발 (Next.js)
- 백엔드 API 개발 (Next.js API Routes)
- 데이터베이스 설계 및 구현 (PostgreSQL)
- ML 서버 통합 (FastAPI 연동)
- Vector DB 통합 (pgvector)

### 팀원 역할
- 이미지 분석 모델 서버
- CLIP Image/Text Encoder 서버
- ML 모델 학습 (ml-recommendation)

---

## 📚 참고 문서

- `PROJECT_STATUS.md`: 프로젝트 진행 상황
- `PROJECT_PIPELINE.md`: 전체 파이프라인 설명
- `TODO.md`: 할 일 목록
- `IMPLEMENTATION_GUIDE.md`: 구현 가이드
- `수정_문형서.md`: 최근 수정 내역 상세

---

## 🔍 코드베이스 특징

1. **타입 안정성**: TypeScript로 전체 타입 정의
2. **레포지토리 패턴**: 데이터베이스 추상화
3. **Fallback 전략**: ML 실패 시 규칙 기반 추천
4. **모듈화**: 컴포넌트, API, ML 서버 분리
5. **확장 가능**: Vector DB, 이미지 분석 모델 쉽게 통합 가능

---

## ⚠️ 알려진 이슈

1. **한글 인코딩**: 일부 파일에서 한글 인코딩 깨짐 (실행에는 문제 없음)
2. **모델 파일**: `model/` 디렉토리는 untracked 상태일 수 있음
3. **CLIP 서버**: 현재는 선택적 통합, 실패 시 전체 아이템으로 추론

---

## 🎯 향후 개선 방향

1. **성능 최적화**
   - 벡터 검색 캐싱
   - 추천 결과 캐싱

2. **사용자 경험**
   - 로딩 상태 개선
   - 에러 메시지 개선
   - 피드백 시스템

3. **개인화**
   - 사용자별 옷장 분리
   - 추천 히스토리 기반 개인화

4. **모니터링**
   - 로깅 시스템
   - 성능 모니터링
   - 에러 추적
