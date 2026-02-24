# 프로젝트 전체 파이프라인 및 진행 상황

## 📊 전체 파이프라인

### 1. 옷장 아이템 등록 파이프라인

```
[사용자]
  ↓ 이미지 업로드
[프론트엔드] POST /api/closet/upload
  ↓
[이미지 스토리지] Cloudinary/S3/Vercel Blob
  ↓ 이미지 URL 반환
[이미지 분석 모델 서버] (팀원)
  → 속성 추출 (dummy.json 형식)
  → category, sub_type, color, material, print, detail 등
  ↓
[CLIP Image Encoder] (팀원)
  → 이미지 벡터 생성 (512차원 등)
  ↓
[PostgreSQL]
  → 메타데이터 저장 (속성 정보)
  ↓
[Vector DB] Pinecone/Weaviate/Chroma
  → 벡터 + 아이템 ID 저장
  ↓
[완료] 옷장에 아이템 추가됨
```

### 2. 코디 추천 파이프라인

```
[사용자]
  ↓ 입력: "미니멀 데이트" + 날씨 정보
[프론트엔드] POST /api/recommend
  ↓
[LLM 키워드 확장] (팀원, 선택)
  → "미니멀 데이트" → "심플, 깔끔, 로맨틱..."
  ↓
[CLIP Text Encoder] (팀원)
  → 텍스트 벡터 생성
  ↓
[Vector DB] 유사도 검색
  → Cosine Similarity로 유사한 옷장 아이템 찾기
  → Top-K 아이템 ID 반환
  ↓
[PostgreSQL]
  → 아이템 메타데이터 조회
  ↓
[모델 서버] (팀원)
  → 최종 점수 계산 (MLP)
  → 조합 추천 (상의+하의+아우터)
  ↓
[프론트엔드]
  → 추천 결과 표시
```

---

## ✅ 완료된 작업

### 프론트엔드
- [x] 랜딩 페이지 (Pinterest 스타일)
- [x] OOTD 페이지 UI
- [x] 옷장 관리 UI
- [x] 추천 결과 표시 UI
- [x] 날씨 API 연동 (WeatherAPI.com)

### 백엔드 인프라
- [x] 타입 정의 (`src/lib/types/closet.ts`)
- [x] 데이터베이스 스키마 설계 (`database/schema.sql`)
- [x] 옷장 관리 API (`/api/closet`)
- [x] 추천 API 구조 (`/api/recommend`)
- [x] Vector DB 클라이언트 구조 (`src/lib/vector-db/client.ts`)
- [x] 데이터베이스 레포지토리 인터페이스
- [x] dummy.json 데이터 변환 유틸리티

### 문서화
- [x] 프로젝트 파이프라인 문서
- [x] 내가 담당할 부분 정리
- [x] Vector DB 사용 흐름

---

## 🚧 진행 중 / 해야 할 작업

### Phase 1: 데이터베이스 구축 (최우선)

#### 1.1 PostgreSQL 설정
- [ ] PostgreSQL 데이터베이스 생성 (Supabase/Neon/자체 호스팅)
- [ ] 스키마 마이그레이션 실행 (`database/schema.sql`)
- [ ] 연결 문자열 환경 변수 설정
- [ ] 실제 PostgreSQL 레포지토리 구현 (`src/lib/db/postgres-repository.ts`)

#### 1.2 Vector DB 설정
- [ ] Vector DB 선택 (Pinecone/Weaviate/Chroma)
- [ ] Vector DB 인덱스 생성
- [ ] API 키 및 환경 변수 설정
- [ ] Vector DB 클라이언트 실제 구현 (`src/lib/vector-db/client.ts`)

#### 1.3 이미지 스토리지 설정
- [ ] 이미지 스토리지 선택 (Cloudinary/S3/Vercel Blob)
- [ ] 업로드 API 구현
- [ ] 환경 변수 설정

---

### Phase 2: 모델 서버 통합

#### 2.1 팀원과 API 스펙 협의
- [ ] 이미지 분석 모델 API 스펙
  - 엔드포인트: `POST /analyze`
  - 입력: 이미지 파일
  - 출력: dummy.json 형식의 속성 정보
- [ ] CLIP Image Encoder API 스펙
  - 엔드포인트: `POST /encode-image`
  - 입력: 이미지 파일
  - 출력: `{ vector: number[] }`
- [ ] CLIP Text Encoder API 스펙
  - 엔드포인트: `POST /encode-text`
  - 입력: `{ text: string }`
  - 출력: `{ vector: number[] }`
- [ ] 추천 모델 서버 API 스펙
  - 엔드포인트: `POST /recommend`
  - 입력: `{ user_context, closet_items, top_k }`
  - 출력: `{ recommendations: [...] }`

#### 2.2 모델 서버 통합
- [ ] 이미지 분석 모델 서버 연동 (`src/app/api/closet/upload/route.ts`)
- [ ] CLIP 모델 서버 연동
- [ ] 추천 모델 서버 연동 (`src/app/api/recommend/route.ts`)
- [ ] 에러 처리 및 재시도 로직
- [ ] 타임아웃 설정

---

### Phase 3: 옷장 관리 기능 완성

#### 3.1 이미지 업로드 완성
- [ ] 프론트엔드 업로드 UI 연동
- [ ] 이미지 스토리지 업로드
- [ ] 이미지 분석 모델 호출
- [ ] CLIP 벡터 생성
- [ ] PostgreSQL 저장
- [ ] Vector DB 저장
- [ ] 업로드 진행 상태 표시

#### 3.2 옷장 조회/수정/삭제
- [ ] 프론트엔드에서 `/api/closet` 호출
- [ ] 옷장 그리드에 실제 데이터 표시
- [ ] 아이템 수정 기능
- [ ] 아이템 삭제 기능 (PostgreSQL + Vector DB)

---

### Phase 4: 추천 기능 완성

#### 4.1 추천 API 완성
- [ ] 프론트엔드에서 `/api/recommend` 호출
- [ ] Vector DB 검색 구현
- [ ] 모델 서버 통합
- [ ] 결과 포맷팅

#### 4.2 추천 결과 표시
- [ ] 실제 추천 결과 UI에 표시
- [ ] 로딩 상태 처리
- [ ] 에러 처리

---

### Phase 5: 추가 기능

#### 5.1 피드백 기능
- [ ] 피드백 API 구현 (`POST /api/recommend/:id/feedback`)
- [ ] 피드백 데이터베이스 저장
- [ ] 프론트엔드 피드백 버튼 연동

#### 5.2 추천 히스토리
- [ ] 히스토리 조회 API
- [ ] 히스토리 UI

#### 5.3 사용자 인증 (선택)
- [ ] 인증 시스템 구현
- [ ] 사용자별 옷장 분리

---

## 🔧 기술 스택

### 프론트엔드
- Next.js 14 (App Router)
- React
- TypeScript
- Tailwind CSS
- Shadcn/ui

### 백엔드
- Next.js API Routes
- PostgreSQL (Supabase/Neon)
- Vector DB (Pinecone/Weaviate/Chroma)

### 스토리지
- 이미지: Cloudinary / AWS S3 / Vercel Blob

### 모델 서버 (팀원)
- 이미지 분석 모델
- CLIP Image/Text Encoder
- 추천 MLP 모델

---

## 📋 환경 변수 설정

### 필요한 환경 변수

```env
# 데이터베이스
DATABASE_URL=postgresql://...
POSTGRES_HOST=...
POSTGRES_USER=...
POSTGRES_PASSWORD=...
POSTGRES_DB=...

# Vector DB
VECTOR_DB_PROVIDER=pinecone
VECTOR_DB_API_KEY=...
VECTOR_DB_INDEX_NAME=...

# 이미지 스토리지
CLOUDINARY_URL=...
# 또는
AWS_S3_BUCKET=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# 모델 서버 (팀원과 협의)
IMAGE_ANALYSIS_MODEL_URL=http://localhost:8001
CLIP_MODEL_URL=http://localhost:8002
ML_SERVER_URL=http://localhost:8000

# 날씨 API
WEATHERAPI_KEY=...
```

---

## 🤝 팀원과 협의 필요 사항

### 1. 모델 서버 API 스펙
- [ ] 각 모델 서버의 엔드포인트 URL
- [ ] 입력/출력 형식
- [ ] 인증 방식 (API 키 등)
- [ ] 에러 응답 형식

### 2. 벡터 차원 수
- [ ] CLIP 이미지 벡터 차원 (예: 512, 768)
- [ ] CLIP 텍스트 벡터 차원

### 3. 데이터 형식
- [ ] 추천 결과 형식
- [ ] 아이템 ID 형식 (UUID 등)

### 4. 배포 환경
- [ ] 모델 서버 배포 위치
- [ ] 내부 네트워크 vs 외부 API

---

## 📅 우선순위별 작업 계획

### Week 1: 데이터베이스 구축
1. PostgreSQL 설정 및 스키마 마이그레이션
2. Vector DB 선택 및 설정
3. 실제 DB 레포지토리 구현

### Week 2: 모델 서버 통합
1. 팀원과 API 스펙 협의
2. 모델 서버 연동
3. 테스트

### Week 3: 옷장 관리 완성
1. 이미지 업로드 파이프라인 완성
2. 옷장 조회/수정/삭제
3. 프론트엔드 연동

### Week 4: 추천 기능 완성
1. Vector DB 검색 구현
2. 추천 API 완성
3. 프론트엔드 연동
4. 테스트 및 버그 수정

---

## 🎯 최종 목표

**다른 팀원의 모델을 붙여서 완전한 서비스 완성**

1. ✅ 사용자가 옷장에 이미지 업로드
2. ✅ 이미지 → 벡터 변환 (팀원 모델)
3. ✅ Vector DB에 저장
4. ✅ 사용자가 추천 요청
5. ✅ Vector DB 검색 → 모델 서버 호출 (팀원 모델)
6. ✅ 결과를 UI에 표시

**내 역할: 1-6번의 인프라와 통합 부분 전부 담당**
