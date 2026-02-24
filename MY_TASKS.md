# 내가 담당할 부분 정리

## 🎯 역할 분담

### 다른 팀원 담당
- ❌ ML 모델 학습 코드
- ❌ MLP 모델 구조
- ❌ 대조학습 알고리즘
- ❌ 모델 추론 로직 (학습된 모델로 점수 예측)

### 내가 담당할 부분 ✅
- ✅ 데이터베이스 설계 및 구축
- ✅ API 서버 (모델 서버와 통신)
- ✅ 프론트엔드 (이미 완성)
- ✅ 데이터 파이프라인 (이미지 → 벡터 → Vector DB)
- ✅ 모델 서버와의 통합
- ✅ 인프라/배포
- ✅ 전체 시스템 연결

---

## 📋 구체적 작업 목록

### 1. 데이터베이스 설계 및 구축

#### 1.1 옷장 데이터 저장
```sql
-- 옷장 아이템 테이블
CREATE TABLE closet_items (
  id UUID PRIMARY KEY,
  user_id UUID,
  name VARCHAR(255),
  category VARCHAR(50), -- top, bottom, outer, etc.
  color VARCHAR(50),
  season VARCHAR(50)[],
  tags VARCHAR(50)[],
  image_url TEXT,
  image_vector FLOAT[], -- CLIP 이미지 벡터 (사전 계산)
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

-- 사용자 테이블
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE,
  name VARCHAR(255),
  created_at TIMESTAMP
);
```

#### 1.2 추천 히스토리
```sql
CREATE TABLE recommendation_history (
  id UUID PRIMARY KEY,
  user_id UUID,
  mood TEXT,
  comment TEXT,
  weather_data JSONB,
  recommended_items JSONB,
  feedback JSONB, -- 좋아요/싫어요
  created_at TIMESTAMP
);
```

#### 1.3 Vector DB 설정
- **옵션**: Pinecone, Weaviate, Chroma, 또는 PostgreSQL pgvector
- 옷장 아이템 벡터 저장 및 검색

---

### 2. 데이터 파이프라인 구축

#### 2.1 이미지 업로드 처리
```
사용자가 옷장에 이미지 업로드
  ↓
이미지 저장 (S3, Cloudinary, 또는 로컬)
  ↓
CLIP Image Encoder로 벡터 변환 (팀원 모델 사용)
  ↓
Vector DB에 벡터 저장
  ↓
PostgreSQL에 메타데이터 저장
```

#### 2.2 벡터 업데이트 API
```typescript
// src/app/api/closet/upload/route.ts
POST /api/closet/upload
- 이미지 업로드
- CLIP 모델 서버 호출하여 벡터 생성
- Vector DB에 저장
- PostgreSQL에 메타데이터 저장
```

---

### 3. API 서버 구축

#### 3.1 옷장 관리 API
```
GET    /api/closet              - 옷장 목록 조회
POST   /api/closet/upload       - 옷장 아이템 업로드
GET    /api/closet/:id          - 아이템 상세 조회
PUT    /api/closet/:id          - 아이템 수정
DELETE /api/closet/:id          - 아이템 삭제
```

#### 3.2 추천 API (모델 서버와 통신)
```typescript
// src/app/api/recommend/route.ts
POST /api/recommend
- 사용자 입력 받기
- 옷장 데이터 조회
- 모델 서버 호출 (팀원이 만든 모델)
- 결과 반환
```

#### 3.3 피드백 API
```
POST /api/recommend/:id/feedback
- 추천 결과에 대한 피드백 저장
- 추후 모델 개선에 활용
```

---

### 4. 모델 서버 통합

#### 4.1 모델 서버 인터페이스 정의
```typescript
// 모델 서버가 제공해야 할 API (팀원과 협의)
POST /model/recommend
{
  user_context: {
    text: string,        // "미니멀 데이트"
    weather: {...}       // 날씨 정보
  },
  closet_vectors: [...], // 옷장 벡터들
  top_k: number         // 추천 개수
}

Response:
{
  recommendations: [
    {
      item_id: string,
      score: number,
      reason: string
    }
  ]
}
```

#### 4.2 통합 레이어
```typescript
// src/lib/ml-client.ts
- 모델 서버와 통신
- 에러 처리
- 재시도 로직
- 캐싱 (선택)
```

---

### 5. Vector DB 통합

#### 5.1 Vector DB 선택
- **Pinecone**: 관리형 서비스 (쉬움)
- **Weaviate**: 오픈소스, 자체 호스팅
- **Chroma**: 경량, 임베딩 전용
- **PostgreSQL pgvector**: 기존 DB 활용

#### 5.2 Vector 검색 구현
```typescript
// src/lib/vector-db.ts
- 벡터 저장
- 유사도 검색 (Cosine Similarity)
- Top-K 추출
```

---

### 6. 프론트엔드 연동

#### 6.1 옷장 업로드 기능
```typescript
// src/components/ootd/ClosetUpload.tsx
- 이미지 업로드
- /api/closet/upload 호출
- 업로드 진행 상태 표시
```

#### 6.2 추천 API 호출 수정
```typescript
// src/app/ootd/page.tsx의 handleRecommend 수정
- /api/recommend 호출
- 모델 서버와 통신
- 결과 표시
```

---

### 7. 인프라 및 배포

#### 7.1 데이터베이스
- PostgreSQL (메타데이터)
- Vector DB (벡터 저장)

#### 7.2 스토리지
- 이미지 저장소 (S3, Cloudinary 등)

#### 7.3 배포
- Next.js → Vercel
- PostgreSQL → Supabase, Neon, 또는 자체 호스팅
- Vector DB → Pinecone (관리형) 또는 자체 호스팅

---

## 🔧 기술 스택

### 데이터베이스
- **PostgreSQL**: 메타데이터 저장
- **Vector DB**: Pinecone / Weaviate / Chroma

### 스토리지
- **이미지**: Cloudinary / AWS S3 / Vercel Blob

### API
- **Next.js API Routes**: 옷장 관리, 추천 API
- **모델 서버 통신**: HTTP/REST

### 인프라
- **Vercel**: Next.js 배포
- **Supabase/Neon**: PostgreSQL 호스팅
- **Pinecone**: Vector DB (관리형)

---

## 📝 구현 순서 (우선순위)

### Phase 1: 기본 인프라
1. ✅ PostgreSQL 데이터베이스 설계
2. ✅ Vector DB 선택 및 설정
3. ✅ 이미지 스토리지 설정

### Phase 2: 옷장 관리
4. ✅ 옷장 업로드 API
5. ✅ CLIP 모델 서버 연동 (벡터 생성)
6. ✅ Vector DB 저장
7. ✅ 옷장 조회/수정/삭제 API

### Phase 3: 추천 통합
8. ✅ 모델 서버 인터페이스 정의 (팀원과 협의)
9. ✅ 추천 API 구현
10. ✅ 프론트엔드 연동

### Phase 4: 추가 기능
11. ✅ 피드백 API
12. ✅ 추천 히스토리
13. ✅ 사용자 인증 (선택)

---

## 🤝 팀원과 협의 필요 사항

### 모델 서버 인터페이스
```typescript
// 팀원이 제공해야 할 API 스펙
POST /model/recommend
- 입력 형식
- 출력 형식
- 에러 처리
```

### CLIP 모델 서버
```typescript
// 이미지 → 벡터 변환 API
POST /model/encode-image
- 입력: 이미지 파일
- 출력: 벡터 배열
```

### 데이터 형식
- 옷장 아이템 데이터 구조
- 추천 결과 형식
- 벡터 차원 수

---

## ✅ 체크리스트

### 데이터베이스
- [ ] PostgreSQL 스키마 설계
- [ ] Vector DB 선택 및 설정
- [ ] 마이그레이션 스크립트 작성

### API
- [ ] 옷장 관리 API 구현
- [ ] 추천 API 구현
- [ ] 피드백 API 구현
- [ ] 모델 서버 통합

### 프론트엔드
- [ ] 옷장 업로드 UI
- [ ] 추천 API 호출 수정
- [ ] 에러 처리

### 인프라
- [ ] 데이터베이스 배포
- [ ] Vector DB 설정
- [ ] 이미지 스토리지 설정
- [ ] 환경 변수 설정

---

## 🎯 최종 목표

**다른 팀원의 모델을 붙여서 완전한 서비스 완성**

1. 사용자가 옷장에 이미지 업로드
2. 이미지 → 벡터 변환 (팀원 모델)
3. Vector DB에 저장
4. 사용자가 추천 요청
5. 모델 서버 호출 (팀원 모델)
6. 결과를 UI에 표시

**내 역할: 2-6번의 인프라와 통합 부분 전부 담당**
