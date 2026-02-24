# VectorDB 연결 상태 및 사용 방식 분석

## 🔍 VectorDB 종류

**PostgreSQL + pgvector** (외부 VectorDB 아님)

- 외부 VectorDB (Pinecone, Weaviate, Chroma) 사용 안 함
- PostgreSQL의 `pgvector` 확장 사용
- `closet_items` 테이블의 `image_vector` 컬럼에 벡터 저장

---

## ✅ 연결 상태

### 구현되어 있음

1. **스키마 정의**: `database/schema.sql`
   - `image_vector vector(512)` 컬럼 정의
   - `CREATE EXTENSION IF NOT EXISTS vector;` (pgvector 확장)
   - 벡터 검색 인덱스 생성

2. **레포지토리 구현**: `src/lib/db/postgres-repository.ts`
   - `updateVector()`: 벡터 저장
   - `findSimilar()`: pgvector 유사도 검색

3. **데이터베이스 연결**: `src/lib/db/neon-client.ts`
   - Neon PostgreSQL 연결
   - `DATABASE_URL` 환경 변수 필요

### 실제 연결 여부 확인 필요

- `DATABASE_URL` 환경 변수가 설정되어 있어야 함
- PostgreSQL에 pgvector 확장이 설치되어 있어야 함
- 스키마가 마이그레이션되어 있어야 함

---

## 📊 사용 방식

### 1. 이미지 업로드 시 벡터 저장

**파일**: `src/app/api/closet/upload/route.ts`  
**위치**: 136-160줄

```typescript
// 1. CLIP 모델 서버 호출 (팀원 모델)
const response = await fetch(`${CLIP_MODEL_URL}/encode-image`, {
  method: "POST",
  body: formData,  // 이미지 파일
});

// 2. 벡터 받아오기
const data = await response.json();  // { vector: [0.1, 0.2, ...] }

// 3. PostgreSQL에 벡터 저장
await repository.updateVector(itemId, data.vector);
```

**실행 흐름**:
```
이미지 업로드
  ↓
CLIP Image Encoder (팀원 모델 서버) 호출
  ↓
이미지 벡터 생성 (512차원)
  ↓
PostgreSQL의 image_vector 컬럼에 저장
  ↓
pgvector 인덱스에 자동 반영
```

**특징**:
- ✅ Non-blocking: 벡터 인코딩 실패해도 아이템 저장은 성공
- ✅ Best-effort: 실패 시 경고만 출력하고 계속 진행

---

### 2. 추천 시 벡터 검색 (후보 narrowing)

**파일**: `src/app/api/recommend/route.ts`  
**위치**: 87-106줄

```typescript
async function tryMLRecommendation(...) {
  let candidateItems = allItems;  // 기본값: 모든 아이템
  
  try {
    // 1. 텍스트 벡터 생성 (CLIP Text Encoder)
    const textVector = await encodeTextToVector(mood, comment);
    
    // 2. pgvector로 유사도 검색
    const similarItems = await repository.findSimilar(textVector, 100);
    
    // 3. 검색 결과가 있으면 후보로 사용
    if (similarItems.length > 0) {
      candidateItems = similarItems;
    }
  } catch (vectorError) {
    // 벡터 검색 실패해도 계속 진행
    console.warn("Vector candidate narrowing skipped:", vectorError);
  }
  
  // 4. ML 서버에 후보 아이템 전달
  const response = await fetch(`${ML_SERVER_URL}/recommend`, {
    body: JSON.stringify({
      closet_items: candidateItems,  // narrowing된 후보
      ...
    })
  });
}
```

**실행 흐름**:
```
사용자 입력: "미니멀 데이트"
  ↓
CLIP Text Encoder (팀원 모델 서버) 호출
  ↓
텍스트 벡터 생성 (512차원)
  ↓
pgvector 유사도 검색 (Cosine Similarity)
  ↓
Top-100 유사한 아이템 반환
  ↓
ML 서버에 후보 아이템 전달 (전체 아이템 대신)
  ↓
최종 추천 결과 생성
```

**특징**:
- ✅ Optional: 벡터 검색 실패해도 전체 아이템으로 진행
- ✅ 후보 narrowing: 전체 아이템 대신 유사한 아이템만 ML 서버에 전달
- ✅ 성능 향상: ML 서버 처리량 감소

---

### 3. pgvector 유사도 검색 구현

**파일**: `src/lib/db/postgres-repository.ts`  
**위치**: 142-155줄

```typescript
async findSimilar(
  vector: number[],
  topK: number
): Promise<ClosetItem[]> {
  const vectorStr = `[${vector.join(",")}]`;
  const rows = await this.sql`
    SELECT *
    FROM closet_items
    WHERE image_vector IS NOT NULL
    ORDER BY image_vector <=> ${vectorStr}::vector
    LIMIT ${topK}
  `;
  return rows.map(rowToClosetItem);
}
```

**쿼리 설명**:
- `image_vector <=> ${vectorStr}::vector`: pgvector의 코사인 거리 연산자
- `ORDER BY`: 거리 순으로 정렬 (가까운 순)
- `LIMIT ${topK}`: Top-K 결과만 반환

---

## 🔗 전체 흐름

### 옷장 아이템 등록
```
[사용자] 이미지 업로드
  ↓
[Next API] /api/closet/upload
  ↓
[Vercel Blob] 이미지 저장
  ↓
[이미지 분석 모델] 속성 추출 (팀원 모델)
  ↓
[PostgreSQL] 메타데이터 저장
  ↓
[CLIP Image Encoder] 벡터 생성 (팀원 모델)
  ↓
[PostgreSQL + pgvector] image_vector 컬럼에 저장
```

### 추천 요청
```
[사용자] "미니멀 데이트" 입력
  ↓
[Next API] /api/recommend
  ↓
[CLIP Text Encoder] 텍스트 벡터 생성 (팀원 모델)
  ↓
[PostgreSQL + pgvector] 유사도 검색
  ↓
[후보 narrowing] Top-100 유사 아이템
  ↓
[ML 서버] 최종 추천 점수 계산
  ↓
[추천 결과] 반환
```

---

## ⚠️ 현재 상태

### 구현 완료 ✅
1. ✅ pgvector 스키마 정의
2. ✅ 벡터 저장 로직 (`updateVector`)
3. ✅ 벡터 검색 로직 (`findSimilar`)
4. ✅ 이미지 업로드 시 벡터 저장
5. ✅ 추천 시 벡터 검색 (optional)

### 확인 필요 ⚠️
1. ⚠️ `DATABASE_URL` 환경 변수 설정 여부
2. ⚠️ PostgreSQL에 pgvector 확장 설치 여부
3. ⚠️ 스키마 마이그레이션 실행 여부
4. ⚠️ CLIP 모델 서버 (`CLIP_MODEL_URL`) 연결 여부

### 동작 방식
- ✅ 벡터 검색 성공 시: 후보 narrowing → ML 서버에 전달
- ✅ 벡터 검색 실패 시: 전체 아이템으로 fallback → ML 서버에 전달
- ✅ 항상 ML 서버 추론은 수행됨 (벡터 검색은 optional)

---

## 📝 요약

### VectorDB 종류
- **PostgreSQL + pgvector** (외부 VectorDB 아님)

### 연결 상태
- ✅ **구현되어 있음**
- ⚠️ **실제 연결 여부는 환경 변수 및 DB 설정에 따라 다름**

### 사용 방식
1. **이미지 업로드 시**: CLIP 벡터 생성 → PostgreSQL `image_vector` 컬럼에 저장
2. **추천 시**: 텍스트 벡터 생성 → pgvector 유사도 검색 → 후보 narrowing → ML 서버에 전달

### 특징
- ✅ Optional: 벡터 검색 실패해도 전체 아이템으로 진행
- ✅ Non-blocking: 벡터 인코딩 실패해도 아이템 저장은 성공
- ✅ Best-effort: 실패 시 경고만 출력하고 계속 진행
