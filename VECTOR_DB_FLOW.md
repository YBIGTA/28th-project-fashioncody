# Vector DB 사용 흐름

## 현재 vs 실제 서비스

### 현재 (개발/테스트용)
```
dummy.json → ClosetItem 변환 → 메모리 저장소
```

### 실제 서비스
```
이미지 업로드 → CLIP 벡터 생성 → Vector DB 저장
  ↓
사용자 입력 → 벡터 검색 → Vector DB에서 유사한 벡터 찾기
  ↓
아이템 ID 반환 → PostgreSQL에서 메타데이터 조회
```

---

## 실제 서비스 파이프라인

### 1. 옷장 아이템 등록 시
```
사용자가 이미지 업로드
  ↓
이미지 분석 모델 (팀원) → 속성 추출 (dummy.json 형식)
  ↓
CLIP Image Encoder (팀원 모델) → 이미지 벡터 생성
  ↓
PostgreSQL에 메타데이터 저장 (속성 정보)
  ↓
Vector DB에 벡터 저장 (벡터 + 아이템 ID)
```

### 2. 추천 요청 시
```
사용자 입력: "미니멀 데이트" + 날씨
  ↓
LLM 키워드 확장 (팀원 모델)
  ↓
CLIP Text Encoder (팀원 모델) → 텍스트 벡터 생성
  ↓
Vector DB에서 유사도 검색 (Cosine Similarity)
  ↓
Top-K 아이템 ID 반환
  ↓
PostgreSQL에서 아이템 메타데이터 조회
  ↓
모델 서버에서 최종 점수 계산 (팀원 모델)
  ↓
추천 결과 반환
```

---

## Vector DB 구조

### 저장되는 데이터
```typescript
// Vector DB에 저장
{
  id: "vector-123",           // Vector DB 내부 ID
  vector: [0.1, 0.2, ...],    // CLIP 이미지 벡터 (512차원 등)
  metadata: {
    itemId: "item-456",       // PostgreSQL의 closet_items.id
    userId: "user-789",
    category: "top"
  }
}
```

### 검색 흐름
```typescript
// 1. 텍스트 벡터 생성 (팀원 모델)
const textVector = await clipTextEncoder.encode("미니멀 데이트");

// 2. Vector DB에서 유사도 검색
const similarVectors = await vectorDB.query({
  vector: textVector,
  topK: 100,
  filter: { userId: "user-789" }  // 사용자 옷장만 검색
});

// 3. 아이템 ID 추출
const itemIds = similarVectors.map(v => v.metadata.itemId);

// 4. PostgreSQL에서 메타데이터 조회
const items = await db.closetItems.findByIds(itemIds);

// 5. 모델 서버에서 최종 점수 계산
const recommendations = await modelServer.recommend({
  userContext: { text: "미니멀 데이트", weather: {...} },
  items: items
});
```

---

## dummy.json의 역할

### 개발 단계
- ✅ 초기 데이터로 사용
- ✅ 타입 정의 및 스키마 설계
- ✅ API 테스트

### 실제 서비스
- ❌ dummy.json 직접 사용 안 함
- ✅ 이미지 업로드 → 속성 추출 (dummy.json 형식) → DB 저장
- ✅ Vector DB에서 벡터 검색 사용

---

## 데이터 흐름 정리

### 옷장 등록
```
이미지 업로드
  → 이미지 분석 (속성 추출) → PostgreSQL 저장
  → CLIP 벡터 생성 → Vector DB 저장
```

### 추천 요청
```
사용자 입력
  → 텍스트 벡터 생성
  → Vector DB 검색 (유사한 옷장 아이템 찾기)
  → PostgreSQL에서 메타데이터 조회
  → 모델 서버에서 최종 점수 계산
  → 추천 결과 반환
```

---

## 결론

**dummy.json은 개발용이고, 실제 서비스에서는:**
1. ✅ **Vector DB**: 벡터 검색으로 유사한 아이템 찾기
2. ✅ **PostgreSQL**: 아이템 메타데이터 저장 및 조회
3. ✅ **모델 서버**: 최종 추천 점수 계산

**Vector DB가 핵심입니다!**
