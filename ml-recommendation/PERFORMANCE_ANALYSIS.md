# 코디 추천 시스템 성능 분석 및 개선안

## 현재 구조 평가

### 제안된 구조
```
텍스트 입력 → 토크나이저 → 임베딩 → MLP → 점수 출력
```

## 성능 분석

### ✅ 장점

1. **구현 간단성**
   - MLP는 이해하기 쉬움
   - 디버깅 용이
   - 빠른 프로토타이핑 가능

2. **빠른 추론 속도**
   - MLP는 경량 모델
   - 실시간 추천 가능 (수십 ms)
   - 서버 리소스 효율적

3. **한국어 지원**
   - KoBERT/KoELECTRA 사용
   - 한국어 텍스트 처리 가능

### ⚠️ 단점 및 문제점

#### 1. **데이터 요구사항이 높음**
```
문제: MLP는 많은 학습 데이터 필요
- 각 옷 조합마다 라벨(점수) 필요
- 수천~수만 개의 (입력, 코디, 점수) 쌍 필요
- 실제로는 데이터 수집이 어려움
```

#### 2. **콜드 스타트 문제**
```
문제: 새로운 옷장 아이템 추가 시
- 모델이 학습하지 않은 아이템은 성능 저하
- 재학습 필요할 수 있음
```

#### 3. **조합 공간이 너무 큼**
```
문제: 모든 옷 조합을 평가해야 함
- 옷장에 100개 아이템 → 100 × 100 × 100 = 1,000,000 조합
- 각 조합마다 추론 → 매우 느림
```

#### 4. **대조학습의 한계**
```
문제: 대조학습은 데이터가 많을 때 효과적
- 적은 데이터에서는 오히려 성능 저하 가능
- Positive/Negative 샘플 구성이 어려움
```

## 개선된 구조 제안

### 옵션 1: 하이브리드 접근 (권장) ⭐

```
1단계: 필터링 (규칙 기반)
   - 날씨 조건으로 옷 필터링
   - 카테고리 제약 (상의+하의 필수)
   - → 1,000,000개 → 1,000개로 축소

2단계: 랭킹 (ML 모델)
   - 필터링된 조합만 MLP로 점수 예측
   - Top-K 선택
   - → 빠르고 정확
```

**장점:**
- 추론 속도 1000배 향상
- 데이터 요구사항 감소
- 실용적

### 옵션 2: 임베딩 기반 유사도 (데이터 부족 시)

```
1. 옷 아이템 임베딩 생성
   - 이미지 → Vision Transformer
   - 텍스트(이름, 태그) → 텍스트 임베딩
   - 결합 → 아이템 임베딩 벡터

2. 사용자 의도 임베딩
   - "미니멀 데이트" → 텍스트 임베딩

3. 코사인 유사도로 매칭
   - 사용자 의도와 옷 조합의 유사도 계산
   - Top-K 선택
```

**장점:**
- 학습 데이터 적게 필요
- 새로운 아이템 추가 용이
- 해석 가능

### 옵션 3: Two-Stage 모델

```
Stage 1: 아이템 선택 모델
   - 각 카테고리별로 적합한 아이템 선택
   - 상의 선택, 하의 선택, 아우터 선택

Stage 2: 조합 평가 모델
   - 선택된 아이템들의 조합 점수 예측
   - MLP 사용
```

**장점:**
- 조합 공간 축소
- 더 정확한 추천
- 단계별 최적화 가능

## 성능 비교

| 구조 | 추론 속도 | 데이터 요구 | 정확도 | 확장성 |
|------|----------|-----------|--------|--------|
| 현재 (MLP만) | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ |
| 하이브리드 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 임베딩 유사도 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Two-Stage | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 실용적 권장사항

### MVP 단계 (현재)
```
1. 규칙 기반 필터링
2. 간단한 MLP (또는 유사도 기반)
3. 빠른 프로토타이핑
```

### 프로덕션 단계
```
1. 하이브리드 접근
2. Two-Stage 모델
3. A/B 테스트로 최적화
```

## 구체적 개선 코드

### 개선 1: 필터링 + MLP 하이브리드

```python
def recommend_outfit(mood, weather, closet_items):
    # 1단계: 규칙 기반 필터링
    filtered_combinations = filter_by_rules(
        closet_items,
        min_temp=weather['feels_like'] - 5,
        max_temp=weather['feels_like'] + 5,
        season=get_season(weather)
    )
    
    # 2단계: MLP로 점수 예측 (필터링된 조합만)
    scores = []
    for combo in filtered_combinations:
        score = mlp_model.predict(mood, weather, combo)
        scores.append((combo, score))
    
    # 3단계: Top-K 반환
    return sorted(scores, key=lambda x: x[1], reverse=True)[:10]
```

### 개선 2: 임베딩 기반 유사도

```python
def recommend_by_similarity(mood, closet_items):
    # 사용자 의도 임베딩
    mood_embedding = text_encoder.encode(mood)
    
    # 각 옷 조합의 임베딩
    outfit_embeddings = []
    for combo in generate_combinations(closet_items):
        combo_embedding = combine_item_embeddings(combo)
        outfit_embeddings.append((combo, combo_embedding))
    
    # 코사인 유사도 계산
    similarities = [
        (combo, cosine_similarity(mood_embedding, emb))
        for combo, emb in outfit_embeddings
    ]
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
```

## 결론

### 현재 구조의 문제점
1. ❌ 조합 공간이 너무 큼 (성능 저하)
2. ❌ 데이터 요구사항이 높음
3. ❌ 실용성이 떨어짐

### 권장 개선안
1. ✅ **하이브리드 접근**: 규칙 기반 필터링 + MLP
2. ✅ **Two-Stage 모델**: 아이템 선택 → 조합 평가
3. ✅ **임베딩 유사도**: 데이터 부족 시 대안

### 최종 권장 구조
```
입력: mood, weather, closet_items
  ↓
1. 규칙 기반 필터링 (날씨, 시즌, 카테고리)
  ↓
2. MLP로 점수 예측 (필터링된 조합만)
  ↓
3. Top-K 반환
```

이 구조가 **성능, 속도, 실용성** 모두에서 최적입니다!
