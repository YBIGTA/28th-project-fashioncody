# 모델 사용 흐름 정리

## ✅ 맞습니다! 정확한 이해

### 학습 단계 (한 번만)
```
1. 학습 데이터 수집
2. 모델 학습 (MLP 가중치 업데이트)
3. 모델 저장 (freeze)
```

### 추론 단계 (매번)
```
1. 학습된 모델 로드 (freeze된 가중치)
2. 새로운 입력에 대해 점수만 예측
3. 결과 반환
```

## 실제 동작 흐름

### 학습 (한 번)
```python
# train_model.py
model = OutfitRecommendationModel()
# ... 학습 ...
torch.save(model.state_dict(), 'model.pt')  # 모델 저장
```

### 추론 (매 요청마다)
```python
# predictor.py
model = OutfitRecommendationModel()
model.load_state_dict(torch.load('model.pt'))  # 학습된 가중치 로드
model.eval()  # 평가 모드 (freeze)

# 새로운 입력에 대해 점수만 예측
with torch.no_grad():  # gradient 계산 안 함 (추론만)
    score = model(input_ids, attention_mask, features)
    # 점수만 반환 (학습 X)
```

## 제가 지적한 문제점 재정리

### 문제는 "어떤 조합들을 추론할 것인가?"

#### 현재 구조의 문제
```python
# 옷장에 100개 아이템
closet_items = [item1, item2, ..., item100]

# 모든 조합 생성
all_combinations = []
for top in closet_items:  # 30개
    for bottom in closet_items:  # 30개
        for outer in closet_items:  # 20개
            all_combinations.append((top, bottom, outer))
# → 18,000개 조합

# 각 조합마다 추론 (모델은 freeze, 추론만 수행)
scores = []
for combo in all_combinations:  # 18,000번 반복
    score = model.predict(combo)  # 추론만 (학습 X)
    scores.append(score)
# → 18,000번 추론 = 느림
```

#### 개선안: 필터링 후 추론
```python
# 1단계: 규칙 기반 필터링 (ML 모델 사용 안 함)
filtered = filter_by_rules(closet_items, weather)
# → 18,000개 → 100개로 축소

# 2단계: 필터링된 조합만 추론 (모델은 freeze, 추론만)
scores = []
for combo in filtered:  # 100번만 반복
    score = model.predict(combo)  # 추론만 (학습 X)
    scores.append(score)
# → 100번 추론 = 빠름
```

## 핵심 정리

### ✅ 맞는 부분
1. **모델은 한 번 학습하고 freeze**
2. **추론 시에는 점수만 예측** (학습 안 함)
3. **가중치는 고정** (추론 중 업데이트 안 됨)

### ⚠️ 문제점
1. **모든 조합에 대해 추론하면 너무 느림**
2. **필터링으로 조합 수를 줄인 후 추론하는 것이 효율적**

## 최종 구조

```
[학습 단계 - 한 번만]
데이터 → 모델 학습 → 모델 저장 (freeze)

[추론 단계 - 매 요청마다]
새 입력 → 규칙 필터링 → 필터링된 조합만 추론 → 점수 반환
         (ML 없음)      (freeze된 모델 사용)
```

## 코드 예시

### 학습 (한 번)
```python
# train_model.py
model = OutfitRecommendationModel()
optimizer = optim.Adam(model.parameters())

for epoch in range(epochs):
    # 학습
    loss = criterion(model(inputs), labels)
    loss.backward()
    optimizer.step()

# 모델 저장 (freeze)
torch.save(model.state_dict(), 'model.pt')
```

### 추론 (매번)
```python
# predictor.py
model = OutfitRecommendationModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()  # 평가 모드 (freeze)

# 추론만 수행 (학습 안 함)
with torch.no_grad():  # gradient 계산 안 함
    score = model(input_ids, attention_mask, features)
    return score  # 점수만 반환
```

## 결론

**당신의 이해가 정확합니다!**

- ✅ 모델은 학습 후 freeze
- ✅ 추론 시에는 점수만 예측
- ⚠️ 하지만 모든 조합을 추론하면 느리므로 필터링 필요

**최종 구조:**
```
필터링 (규칙) → 추론 (freeze된 모델) → 점수 반환
```
