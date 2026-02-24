# ML 컴포넌트 관계성 분석

## 📦 3가지 컴포넌트 개요

### 1. `ml-recommendation/` - 모델 학습 코드
- **역할**: ML 모델을 학습하고 저장하는 코드
- **위치**: `ootd_ai/ml-recommendation/`
- **주요 파일**:
  - `train/train_model.py`: 모델 학습 스크립트
  - `train/preprocess.py`: 데이터 전처리
  - `train/tokenizer.py`: 토크나이저 설정

### 2. `model/artifacts.pt` - 학습된 모델 파일
- **역할**: 학습 완료된 모델의 모든 정보를 담은 파일
- **위치**: `ootd_ai/model/artifacts.pt`
- **포함 내용**: 인코더 가중치 + vocab/maps + 인덱스 등

### 3. `ml-server/` - 추론 서버
- **역할**: 학습된 모델을 로드하여 실시간 추론 서비스 제공
- **위치**: `ootd_ai/ml-server/`
- **주요 파일**:
  - `app/model_loader.py`: artifacts.pt 로더
  - `app/predictor.py`: 추론 로직
  - `app/main.py`: FastAPI 서버

---

## 🔄 관계성 흐름도

```
┌─────────────────────────────────┐
│  1. ml-recommendation (학습)     │
│                                  │
│  train_model.py                 │
│    ↓                            │
│  데이터 전처리                   │
│    ↓                            │
│  모델 학습 (MLP)                │
│    ↓                            │
│  artifacts.pt 생성              │
└────────────┬────────────────────┘
             │
             │ (생성)
             ↓
┌─────────────────────────────────┐
│  2. model/artifacts.pt          │
│                                  │
│  포함 내용:                      │
│  - text_enc_state (텍스트 인코더)│
│  - item_enc_state (아이템 인코더)│
│  - text_vocab (토큰 사전)        │
│  - maps (카테고리 매핑)          │
│  - feature_cols (특징 컬럼)      │
│  - cfg (설정)                    │
│  - item_embs (아이템 임베딩)     │
└────────────┬────────────────────┘
             │
             │ (로드)
             ↓
┌─────────────────────────────────┐
│  3. ml-server (추론)            │
│                                  │
│  model_loader.py                │
│    ↓                            │
│  artifacts.pt 로드              │
│    ↓                            │
│  TextEncoder, ItemEncoder 재구성│
│    ↓                            │
│  predictor.py                   │
│    ↓                            │
│  실시간 추론 수행               │
│    ↓                            │
│  FastAPI /recommend 엔드포인트  │
└─────────────────────────────────┘
```

---

## 📝 상세 설명

### 1단계: ml-recommendation (학습)

#### 목적
- 코디 추천을 위한 ML 모델을 학습
- 학습 데이터로부터 패턴을 학습하여 모델 가중치 생성

#### 주요 과정
```python
# train_model.py 예시 구조
1. 데이터 로드 (CSV)
2. 전처리 (preprocess.py)
3. 토크나이저 설정 (tokenizer.py)
4. 모델 정의 (TextEncoder + ItemEncoder + MLP)
5. 학습 루프
   - Forward pass
   - Loss 계산
   - Backward pass
   - 가중치 업데이트
6. 모델 저장 → artifacts.pt
```

#### 출력물
- `artifacts.pt`: 학습된 모델의 모든 정보
  - 인코더 가중치 (text_enc_state, item_enc_state)
  - Vocab 사전 (text_vocab)
  - 카테고리 매핑 (maps)
  - 설정 정보 (cfg)
  - 기타 메타데이터

---

### 2단계: artifacts.pt (모델 파일)

#### 구조 (수정_문형서.md 참고)

**이슈 A. 모델 포맷 불일치**에서 언급:
- 기존: `best_model.pt` (단일 가중치만)
- 실제: `artifacts.pt` (인코더 상태 + vocab/maps + 인덱스 포함)

#### 포함 내용
```python
artifacts.pt = {
    "cfg": {...},                    # 모델 설정
    "FEATURE_COLS": [...],            # 특징 컬럼 목록
    "maps": {                         # 카테고리 매핑
        "part": {...},
        "카테고리": {...},
        "색상": {...},
        ...
    },
    "text_vocab": {                   # 텍스트 토큰 사전
        "stoi": {...},                # string → index
        "itos": [...],                # index → string
        "pad_idx": 0,
        "unk_idx": 1
    },
    "text_enc_state": {...},          # TextEncoder 가중치
    "item_enc_state": {...},          # ItemEncoder 가중치
    "item_embs": tensor(...),         # 아이템 임베딩 (선택적)
    "item_metas": [...],              # 아이템 메타데이터
    "item_table_min": {...},          # 아이템 테이블
    "WEATHER_LABEL_TO_TEMP_RANGE": {...}  # 날씨 매핑
}
```

#### 특징
- 단순 가중치가 아닌 **완전한 아티팩트 번들**
- 추론에 필요한 모든 정보 포함
- 학습 시점의 vocab/maps를 그대로 보존

---

### 3단계: ml-server (추론)

#### 목적
- 학습된 모델을 로드하여 실시간 추론 서비스 제공
- FastAPI로 HTTP API 엔드포인트 제공

#### 주요 과정

##### 3-1. 모델 로딩 (`model_loader.py`)
```python
def load_artifacts(artifacts_path):
    # 1. artifacts.pt 파일 로드
    payload = torch.load(artifacts_path)
    
    # 2. TextEncoder 재구성
    text_encoder = TextEncoder(...)
    text_encoder.load_state_dict(payload["text_enc_state"])
    
    # 3. ItemEncoder 재구성
    item_encoder = ItemEncoder(...)
    item_encoder.load_state_dict(payload["item_enc_state"])
    
    # 4. Vocab/maps 로드
    text_stoi = payload["text_vocab"]["stoi"]
    maps = payload["maps"]
    
    # 5. ArtifactsBundle 반환
    return ArtifactsBundle(...)
```

##### 3-2. 추론 (`predictor.py`)
```python
def recommend_outfits(bundle, mood, temperature, closet_items):
    # 1. 텍스트 인코딩
    text_emb = bundle.encode_text(mood)
    
    # 2. 아이템 인코딩
    item_embs = bundle.encode_items(item_features)
    
    # 3. 유사도 계산 (내적)
    similarities = text_emb @ item_embs.T
    
    # 4. 조합 생성 및 점수 계산
    # 5. Top-K 추천 반환
```

##### 3-3. API 서버 (`main.py`)
```python
@app.on_event("startup")
async def startup_event():
    global artifacts
    artifacts = load_artifacts()  # 서버 시작 시 한 번만 로드

@app.post("/recommend")
async def recommend(request):
    results = recommend_outfits(
        bundle=artifacts,
        mood=request.user_context.text,
        temperature=request.user_context.weather.temperature,
        closet_items=request.closet_items
    )
    return {"recommendations": results}
```

---

## 🔗 핵심 관계성

### 1. 학습 → 모델 파일
```
ml-recommendation/train_model.py
    ↓ (학습 완료 후 저장)
model/artifacts.pt
```

**특징**:
- 학습은 **한 번만** 수행
- 학습 완료 후 `artifacts.pt` 생성
- 이후 모델은 **freeze** 상태 (가중치 고정)

### 2. 모델 파일 → 추론 서버
```
model/artifacts.pt
    ↓ (서버 시작 시 로드)
ml-server/app/model_loader.py
    ↓ (ArtifactsBundle 생성)
ml-server/app/predictor.py
    ↓ (실시간 추론)
FastAPI /recommend 엔드포인트
```

**특징**:
- 서버 시작 시 **한 번만** 로드
- 이후 모든 추론 요청에 재사용
- 모델은 **읽기 전용** (추론만 수행)

### 3. 데이터 흐름
```
[학습 단계]
CSV 데이터 → 전처리 → 모델 학습 → artifacts.pt

[추론 단계]
사용자 입력 → artifacts.pt 로드 → 추론 → 결과 반환
```

---

## ⚠️ 중요 포인트

### 1. 모델 포맷 불일치 해결 (수정_문형서.md 참고)

**문제**:
- 기존 `ml-server`는 `best_model.pt` 단일 가중치만 로딩
- 실제 `artifacts.pt`는 인코더 + vocab/maps + 인덱스 포함

**해결**:
- `model_loader.py` 전면 교체
- `artifacts.pt` 기반 로더 구현
- TextEncoder, ItemEncoder 재구성
- Vocab/maps 로딩 추가

### 2. 스타일 매핑 우선순위 (predictor.py)

```python
# 학습 데이터 기준으로 '스타일' 값 우선 반영
pick("스타일", "style", "서브스타일", "sub_style")
```

**우선순위**:
1. `스타일` (한글)
2. `style` (영문)
3. `서브스타일` (한글)
4. `sub_style` (영문)

### 3. 원피스 처리 분리

**이전**: `top == bottom == dress` 형태로 처리
**현재**: `dress` 전용 추천 타입으로 분리
- `outfit_type: "dress"` + `dress_id` 응답 구조

---

## 📊 실제 사용 흐름

### 개발/학습 단계
```bash
# 1. 모델 학습
cd ml-recommendation
python train/train_model.py
# → model/artifacts.pt 생성
```

### 서비스 운영 단계
```bash
# 1. ML 서버 시작
cd ml-server
uvicorn app.main:app --host 0.0.0.0 --port 8000
# → artifacts.pt 로드 (startup 시)

# 2. 추론 요청
POST http://localhost:8000/recommend
{
  "user_context": {"text": "미니멀 데이트", ...},
  "closet_items": [...],
  "top_k": 10
}
# → 실시간 추론 수행
```

---

## 🎯 요약

### 관계성
1. **ml-recommendation**: 모델 학습 → `artifacts.pt` 생성
2. **artifacts.pt**: 학습된 모델의 완전한 정보 저장
3. **ml-server**: `artifacts.pt` 로드 → 실시간 추론 서비스

### 특징
- **학습**: 한 번만 수행, `artifacts.pt` 생성
- **추론**: 서버 시작 시 로드, 이후 재사용
- **모델**: freeze 상태 (추론만 수행, 학습 안 함)

### 데이터 흐름
```
학습 데이터 → ml-recommendation → artifacts.pt → ml-server → 추론 결과
```

---

## 📚 참고 문서

- `수정_문형서.md`: 모델 포맷 불일치 해결 과정
- `ml-recommendation/EXPLANATION.md`: 토크나이저, MLP, 대조학습 설명
- `ml-recommendation/MODEL_USAGE.md`: 모델 사용 흐름
- `ml-server/app/model_loader.py`: 실제 로더 구현
- `ml-server/app/predictor.py`: 실제 추론 로직
