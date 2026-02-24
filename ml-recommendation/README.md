# 코디 추천 ML 시스템 구조

## 기술 스택

### 1. 모델 학습 (Python)
- **PyTorch** 또는 **TensorFlow**: MLP 모델 구축
- **transformers** (Hugging Face): 한국어 토크나이저 (KoBERT, KoELECTRA)
- **scikit-learn**: 전처리 및 유틸리티
- **pandas, numpy**: 데이터 처리

### 2. 모델 서빙 옵션

#### 옵션 A: Next.js 통합 (권장 - 단순한 경우)
- **ONNX Runtime** (Node.js): 학습된 모델을 ONNX로 변환 후 Next.js에서 직접 추론
- **TensorFlow.js**: 브라우저/Node.js에서 직접 실행

#### 옵션 B: 별도 ML 서버 (권장 - 복잡한 모델)
- **FastAPI** (Python): ML 모델 서빙 전용 서버
- **PyTorch/TensorFlow Serving**: 프로덕션 환경

### 3. 데이터 처리
- **KoNLPy**: 한국어 자연어 처리
- **sentence-transformers**: 텍스트 임베딩

## 프로젝트 구조

```
ybigta_newbieproject/
├── ml-recommendation/          # ML 학습 코드
│   ├── train/
│   │   ├── train_model.py      # 모델 학습 스크립트
│   │   ├── preprocess.py       # 데이터 전처리
│   │   └── tokenizer.py        # 토크나이저 설정
│   ├── models/                 # 학습된 모델 저장
│   │   ├── model.onnx          # ONNX 형식
│   │   ├── tokenizer.json      # 토크나이저
│   │   └── config.json         # 모델 설정
│   ├── data/                   # 학습 데이터
│   │   ├── train.csv
│   │   └── test.csv
│   └── requirements.txt
├── ml-server/                  # ML 서버 (옵션 B)
│   ├── app/
│   │   ├── main.py            # FastAPI 앱
│   │   ├── model_loader.py    # 모델 로드
│   │   └── predictor.py       # 추론 로직
│   └── requirements.txt
└── src/
    └── app/
        └── api/
            └── recommend/
                └── route.ts   # Next.js API Route
```

## 모델 설계

### 입력 특징 (Feature Engineering)
1. **텍스트 특징** (토크나이저 사용)
   - mood 텍스트 → 임베딩 벡터
   - comment 텍스트 → 임베딩 벡터
   
2. **카테고리 특징** (One-hot encoding)
   - 상의 카테고리
   - 하의 카테고리
   - 아우터 카테고리
   
3. **속성 특징**
   - 색상 (embedding)
   - 시즌 (multi-hot)
   - 태그 (multi-hot)
   
4. **날씨 특징**
   - 기온
   - 체감 온도
   - 날씨 타입
   - 강수량

### 출력
- **분류**: 각 옷 조합의 매칭 점수 (0-1)
- **랭킹**: Top-K 추천

## 학습 파이프라인

1. 데이터 수집 및 라벨링
2. 전처리 (토크나이저, 정규화)
3. 특징 추출
4. MLP 모델 학습
5. 모델 평가 및 최적화
6. 모델 변환 (ONNX/TensorFlow.js)
7. 배포
