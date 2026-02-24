# 코디 추천 ML 시스템 구현 가이드

## 🎯 추천 구조

### 옵션 A: Next.js 통합 (간단한 모델)
- **장점**: 배포 간단, 서버 하나로 관리
- **단점**: Node.js에서 ML 추론은 제한적
- **사용 기술**: ONNX Runtime, TensorFlow.js

### 옵션 B: 별도 ML 서버 (권장) ⭐
- **장점**: Python 생태계 활용, 복잡한 모델 가능, 확장성 좋음
- **단점**: 서버 2개 관리 필요
- **사용 기술**: FastAPI + PyTorch/TensorFlow

## 📋 구현 단계

### 1단계: 데이터 준비
```bash
# 학습 데이터 수집
# - 사용자 입력 (mood, comment)
# - 날씨 정보
# - 옷 조합 및 매칭 점수 (라벨)
```

### 2단계: 모델 학습
```bash
cd ml-recommendation
pip install -r requirements.txt

# 데이터 전처리
python train/preprocess.py

# 모델 학습
python train/train_model.py
```

### 3단계: 모델 서빙
```bash
# 옵션 B 선택 시
cd ml-server
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4단계: Next.js 연동
```typescript
// src/app/ootd/page.tsx의 handleRecommend 수정
const response = await fetch('/api/recommend', {
  method: 'POST',
  body: JSON.stringify({
    mood: moodText,
    comment: commentText,
    temperature: weatherData.temperature,
    feelsLike: weatherData.feelsLike,
    precipitation: weatherData.precipitation,
    closetItems: mockClosetItems
  })
});
```

## 🔧 환경 변수 설정

### Next.js (.env.local)
```env
ML_SERVER_URL=http://localhost:8000
```

### ML 서버 (ml-server/.env)
```env
MODEL_PATH=../ml-recommendation/models/best_model.pt
TOKENIZER_PATH=../ml-recommendation/models/
```

## 📊 모델 구조

### 입력 특징
1. **텍스트 임베딩** (토크나이저)
   - mood + comment → 256차원 벡터

2. **수치 특징**
   - 기온, 체감온도, 강수량
   - 카테고리 인덱스 (상의, 하의, 아우터)

3. **결합**
   - 텍스트 벡터 + 수치 특징 → MLP 입력

### 출력
- 매칭 점수 (0.0 ~ 1.0)
- Top-K 추천 (점수 순 정렬)

## 🚀 배포

### Vercel (Next.js)
- 자동 배포
- ML 서버는 별도 배포 필요

### ML 서버 배포 옵션
1. **Railway** (권장): Python 앱 배포 간단
2. **Render**: 무료 티어 제공
3. **AWS/GCP**: 프로덕션 환경

## 📝 다음 단계

1. ✅ 데이터 수집 및 라벨링
2. ✅ 모델 학습
3. ✅ API 연동
4. ⬜ 모델 최적화
5. ⬜ A/B 테스트
6. ⬜ 피드백 루프 구축
