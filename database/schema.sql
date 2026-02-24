-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 옷장 아이템 테이블
CREATE TABLE closet_items (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- 이미지 정보
  image_url TEXT NOT NULL,
  image_id VARCHAR(100),    -- CSV image_id (원본 파일명 매핑)
  image_vector vector(512), -- CLIP 이미지 벡터 (pgvector)

  -- 속성 정보 (dummy.json 구조 반영)
  category VARCHAR(50) NOT NULL, -- top, bottom, outer, shoes, bag, accessory, dress
  detection_confidence FLOAT,

  -- 타입 정보
  sub_type VARCHAR(100),
  sub_type_confidence FLOAT,

  -- 색상 정보
  color VARCHAR(50),
  color_confidence FLOAT,
  sub_color VARCHAR(50),
  sub_color_confidence FLOAT,

  -- 디자인 정보
  sleeve_length VARCHAR(50),
  sleeve_length_confidence FLOAT,
  length VARCHAR(50),
  length_confidence FLOAT,
  fit VARCHAR(50),
  fit_confidence FLOAT,
  collar VARCHAR(50),
  collar_confidence FLOAT,

  -- 상세 정보 (JSON 배열로 저장)
  material JSONB, -- [{"value": "면", "confidence": 0.9}]
  print JSONB,
  detail JSONB,

  -- 사용자 입력 정보
  name VARCHAR(255),
  tags VARCHAR(50)[],
  season VARCHAR(10)[], -- spring, summer, fall, winter

  -- 메타데이터
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX idx_closet_items_category ON closet_items(category);
CREATE INDEX idx_closet_items_created_at ON closet_items(created_at DESC);
CREATE INDEX idx_closet_items_image_id ON closet_items(image_id);

-- pgvector 코사인 유사도 검색 인덱스
CREATE INDEX idx_closet_items_vector ON closet_items
  USING ivfflat (image_vector vector_cosine_ops)
  WITH (lists = 10);

-- 추천 히스토리 테이블
CREATE TABLE recommendation_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- 입력 정보
  mood TEXT,
  comment TEXT,
  weather_data JSONB,

  -- 추천 결과
  recommended_items JSONB,

  -- 피드백
  feedback JSONB,

  -- 메타데이터
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_recommendation_history_created_at ON recommendation_history(created_at DESC);

-- 하이퍼파라미터 기준점 저장 (단일 행 싱글톤)
CREATE TABLE user_hyperparams (
    id           UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
    alpha_tb     FLOAT DEFAULT 0.65,   -- top-bottom 색상 vs 임베딩 비중
    alpha_oi     FLOAT DEFAULT 0.70,   -- outer-inner 색상 vs 임베딩 비중
    mmr_lambda   FLOAT DEFAULT 0.75,   -- MMR quality vs diversity
    beta_tb      FLOAT DEFAULT 0.50,   -- outer-top vs outer-bottom 균형
    lambda_tbset FLOAT DEFAULT 0.15,   -- inner 응집력 반영 비중
    sigma        FLOAT DEFAULT 0.05,   -- 탐색 노이즈 표준편차
    eta          FLOAT DEFAULT 0.10,   -- 학습률 (기준점 갱신 폭)
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO user_hyperparams DEFAULT VALUES;

-- recommendation_history: 해당 추천에 사용된 하이퍼파라미터 기록
ALTER TABLE recommendation_history
    ADD COLUMN IF NOT EXISTS hyperparams_used JSONB;
