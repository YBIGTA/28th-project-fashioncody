-- 기존 테이블에 image_id 컬럼 추가 (이미 데이터가 있는 경우 사용)
ALTER TABLE closet_items ADD COLUMN IF NOT EXISTS image_id VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_closet_items_image_id ON closet_items(image_id);
