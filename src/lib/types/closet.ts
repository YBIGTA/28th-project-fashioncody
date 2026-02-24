/**
 * 옷장 아이템 타입 정의
 * dummy.json의 이미지 분석 결과를 기반으로 확장
 */

export type ClosetItemCategory =
  | "top"
  | "bottom"
  | "outer"
  | "shoes"
  | "bag"
  | "accessory"
  | "dress";

export type AttributeWithConfidence = {
  value: string;
  confidence: number;
};

export type ClosetItemAttributes = {
  // 기본 정보
  category: ClosetItemCategory;
  detection_confidence: number;

  // 타입 정보
  sub_type?: string;
  sub_type_confidence?: number;

  // 색상 정보
  color?: string;
  color_confidence?: number;
  sub_color?: string;
  sub_color_confidence?: number;

  // 디자인 정보
  sleeve_length?: string;
  sleeve_length_confidence?: number;
  length?: string;
  length_confidence?: number;
  fit?: string;
  fit_confidence?: number;
  collar?: string;
  collar_confidence?: number;

  // 상세 정보
  material?: AttributeWithConfidence[];
  print?: AttributeWithConfidence[];
  detail?: AttributeWithConfidence[];
};

export type ClosetItem = {
  id: string;

  // 이미지 정보
  imageUrl: string;
  imageId?: string;        // CSV의 image_id (원본 파일명 매핑)
  imageVector?: number[]; // CLIP 이미지 벡터 (사전 계산)

  // 속성 정보 (dummy.json에서 추출)
  attributes: ClosetItemAttributes;

  // 사용자 입력 정보 (선택)
  name?: string;
  tags?: string[];
  season?: ("spring" | "summer" | "fall" | "winter")[];

  // 메타데이터
  createdAt: string;
  updatedAt?: string;
};
