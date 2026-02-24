/**
 * 프론트엔드 컴포넌트용 플랫 타입
 * ClosetItem (도메인 객체)에서 변환하여 사용
 */

import type { ClosetItem, ClosetItemCategory } from "./closet";

export type ClosetItemView = {
  id: string;
  name: string;
  imageUrl: string;
  category: ClosetItemCategory;
  color?: string;
  subType?: string;
  material?: string;
  fit?: string;
  season?: ("spring" | "summer" | "fall" | "winter")[];
  tags?: string[];
  createdAt: string;
};

/**
 * ClosetItem → ClosetItemView 변환
 */
export function toClosetItemView(item: ClosetItem): ClosetItemView {
  return {
    id: item.id,
    name: item.name || item.attributes.sub_type || item.attributes.category,
    imageUrl: item.imageUrl,
    category: item.attributes.category,
    color: item.attributes.color,
    subType: item.attributes.sub_type,
    material: item.attributes.material?.[0]?.value,
    fit: item.attributes.fit,
    season: item.season,
    tags: item.tags,
    createdAt: item.createdAt,
  };
}

const categoryLabels: Record<ClosetItemCategory, string> = {
  top: "상의",
  bottom: "하의",
  outer: "아우터",
  shoes: "신발",
  bag: "가방",
  accessory: "액세서리",
  dress: "원피스",
};

export function getCategoryLabel(category: string): string {
  return categoryLabels[category as ClosetItemCategory] || category;
}

const seasonLabels: Record<string, string> = {
  spring: "봄",
  summer: "여름",
  fall: "가을",
  winter: "겨울",
};

export function getSeasonLabel(season: string): string {
  return seasonLabels[season] || season;
}
