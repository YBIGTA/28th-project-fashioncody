/**
 * 옷장 아이템 데이터베이스 레포지토리 인터페이스
 */

import type { ClosetItem } from "@/lib/types/closet";

export interface ClosetRepository {
  findAll(): Promise<ClosetItem[]>;
  findById(id: string): Promise<ClosetItem | null>;
  findByCategory(category: string): Promise<ClosetItem[]>;
  create(item: Omit<ClosetItem, "id" | "createdAt">): Promise<ClosetItem>;
  update(id: string, updates: Partial<ClosetItem>): Promise<ClosetItem>;
  delete(id: string): Promise<void>;
  updateVector(id: string, vector: number[]): Promise<void>;
  findSimilar(vector: number[], topK: number): Promise<ClosetItem[]>;
  findByImageId(imageId: string): Promise<ClosetItem | null>;
}

/**
 * 메모리 기반 임시 구현 (fallback/테스트용)
 */
export class InMemoryClosetRepository implements ClosetRepository {
  private items: Map<string, ClosetItem> = new Map();

  async findAll(): Promise<ClosetItem[]> {
    return Array.from(this.items.values());
  }

  async findById(id: string): Promise<ClosetItem | null> {
    return this.items.get(id) || null;
  }

  async findByCategory(category: string): Promise<ClosetItem[]> {
    const allItems = await this.findAll();
    return allItems.filter((item) => item.attributes.category === category);
  }

  async create(
    item: Omit<ClosetItem, "id" | "createdAt">
  ): Promise<ClosetItem> {
    const newItem: ClosetItem = {
      ...item,
      id: `item-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date().toISOString().split("T")[0],
    };
    this.items.set(newItem.id, newItem);
    return newItem;
  }

  async update(id: string, updates: Partial<ClosetItem>): Promise<ClosetItem> {
    const item = this.items.get(id);
    if (!item) {
      throw new Error(`Item ${id} not found`);
    }
    const updated = {
      ...item,
      ...updates,
      updatedAt: new Date().toISOString().split("T")[0],
    };
    this.items.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    this.items.delete(id);
  }

  async updateVector(id: string, vector: number[]): Promise<void> {
    const item = this.items.get(id);
    if (item) {
      item.imageVector = vector;
      this.items.set(id, item);
    }
  }

  async findByImageId(imageId: string): Promise<ClosetItem | null> {
    const allItems = await this.findAll();
    return allItems.find((item) => item.imageId === imageId) || null;
  }

  async findSimilar(vector: number[], topK: number): Promise<ClosetItem[]> {
    const items = await this.findAll();
    return items
      .filter(
        (item) =>
          item.imageVector && item.imageVector.length === vector.length
      )
      .map((item) => ({
        item,
        similarity: cosineSimilarity(vector, item.imageVector!),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK)
      .map(({ item }) => item);
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
