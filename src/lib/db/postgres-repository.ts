import { getDb } from "./neon-client";
import type { ClosetRepository } from "./closet-repository";
import type {
  ClosetItem,
  ClosetItemAttributes,
  AttributeWithConfidence,
} from "@/lib/types/closet";

/**
 * Neon PostgreSQL + pgvector 기반 ClosetRepository 구현
 */
export class PostgresClosetRepository implements ClosetRepository {
  private get sql() {
    return getDb();
  }

  async findAll(): Promise<ClosetItem[]> {
    const rows = await this.sql`
      SELECT * FROM closet_items ORDER BY created_at DESC
    `;
    return rows.map(rowToClosetItem);
  }

  async findById(id: string): Promise<ClosetItem | null> {
    const rows = await this.sql`
      SELECT * FROM closet_items WHERE id = ${id}
    `;
    return rows.length > 0 ? rowToClosetItem(rows[0]) : null;
  }

  async findByCategory(category: string): Promise<ClosetItem[]> {
    const rows = await this.sql`
      SELECT * FROM closet_items WHERE category = ${category} ORDER BY created_at DESC
    `;
    return rows.map(rowToClosetItem);
  }

  async create(
    item: Omit<ClosetItem, "id" | "createdAt">
  ): Promise<ClosetItem> {
    const attrs = item.attributes;
    const rows = await this.sql`
      INSERT INTO closet_items (
        image_url, image_id, category, detection_confidence,
        sub_type, sub_type_confidence,
        color, color_confidence, sub_color, sub_color_confidence,
        sleeve_length, sleeve_length_confidence,
        length, length_confidence,
        fit, fit_confidence,
        collar, collar_confidence,
        material, print, detail,
        name, tags, season
      ) VALUES (
        ${item.imageUrl},
        ${item.imageId ?? null},
        ${attrs.category},
        ${attrs.detection_confidence},
        ${attrs.sub_type ?? null},
        ${attrs.sub_type_confidence ?? null},
        ${attrs.color ?? null},
        ${attrs.color_confidence ?? null},
        ${attrs.sub_color ?? null},
        ${attrs.sub_color_confidence ?? null},
        ${attrs.sleeve_length ?? null},
        ${attrs.sleeve_length_confidence ?? null},
        ${attrs.length ?? null},
        ${attrs.length_confidence ?? null},
        ${attrs.fit ?? null},
        ${attrs.fit_confidence ?? null},
        ${attrs.collar ?? null},
        ${attrs.collar_confidence ?? null},
        ${attrs.material ? JSON.stringify(attrs.material) : null},
        ${attrs.print ? JSON.stringify(attrs.print) : null},
        ${attrs.detail ? JSON.stringify(attrs.detail) : null},
        ${item.name ?? null},
        ${item.tags ?? null},
        ${item.season ?? null}
      )
      RETURNING *
    `;
    return rowToClosetItem(rows[0]);
  }

  async update(
    id: string,
    updates: Partial<ClosetItem>
  ): Promise<ClosetItem> {
    const sets: string[] = [];
    const values: unknown[] = [];
    let paramIdx = 1;

    if (updates.name !== undefined) {
      sets.push(`name = $${paramIdx++}`);
      values.push(updates.name);
    }
    if (updates.tags !== undefined) {
      sets.push(`tags = $${paramIdx++}`);
      values.push(updates.tags);
    }
    if (updates.season !== undefined) {
      sets.push(`season = $${paramIdx++}`);
      values.push(updates.season);
    }
    if (updates.imageUrl !== undefined) {
      sets.push(`image_url = $${paramIdx++}`);
      values.push(updates.imageUrl);
    }

    sets.push("updated_at = CURRENT_TIMESTAMP");

    // Use tagged template for safety - build a simple update
    const rows = await this.sql`
      UPDATE closet_items
      SET name = COALESCE(${updates.name ?? null}, name),
          tags = COALESCE(${updates.tags ?? null}, tags),
          season = COALESCE(${updates.season ?? null}, season),
          image_url = COALESCE(${updates.imageUrl ?? null}, image_url),
          updated_at = CURRENT_TIMESTAMP
      WHERE id = ${id}
      RETURNING *
    `;

    if (rows.length === 0) {
      throw new Error(`Item ${id} not found`);
    }
    return rowToClosetItem(rows[0]);
  }

  async delete(id: string): Promise<void> {
    await this.sql`DELETE FROM closet_items WHERE id = ${id}`;
  }

  async updateVector(id: string, vector: number[]): Promise<void> {
    const vectorStr = `[${vector.join(",")}]`;
    await this.sql`
      UPDATE closet_items
      SET image_vector = ${vectorStr}::vector,
          updated_at = CURRENT_TIMESTAMP
      WHERE id = ${id}
    `;
  }

  async findByImageId(imageId: string): Promise<ClosetItem | null> {
    const rows = await this.sql`
      SELECT * FROM closet_items WHERE image_id = ${imageId}
    `;
    return rows.length > 0 ? rowToClosetItem(rows[0]) : null;
  }

  async findSimilar(
    vector: number[],
    topK: number
  ): Promise<ClosetItem[]> {
    const vectorStr = `[${vector.join(",")}]`;
    const rows = await this.sql`
      SELECT *
      FROM closet_items
      WHERE image_vector IS NOT NULL
      ORDER BY image_vector <=> ${vectorStr}::vector
      LIMIT ${topK}
    `;
    return rows.map(rowToClosetItem);
  }
}

/**
 * DB row → ClosetItem 도메인 객체 변환
 */
function rowToClosetItem(row: Record<string, unknown>): ClosetItem {
  const attributes: ClosetItemAttributes = {
    category: row.category as ClosetItemAttributes["category"],
    detection_confidence: (row.detection_confidence as number) ?? 0,
    sub_type: row.sub_type as string | undefined,
    sub_type_confidence: row.sub_type_confidence as number | undefined,
    color: row.color as string | undefined,
    color_confidence: row.color_confidence as number | undefined,
    sub_color: row.sub_color as string | undefined,
    sub_color_confidence: row.sub_color_confidence as number | undefined,
    sleeve_length: row.sleeve_length as string | undefined,
    sleeve_length_confidence: row.sleeve_length_confidence as number | undefined,
    length: row.length as string | undefined,
    length_confidence: row.length_confidence as number | undefined,
    fit: row.fit as string | undefined,
    fit_confidence: row.fit_confidence as number | undefined,
    collar: row.collar as string | undefined,
    collar_confidence: row.collar_confidence as number | undefined,
    material: parseJsonb(row.material) as AttributeWithConfidence[] | undefined,
    print: parseJsonb(row.print) as AttributeWithConfidence[] | undefined,
    detail: parseJsonb(row.detail) as AttributeWithConfidence[] | undefined,
  };

  return {
    id: row.id as string,
    imageUrl: row.image_url as string,
    imageId: (row.image_id as string) || undefined,
    attributes,
    name: row.name as string | undefined,
    tags: row.tags as string[] | undefined,
    season: row.season as ClosetItem["season"] | undefined,
    createdAt: formatDate(row.created_at),
    updatedAt: row.updated_at ? formatDate(row.updated_at) : undefined,
  };
}

function parseJsonb(value: unknown): unknown {
  if (value === null || value === undefined) return undefined;
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return undefined;
    }
  }
  return value;
}

function formatDate(value: unknown): string {
  if (value instanceof Date) {
    return value.toISOString().split("T")[0];
  }
  if (typeof value === "string") {
    return value.split("T")[0];
  }
  return new Date().toISOString().split("T")[0];
}
