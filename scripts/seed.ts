import { config } from "dotenv";
import { neon } from "@neondatabase/serverless";
import fs from "fs";
import path from "path";

// .env.local 로드
config({ path: path.join(process.cwd(), ".env.local") });

type AttributeWithConfidence = {
  value: string;
  confidence: number;
};

type DummyItem = {
  category: string;
  detection_confidence: number;
  sub_type?: string;
  sub_type_confidence?: number;
  color?: string;
  color_confidence?: number;
  sub_color?: string;
  sub_color_confidence?: number;
  sleeve_length?: string;
  sleeve_length_confidence?: number;
  length?: string;
  length_confidence?: number;
  fit?: string;
  fit_confidence?: number;
  collar?: string;
  collar_confidence?: number;
  material?: AttributeWithConfidence[];
  print?: AttributeWithConfidence[];
  detail?: AttributeWithConfidence[];
};

function inferSeason(
  attrs: DummyItem
): string[] {
  const seasons: string[] = [];

  if (attrs.sleeve_length === "반팔" || attrs.sleeve_length === "민소매" || attrs.sleeve_length === "캡") {
    seasons.push("spring", "summer");
  } else if (attrs.sleeve_length === "긴팔") {
    seasons.push("spring", "fall", "winter");
  } else if (attrs.sleeve_length === "7부소매") {
    seasons.push("spring", "summer", "fall");
  }

  if (attrs.material) {
    const materialValues = attrs.material.map((m) => m.value);
    if (materialValues.some((m) => m.includes("니트") || m.includes("울") || m.includes("무스탕") || m.includes("퍼"))) {
      seasons.push("fall", "winter");
    }
    if (materialValues.some((m) => m.includes("린넨") || m.includes("면"))) {
      seasons.push("spring", "summer");
    }
  }

  if (attrs.category === "outer") {
    seasons.push("fall", "winter");
  }

  const uniqueSeasons = Array.from(new Set(seasons));
  return uniqueSeasons.length > 0
    ? uniqueSeasons.sort()
    : ["spring", "summer", "fall", "winter"];
}

async function seed() {
  const databaseUrl = process.env.DATABASE_URL;
  if (!databaseUrl) {
    console.error("DATABASE_URL 환경 변수가 설정되지 않았습니다.");
    console.error(".env.local 파일에 DATABASE_URL을 설정해주세요.");
    process.exit(1);
  }

  const sql = neon(databaseUrl);

  // dummy.json 로드
  const filePath = path.join(process.cwd(), "dummy.json");
  const fileContents = fs.readFileSync(filePath, "utf-8");
  const items: DummyItem[] = JSON.parse(fileContents);

  console.log(`${items.length}개의 아이템을 시드합니다...`);

  for (let i = 0; i < items.length; i++) {
    const attrs = items[i];
    const imageUrl = `https://picsum.photos/seed/closet-${i + 1}/400/500`;
    const name =
      attrs.sub_type && attrs.color
        ? `${attrs.color} ${attrs.sub_type}`
        : attrs.sub_type || attrs.category;

    const tags: string[] = [];
    if (attrs.material) tags.push(...attrs.material.map((m) => m.value));
    if (attrs.print) tags.push(...attrs.print.map((p) => p.value));
    if (attrs.detail) tags.push(...attrs.detail.map((d) => d.value));

    const season = inferSeason(attrs);

    await sql`
      INSERT INTO closet_items (
        image_url, category, detection_confidence,
        sub_type, sub_type_confidence,
        color, color_confidence, sub_color, sub_color_confidence,
        sleeve_length, sleeve_length_confidence,
        length, length_confidence,
        fit, fit_confidence,
        collar, collar_confidence,
        material, print, detail,
        name, tags, season
      ) VALUES (
        ${imageUrl},
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
        ${name},
        ${tags.length > 0 ? tags : null},
        ${season}
      )
    `;

    console.log(`  [${i + 1}/${items.length}] ${name} (${attrs.category})`);
  }

  console.log("시드 완료!");
}

seed().catch((err) => {
  console.error("시드 실패:", err);
  process.exit(1);
});
