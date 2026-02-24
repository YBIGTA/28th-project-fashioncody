/**
 * CSV + 이미지 일괄 업로드 스크립트
 *
 * 사용법:
 *   npx tsx scripts/bulk-seed.ts --csv data/items.csv --images data/images/
 *   npx tsx scripts/bulk-seed.ts --csv data/items.csv --images data/images/ --reset
 *
 * --reset 옵션:
 *   1) 기존 Vercel Blob 이미지 삭제
 *   2) closet_items 테이블 TRUNCATE
 *   3) 새로운 이미지로 재업로드
 *
 * CSV 컬럼 매핑:
 *   image_id     → DB image_id + 이미지 파일 매칭 (image_id.jpg)
 *   part         → DB category (상의→top, 하의→bottom, 아우터→outer, 원피스→dress)
 *   카테고리      → DB sub_type (블라우스, 니트웨어, 팬츠 등)
 *   색상         → DB color
 *   서브색상      → DB sub_color
 *   소매기장      → DB sleeve_length
 *   기장         → DB length
 *   핏           → DB fit
 *   옷깃         → DB collar
 *   소재_*       → DB material (one-hot → JSONB 배열)
 *   프린트_*     → DB print (one-hot → JSONB 배열)
 *   디테일_*     → DB detail (one-hot → JSONB 배열)
 *   날씨         → season 추론에 사용
 *
 * 이미지가 있는 행만 처리 (CSV 85K행 중 이미지 매칭되는 ~1000행만 업로드)
 */

import { config } from "dotenv";
import { neon } from "@neondatabase/serverless";
import { v2 as cloudinary } from "cloudinary";
import fs from "fs";
import path from "path";

// .env.local 로드
config({ path: path.join(process.cwd(), ".env.local") });

// ─── CSV 파싱 (quoted fields 지원) ──────────────────────────────

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  if (lines.length < 2) return [];

  const headers = parseCSVLine(lines[0]);
  const rows: Record<string, string>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    const row: Record<string, string> = {};
    for (let j = 0; j < headers.length; j++) {
      row[headers[j].trim()] = (values[j] ?? "").trim();
    }
    rows.push(row);
  }

  return rows;
}

function parseCSVLine(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') {
        current += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        current += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        fields.push(current);
        current = "";
      } else {
        current += ch;
      }
    }
  }
  fields.push(current);
  return fields;
}

// ─── 이미지 파일 탐색 ──────────────────────────────────────────

const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"];

function findImageFile(imageDir: string, imageId: string): string | null {
  for (const ext of IMAGE_EXTENSIONS) {
    const filePath = path.join(imageDir, `${imageId}${ext}`);
    if (fs.existsSync(filePath)) return filePath;
  }
  return null;
}

// ─── CSV 컬럼 매핑 ─────────────────────────────────────────────

const PART_TO_CATEGORY: Record<string, string> = {
  상의: "top",
  하의: "bottom",
  아우터: "outer",
  원피스: "dress",
};

const WEATHER_TO_SEASON: Record<string, string[]> = {
  폭염: ["summer"],
  더움: ["summer"],
  따뜻: ["spring", "summer"],
  선선: ["spring", "fall"],
  쌀쌀: ["fall", "winter"],
  추움: ["winter"],
  한파: ["winter"],
};

function toNullable(value: string | undefined): string | null {
  if (!value || value === "없음" || value === "null" || value === "") return null;
  return value;
}

/**
 * one-hot 인코딩된 컬럼들에서 값이 1인 항목을 추출
 * 예: 소재_면=1, 소재_니트=0 → [{"value":"면","confidence":0.8}]
 */
function extractOneHot(
  row: Record<string, string>,
  prefix: string
): string | null {
  const items: { value: string; confidence: number }[] = [];

  for (const [key, val] of Object.entries(row)) {
    if (key.startsWith(`${prefix}_`) && val === "1") {
      const label = key.slice(prefix.length + 1); // "소재_면" → "면"
      items.push({ value: label, confidence: 0.8 });
    }
  }

  return items.length > 0 ? JSON.stringify(items) : null;
}

function inferSeason(row: Record<string, string>): string[] {
  // 날씨 컬럼에서 시즌 추론
  const weather = row["날씨"];
  if (weather && WEATHER_TO_SEASON[weather]) {
    return WEATHER_TO_SEASON[weather];
  }

  // 소매기장 기반 fallback
  const sleeve = toNullable(row["소매기장"]);
  const part = row["part"];
  const seasons: string[] = [];

  if (sleeve && ["반팔", "민소매", "캡"].includes(sleeve)) {
    seasons.push("spring", "summer");
  } else if (sleeve === "긴팔") {
    seasons.push("spring", "fall", "winter");
  }

  if (part === "아우터") {
    seasons.push("fall", "winter");
  }

  const unique = Array.from(new Set(seasons));
  return unique.length > 0 ? unique.sort() : ["spring", "summer", "fall", "winter"];
}

// ─── Cloudinary 업로드 ────────────────────────────────────────

async function uploadToCloudinary(filePath: string, imageId: string): Promise<string> {
  const result = await cloudinary.uploader.upload(filePath, {
    folder: "closet",
    public_id: imageId,
    overwrite: true,
  });
  return result.secure_url;
}

// ─── 기존 Cloudinary 이미지 삭제 ─────────────────────────────────

async function deleteExistingCloudinaryImages(): Promise<number> {
  try {
    // closet 폴더의 모든 리소스 삭제
    const result = await cloudinary.api.delete_resources_by_prefix("closet/");
    const deleted = Object.keys(result.deleted || {}).length;
    console.log(`Cloudinary 이미지 ${deleted}개 삭제 완료`);
    return deleted;
  } catch (err) {
    console.warn("Cloudinary 이미지 삭제 실패 (무시):", err);
    return 0;
  }
}

// ─── DB 삽입 ───────────────────────────────────────────────────

async function insertItem(
  sql: ReturnType<typeof neon>,
  row: Record<string, string>,
  imageUrl: string,
  imageId: string
) {
  const category = PART_TO_CATEGORY[row["part"]] || "top";
  const subType = toNullable(row["카테고리"]);
  const color = toNullable(row["색상"]);
  const subColor = toNullable(row["서브색상"]);
  const sleeveLength = toNullable(row["소매기장"]);
  const length = toNullable(row["기장"]);
  const fit = toNullable(row["핏"]);
  const collar = toNullable(row["옷깃"]);

  const material = extractOneHot(row, "소재");
  const print = extractOneHot(row, "프린트");
  const detail = extractOneHot(row, "디테일");

  const name = [color, subType || category].filter(Boolean).join(" ");
  const season = inferSeason(row);

  // 태그 생성: 스타일 + 서브스타일 + 소재/프린트/디테일 항목
  const tags: string[] = [];
  const style = toNullable(row["스타일"]);
  const subStyle = toNullable(row["서브스타일"]);
  if (style) tags.push(style);
  if (subStyle) tags.push(subStyle);

  await sql`
    INSERT INTO closet_items (
      image_url, image_id, category, detection_confidence,
      sub_type, color, sub_color,
      sleeve_length, length, fit, collar,
      material, print, detail,
      name, tags, season
    ) VALUES (
      ${imageUrl},
      ${imageId},
      ${category},
      ${0.9},
      ${subType},
      ${color},
      ${subColor},
      ${sleeveLength},
      ${length},
      ${fit},
      ${collar},
      ${material},
      ${print},
      ${detail},
      ${name},
      ${tags.length > 0 ? tags : null},
      ${season}
    )
  `;
}

// ─── 메인 ──────────────────────────────────────────────────────

async function main() {
  // CLI 인자 파싱
  const args = process.argv.slice(2);
  let csvPath = "";
  let imageDir = "";
  let resetMode = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--csv" && args[i + 1]) csvPath = args[++i];
    if (args[i] === "--images" && args[i + 1]) imageDir = args[++i];
    if (args[i] === "--reset") resetMode = true;
  }

  if (!csvPath || !imageDir) {
    console.error("사용법: npx tsx scripts/bulk-seed.ts --csv <csv파일> --images <이미지폴더> [--reset]");
    console.error("예시:  npx tsx scripts/bulk-seed.ts --csv data/items.csv --images data/images/ --reset");
    process.exit(1);
  }

  // 경로 검증
  csvPath = path.resolve(csvPath);
  imageDir = path.resolve(imageDir);

  if (!fs.existsSync(csvPath)) {
    console.error(`CSV 파일을 찾을 수 없습니다: ${csvPath}`);
    process.exit(1);
  }
  if (!fs.existsSync(imageDir)) {
    console.error(`이미지 폴더를 찾을 수 없습니다: ${imageDir}`);
    process.exit(1);
  }

  // 환경변수 검증
  const databaseUrl = process.env.DATABASE_URL;
  if (!databaseUrl) {
    console.error("DATABASE_URL이 .env.local에 설정되지 않았습니다.");
    process.exit(1);
  }
  if (!process.env.CLOUDINARY_CLOUD_NAME || !process.env.CLOUDINARY_API_KEY || !process.env.CLOUDINARY_API_SECRET) {
    console.error("CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET이 .env.local에 설정되지 않았습니다.");
    process.exit(1);
  }

  cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
  });

  const sql = neon(databaseUrl);

  // ─── --reset 모드: 기존 데이터 정리 ───────────────────────────
  if (resetMode) {
    console.log("=== RESET 모드: 기존 데이터 정리 ===\n");

    // 1) 기존 Cloudinary 이미지 삭제
    await deleteExistingCloudinaryImages();

    // 2) DB 테이블 초기화
    console.log("\ncloset_items 테이블 초기화 중...");
    await sql`TRUNCATE closet_items`;
    console.log("테이블 초기화 완료!\n");
    console.log("=== 새로운 데이터 시드 시작 ===\n");
  }

  // 1. 이미지 파일 목록을 먼저 수집 (빠른 필터링용)
  const imageFiles = new Set<string>();
  for (const file of fs.readdirSync(imageDir)) {
    const match = file.match(/^(.+)\.(jpg|jpeg|png|webp)$/i);
    if (match) imageFiles.add(match[1]);
  }
  console.log(`이미지 폴더에서 ${imageFiles.size}개 파일 발견`);

  // 2. CSV 파싱
  const csvText = fs.readFileSync(csvPath, "utf-8");
  const allRows = parseCSV(csvText);
  console.log(`CSV에서 ${allRows.length}개 행 읽음`);

  // 3. 이미지가 있는 행만 필터링 + 동일 image_id 중복 제거
  const seenImageIds = new Set<string>();
  const rows = allRows.filter((row) => {
    const imageId = row.image_id;
    if (!imageId || !imageFiles.has(imageId)) return false;
    if (seenImageIds.has(imageId)) return false; // 동일 이미지 중복 방지
    seenImageIds.add(imageId);
    return true;
  });
  console.log(`이미지 매칭되는 행: ${rows.length}개 (중복 제거 후 처리 대상)`);
  console.log("---");

  // 4. 배치 처리 (배치 간 1초 딜레이로 rate limit 방지)
  const BATCH_SIZE = 5;
  const BATCH_DELAY_MS = 1000;
  let success = 0;
  let failed = 0;
  const errors: string[] = [];

  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    const batch = rows.slice(i, i + BATCH_SIZE);
    const promises = batch.map(async (row, batchIdx) => {
      const globalIdx = i + batchIdx + 1;
      const imageId = row.image_id;

      const imagePath = findImageFile(imageDir, imageId);
      if (!imagePath) {
        failed++;
        errors.push(`${imageId}: 이미지 파일 없음`);
        return;
      }

      try {
        const imageUrl = await uploadToCloudinary(imagePath, imageId);
        await insertItem(sql, row, imageUrl, imageId);

        success++;
        const color = toNullable(row["색상"]) || "";
        const subType = toNullable(row["카테고리"]) || row["part"];
        console.log(`  [${globalIdx}/${rows.length}] ${color} ${subType} (${row["part"]}) - ${imageId}`);
      } catch (err) {
        failed++;
        const msg = err instanceof Error ? err.message : String(err);
        errors.push(`${imageId}: ${msg}`);
        console.error(`  [${globalIdx}/${rows.length}] 실패 - ${imageId}: ${msg}`);
      }
    });

    await Promise.all(promises);

    // Rate limit 방지: 배치 간 딜레이
    if (i + BATCH_SIZE < rows.length) {
      await new Promise((r) => setTimeout(r, BATCH_DELAY_MS));
    }
  }

  // 결과 출력
  console.log("\n===== 결과 =====");
  console.log(`성공: ${success}개`);
  console.log(`실패: ${failed}개`);
  console.log(`전체 CSV 행: ${allRows.length}개`);
  console.log(`이미지 없어서 건너뜀: ${allRows.length - rows.length}개`);

  if (errors.length > 0 && errors.length <= 20) {
    console.log(`\n오류 상세:`);
    errors.forEach((e) => console.log(`  - ${e}`));
  } else if (errors.length > 20) {
    console.log(`\n오류 ${errors.length}건 (처음 20건만 표시):`);
    errors.slice(0, 20).forEach((e) => console.log(`  - ${e}`));
  }

  console.log("\n완료!");
}

main().catch((err) => {
  console.error("스크립트 실행 실패:", err);
  process.exit(1);
});
