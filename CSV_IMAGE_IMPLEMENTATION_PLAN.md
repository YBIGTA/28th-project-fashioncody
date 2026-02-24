# CSV image_id 기반 이미지 저장 및 UI 표시 구현 계획

## 🎯 목표

1. CSV 파일의 `image_id`와 실제 이미지 파일(`image_id.jpg`) 연결
2. 이미지를 Vercel Blob에 저장 (1000장 이하, 무료 티어)
3. PostgreSQL에 `image_id`와 `imageUrl` 매핑 저장
4. 추천 결과에서 이미지를 UI에 표시

## 📌 핵심 요약

### Vercel Blob 연결 목적
- **목적**: 이미지 파일 저장소 (1000장 이하, 무료)
- **위치**: `src/app/api/closet/bulk-upload/route.ts`의 `uploadImageToBlob` 함수
- **결과**: `imageUrl` 반환 (예: `https://xxx.vercel-storage.com/closet/12345.jpg`)

### CSV의 image_id 연결
- **CSV**: `image_id` 컬럼 (예: "12345")
- **이미지 파일**: `12345.jpg`
- **매핑**: `image_id` → Vercel Blob 업로드 → `imageUrl` → PostgreSQL 저장

### UI 표시
- **추천 결과**: ML 서버가 UUID 반환
- **조회**: `candidateItems.find(i => i.id === UUID)` (이미 imageUrl 포함)
- **표시**: `<img src={item.imageUrl} />` → Vercel Blob에서 이미지 제공

---

## 📋 전체 흐름

```
[초기 설정]
CSV 파일 + 이미지 폴더
  ↓
일괄 업로드 API 호출
  ↓
각 image_id.jpg 파일 찾기
  ↓
Vercel Blob에 업로드
  ↓
PostgreSQL에 image_id + imageUrl 저장

[추천 요청]
ML 서버에서 image_id 반환
  ↓
image_id로 PostgreSQL 조회
  ↓
imageUrl 가져오기
  ↓
UI에 이미지 표시
```

---

## 🔧 구현 단계

### Step 1: 데이터베이스 스키마 수정

#### 1.1 PostgreSQL에 `image_id` 컬럼 추가
**파일**: `database/schema.sql` (마이그레이션 스크립트)

```sql
-- image_id 컬럼 추가
ALTER TABLE closet_items 
ADD COLUMN image_id VARCHAR(100);

-- 인덱스 생성 (빠른 조회를 위해)
CREATE INDEX idx_closet_items_image_id ON closet_items(image_id);

-- 기존 데이터가 있다면 NULL 허용
-- 새로 추가되는 데이터만 image_id 필수
```

**실행 방법**:
```bash
# PostgreSQL에 직접 실행
psql $DATABASE_URL -f database/add_image_id_column.sql

# 또는 마이그레이션 스크립트 생성
```

---

### Step 2: 타입 정의 수정

#### 2.1 ClosetItem 타입에 imageId 추가
**파일**: `src/lib/types/closet.ts`

```typescript
export type ClosetItem = {
  id: string;
  imageUrl: string;
  imageId?: string;  // CSV의 image_id 추가
  imageVector?: number[];
  attributes: ClosetItemAttributes;
  name?: string;
  tags?: string[];
  season?: ("spring" | "summer" | "fall" | "winter")[];
  createdAt: string;
  updatedAt?: string;
};
```

#### 2.2 레포지토리 인터페이스에 메서드 추가
**파일**: `src/lib/db/closet-repository.ts`

```typescript
export interface ClosetRepository {
  // 기존 메서드들...
  findAll(): Promise<ClosetItem[]>;
  findById(id: string): Promise<ClosetItem | null>;
  findByCategory(category: string): Promise<ClosetItem[]>;
  create(item: Omit<ClosetItem, "id" | "createdAt">): Promise<ClosetItem>;
  update(id: string, updates: Partial<ClosetItem>): Promise<ClosetItem>;
  delete(id: string): Promise<void>;
  updateVector(id: string, vector: number[]): Promise<void>;
  findSimilar(vector: number[], topK: number): Promise<ClosetItem[]>;
  
  // 새로 추가
  findByImageId(imageId: string): Promise<ClosetItem | null>;
}
```

---

### Step 3: PostgreSQL 레포지토리 구현

#### 3.1 findByImageId 메서드 구현
**파일**: `src/lib/db/postgres-repository.ts`

**위치**: `findByCategory` 메서드 다음에 추가

```typescript
async findByImageId(imageId: string): Promise<ClosetItem | null> {
  const rows = await this.sql`
    SELECT * FROM closet_items WHERE image_id = ${imageId}
  `;
  return rows.length > 0 ? rowToClosetItem(rows[0]) : null;
}
```

#### 3.2 create 메서드 수정 (image_id 저장)
**파일**: `src/lib/db/postgres-repository.ts`

**위치**: `create` 메서드 (38-81줄) 수정

```typescript
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
      ${item.imageId ?? null},  // image_id 추가
      ${attrs.category},
      ${attrs.detection_confidence},
      // ... 나머지 필드들
    )
    RETURNING *
  `;
  return rowToClosetItem(rows[0]);
}
```

#### 3.3 rowToClosetItem 함수 수정
**파일**: `src/lib/db/postgres-repository.ts`

**위치**: `rowToClosetItem` 함수 (161-194줄) 수정

```typescript
function rowToClosetItem(row: Record<string, unknown>): ClosetItem {
  // ... 기존 코드 ...
  
  return {
    id: row.id as string,
    imageUrl: row.image_url as string,
    imageId: row.image_id as string | undefined,  // image_id 추가
    attributes,
    name: row.name as string | undefined,
    tags: row.tags as string[] | undefined,
    season: row.season as ClosetItem["season"] | undefined,
    createdAt: formatDate(row.created_at),
    updatedAt: row.updated_at ? formatDate(row.updated_at) : undefined,
  };
}
```

---

### Step 4: CSV 기반 일괄 업로드 API 구현

#### 4.1 일괄 업로드 API 생성
**새 파일**: `src/app/api/closet/bulk-upload/route.ts`

```typescript
import { NextRequest, NextResponse } from "next/server";
import { put } from "@vercel/blob";
import { getRepository } from "@/lib/db/repository";
import type { ClosetItemAttributes } from "@/lib/types/closet";

const repository = getRepository();

/**
 * POST /api/closet/bulk-upload
 * CSV 파일과 이미지 파일들을 받아서 일괄 업로드
 * 
 * 요청 형식:
 * - FormData
 *   - csv: CSV 파일
 *   - images: 이미지 파일들 (FileList 또는 배열)
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const csvFile = formData.get("csv") as File;
    
    if (!csvFile) {
      return NextResponse.json(
        { error: "CSV 파일이 필요합니다." },
        { status: 400 }
      );
    }

    // 1. CSV 파싱
    const csvText = await csvFile.text();
    const rows = parseCSV(csvText);
    
    // 2. 이미지 파일들을 Map으로 변환 (image_id -> File)
    const imageMap = new Map<string, File>();
    const imageFiles = formData.getAll("images") as File[];
    
    for (const file of imageFiles) {
      const imageId = extractImageId(file.name); // "12345.jpg" -> "12345"
      if (imageId) {
        imageMap.set(imageId, file);
      }
    }

    const results = {
      success: 0,
      failed: 0,
      errors: [] as string[],
    };

    // 3. 각 CSV 행에 대해 처리
    for (const row of rows) {
      try {
        const imageId = row.image_id;
        if (!imageId) {
          results.failed++;
          results.errors.push(`image_id가 없는 행: ${JSON.stringify(row)}`);
          continue;
        }

        // 4. 이미지 파일 찾기
        const imageFile = imageMap.get(imageId);
        if (!imageFile) {
          results.failed++;
          results.errors.push(`이미지 파일을 찾을 수 없습니다: ${imageId}.jpg`);
          continue;
        }

        // 5. Vercel Blob에 업로드
        const imageUrl = await uploadImageToBlob(imageFile, imageId);

        // 6. CSV 데이터를 ClosetItemAttributes로 변환
        const attributes: ClosetItemAttributes = {
          category: row.category || "top",
          detection_confidence: parseFloat(row.detection_confidence || "0.9"),
          sub_type: row.sub_type,
          sub_type_confidence: row.sub_type_confidence 
            ? parseFloat(row.sub_type_confidence) 
            : undefined,
          color: row.color,
          color_confidence: row.color_confidence 
            ? parseFloat(row.color_confidence) 
            : undefined,
          // ... 나머지 필드들도 동일하게 변환
        };

        // 7. PostgreSQL에 저장
        await repository.create({
          imageUrl,
          imageId,  // image_id 저장
          attributes,
          name: row.name || `${row.color || ""} ${row.sub_type || row.category}`.trim(),
          tags: row.tags ? row.tags.split(",").map(t => t.trim()) : undefined,
          season: parseSeason(row.season),
        });

        results.success++;
      } catch (error) {
        results.failed++;
        results.errors.push(
          `행 처리 실패 (image_id: ${row.image_id}): ${error instanceof Error ? error.message : String(error)}`
        );
      }
    }

    return NextResponse.json({
      message: `${results.success}개의 아이템이 업로드되었습니다.`,
      results,
    });
  } catch (error) {
    console.error("Bulk upload error:", error);
    return NextResponse.json(
      { 
        error: "일괄 업로드 실패",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}

/**
 * CSV 텍스트를 파싱하여 객체 배열로 변환
 */
function parseCSV(csvText: string): Record<string, string>[] {
  const lines = csvText.split("\n").filter(line => line.trim());
  if (lines.length === 0) return [];

  // 첫 번째 줄은 헤더
  const headers = lines[0].split(",").map(h => h.trim());
  
  const rows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(",").map(v => v.trim());
    const row: Record<string, string> = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || "";
    });
    rows.push(row);
  }
  
  return rows;
}

/**
 * 파일 이름에서 image_id 추출
 * "12345.jpg" -> "12345"
 */
function extractImageId(fileName: string): string | null {
  const match = fileName.match(/^(.+)\.(jpg|jpeg|png)$/i);
  return match ? match[1] : null;
}

/**
 * Vercel Blob에 이미지 업로드
 */
async function uploadImageToBlob(
  file: File,
  imageId: string
): Promise<string> {
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    throw new Error("BLOB_READ_WRITE_TOKEN이 설정되지 않았습니다.");
  }

  const blob = await put(`closet/${imageId}.jpg`, file, {
    access: "public",
  });
  
  return blob.url;
}

/**
 * 시즌 문자열을 배열로 변환
 */
function parseSeason(seasonStr?: string): ("spring" | "summer" | "fall" | "winter")[] | undefined {
  if (!seasonStr) return undefined;
  
  const seasons = seasonStr.split(",").map(s => s.trim().toLowerCase());
  const validSeasons: ("spring" | "summer" | "fall" | "winter")[] = [];
  
  for (const s of seasons) {
    if (["spring", "summer", "fall", "winter"].includes(s)) {
      validSeasons.push(s as "spring" | "summer" | "fall" | "winter");
    }
  }
  
  return validSeasons.length > 0 ? validSeasons : undefined;
}
```

---

### Step 5: ML 서버에 image_id 전달 및 추천 결과 처리

#### 5.1 ML 서버 요청에 image_id 포함
**파일**: `src/app/api/recommend/route.ts`

**위치**: `tryMLRecommendation` 함수 (124-129줄) 수정

```typescript
// 현재
closet_items: candidateItems.map((item) => ({
  id: item.id,  // UUID
  vector: item.imageVector,
  attributes: item.attributes,
  season: item.season,
})),

// 수정: image_id도 포함
closet_items: candidateItems.map((item) => ({
  id: item.id,  // UUID (기존 호환성 유지)
  image_id: item.imageId,  // CSV의 image_id 추가
  vector: item.imageVector,
  attributes: item.attributes,
  season: item.season,
})),
```

#### 5.2 ML 서버가 image_id 반환하도록 수정 (선택)
**파일**: `ml-server/app/predictor.py`

**옵션 A**: ML 서버 수정 없이 Next API에서 처리 (권장)
- ML 서버는 UUID를 그대로 반환
- Next API에서 UUID로 candidateItems에서 찾기 (이미 imageUrl 포함)

**옵션 B**: ML 서버가 image_id 반환하도록 수정
- `predictor.py`에서 `item_id` 대신 `item.get("image_id")` 사용
- 더 복잡하지만 명확함

**권장**: 옵션 A (기존 코드 영향 최소화)

#### 5.3 추천 결과 처리 (기존 방식 유지)
**파일**: `src/app/api/recommend/route.ts`

**위치**: `tryMLRecommendation` 함수 (143-187줄)

```typescript
// 기존 방식 유지 (candidateItems에 이미 imageUrl 포함)
const mapped = rows
  .map((rec, index): UIRecommendation | null => {
    const outfitType =
      rec.outfit_type === "dress" || rec.dress_id ? "dress" : "two_piece";
    
    // UUID로 candidateItems에서 찾기 (이미 imageUrl 포함)
    const outer = rec.outer_id
      ? candidateItems.find((i) => i.id === rec.outer_id)
      : undefined;

    if (outfitType === "dress") {
      const dress = rec.dress_id
        ? candidateItems.find((i) => i.id === rec.dress_id)
        : undefined;
      if (!dress) return null;

      return {
        id: `rec_${index + 1}`,
        type: "dress",
        dress,  // 이미 imageUrl 포함
        outer,
        score: Number(rec.score ?? 0),
        reason: rec.reason || generateReason(mood, Number(rec.score ?? 0)),
      };
    }

    const top = rec.top_id
      ? candidateItems.find((i) => i.id === rec.top_id)
      : undefined;
    const bottom = rec.bottom_id
      ? candidateItems.find((i) => i.id === rec.bottom_id)
      : undefined;
    if (!top || !bottom) return null;

    return {
      id: `rec_${index + 1}`,
      type: "two_piece",
      top,  // 이미 imageUrl 포함
      bottom,  // 이미 imageUrl 포함
      outer,
      score: Number(rec.score ?? 0),
      reason: rec.reason || generateReason(mood, Number(rec.score ?? 0)),
    };
  })
  .filter((row): row is UIRecommendation => row !== null);

return mapped;
```

**핵심**: `candidateItems`에 이미 `imageUrl`이 포함되어 있으므로, UUID로 찾으면 바로 사용 가능

---

### Step 6: ML 서버에서 image_id 반환 확인

#### 6.1 ML 서버 응답 형식 확인
**파일**: `ml-server/app/predictor.py`

**현재 상황**:
- ML 서버는 `closet_items`에서 받은 `item.id`를 그대로 반환
- `item.id`는 PostgreSQL의 UUID (예: `"550e8400-e29b-41d4-a716-446655440000"`)
- ML 서버는 `image_id`를 직접 반환하지 않음

**해결 방안**: UUID → image_id 매핑 사용

#### 6.2 추천 결과 처리 수정
**파일**: `src/app/api/recommend/route.ts`

**방법 1**: UUID로 직접 조회 (기존 방식 유지)
```typescript
// ML 서버가 UUID를 반환하므로 기존 방식 그대로 사용
const top = rec.top_id
  ? candidateItems.find((i) => i.id === rec.top_id)  // UUID로 조회
  : undefined;
```

**방법 2**: UUID → image_id → imageUrl (새로운 방식)
```typescript
// UUID로 아이템 조회 (이미 imageId 포함)
const top = rec.top_id
  ? candidateItems.find((i) => i.id === rec.top_id)
  : undefined;

// top이 있으면 imageUrl이 이미 포함되어 있음
// UI에서 바로 사용 가능: <img src={top.imageUrl} />
```

**결론**: 기존 방식 유지 가능 (candidateItems에 이미 imageUrl 포함)

---

## 🔄 전체 흐름 상세

### 1. 초기 설정 (일괄 업로드)

```
[사용자]
CSV 파일 + 이미지 폴더 선택
  ↓
[프론트엔드]
POST /api/closet/bulk-upload
FormData:
  - csv: CSV 파일 (image_id, category, color, ...)
  - images: 이미지 파일들 (12345.jpg, 67890.jpg, ...)
  ↓
[Next API: bulk-upload/route.ts]
1. CSV 파싱 → rows 배열
2. 이미지 파일들을 Map으로 변환 (image_id -> File)
   - "12345.jpg" → image_id: "12345"
3. 각 row에 대해:
   a. image_id 추출 (예: "12345")
   b. image_id.jpg 파일 찾기 (imageMap.get("12345"))
   c. Vercel Blob에 업로드
      - put(`closet/12345.jpg`, file)
      - → imageUrl: "https://xxx.vercel-storage.com/closet/12345.jpg"
   d. PostgreSQL에 저장
      - image_id: "12345"
      - image_url: "https://xxx.vercel-storage.com/closet/12345.jpg"
      - attributes: {...}
  ↓
[PostgreSQL: closet_items 테이블]
{
  id: "550e8400-e29b-41d4-a716-446655440000" (UUID),
  image_id: "12345" (CSV에서),
  image_url: "https://xxx.vercel-storage.com/closet/12345.jpg",
  category: "top",
  color: "화이트",
  ...
}
  ↓
[완료]
성공/실패 결과 반환
```

### 2. 추천 요청 시

```
[사용자]
"미니멀 데이트" 입력 + 추천 요청
  ↓
[Next API: recommend/route.ts]
POST /api/recommend
  ↓
[PostgreSQL]
모든 아이템 조회 (repository.findAll())
  → candidateItems: [
      { id: "uuid-1", imageId: "12345", imageUrl: "https://...", ... },
      { id: "uuid-2", imageId: "67890", imageUrl: "https://...", ... },
      ...
    ]
  ↓
[ML 서버]
closet_items 전달 (id: UUID, image_id 포함)
  ↓
[ML 서버: predictor.py]
추천 결과 반환:
  {
    top_id: "uuid-1",      // UUID (PostgreSQL의 id)
    bottom_id: "uuid-2",  // UUID
    outer_id: "uuid-3",   // UUID
    score: 0.85
  }
  ↓
[Next API: recommend/route.ts]
UUID로 candidateItems에서 찾기:
  - candidateItems.find(i => i.id === "uuid-1")
  - → { id: "uuid-1", imageId: "12345", imageUrl: "https://...", ... }
  ↓
[UI: RecommendationResult.tsx]
<img src={result.top.imageUrl} />
  → "https://xxx.vercel-storage.com/closet/12345.jpg"
  ↓
[Vercel Blob]
이미지 제공
  ↓
[사용자]
이미지 표시됨 ✅
```

**핵심**: 
- ML 서버는 UUID를 반환하지만, candidateItems에 이미 imageUrl이 포함되어 있음
- UUID로 찾으면 바로 imageUrl 사용 가능
- image_id는 저장/관리 목적으로만 사용

---

## 📝 구현 체크리스트

### Step 1: 데이터베이스
- [ ] `database/schema.sql`에 `image_id` 컬럼 추가 스크립트 작성
- [ ] PostgreSQL에 마이그레이션 실행
- [ ] 인덱스 생성 확인

### Step 2: 타입 정의
- [ ] `src/lib/types/closet.ts`: `imageId` 추가
- [ ] `src/lib/db/closet-repository.ts`: `findByImageId` 인터페이스 추가

### Step 3: 레포지토리 구현
- [ ] `src/lib/db/postgres-repository.ts`: `findByImageId` 구현
- [ ] `src/lib/db/postgres-repository.ts`: `create` 메서드에 `image_id` 추가
- [ ] `src/lib/db/postgres-repository.ts`: `rowToClosetItem`에 `imageId` 추가

### Step 4: 일괄 업로드 API
- [ ] `src/app/api/closet/bulk-upload/route.ts` 생성
- [ ] CSV 파싱 로직 구현
- [ ] 이미지 파일 매핑 로직 구현
- [ ] Vercel Blob 업로드 로직 구현
- [ ] PostgreSQL 저장 로직 구현

### Step 5: 추천 결과 처리
- [ ] `src/app/api/recommend/route.ts`: `findByImageId` 사용
- [ ] ML 서버 응답 형식 확인 (image_id vs UUID)
- [ ] 매핑 로직 구현

### Step 6: 테스트
- [ ] CSV 파일 업로드 테스트
- [ ] 이미지 파일 매칭 테스트
- [ ] 추천 결과 이미지 표시 테스트

---

## 🎯 핵심 포인트

### 1. Vercel Blob 연결 목적
- **목적**: 이미지 파일 저장소
- **위치**: `src/app/api/closet/bulk-upload/route.ts`의 `uploadImageToBlob` 함수
- **결과**: `imageUrl` 반환 (예: `https://xxx.vercel-storage.com/closet/12345.jpg`)

### 2. CSV의 image_id 연결
- **CSV**: `image_id` 컬럼 (예: "12345")
- **이미지 파일**: `12345.jpg`
- **매핑**: `image_id` → `imageUrl` → PostgreSQL 저장

### 3. UI 표시
- **추천 결과**: ML 서버가 `image_id` 반환
- **조회**: `repository.findByImageId(image_id)`
- **표시**: `<img src={item.imageUrl} />`

---

## ⚠️ 주의사항

### 1. ML 서버 응답 형식
- ML 서버가 `UUID`를 반환하는지 `image_id`를 반환하는지 확인 필요
- UUID를 반환하면 매핑 테이블 또는 조회 로직 추가 필요

### 2. CSV 형식
- CSV의 컬럼명 확인 필요 (`image_id`, `category`, `color` 등)
- CSV 인코딩 확인 (UTF-8 권장)

### 3. 이미지 파일 형식
- `image_id.jpg` 형식 가정
- 다른 형식 (`.png`, `.jpeg`)도 지원하도록 확장 가능

---

## 🚀 실행 순서

1. **데이터베이스 마이그레이션** (1회)
   ```bash
   psql $DATABASE_URL -c "ALTER TABLE closet_items ADD COLUMN image_id VARCHAR(100);"
   psql $DATABASE_URL -c "CREATE INDEX idx_closet_items_image_id ON closet_items(image_id);"
   ```

2. **코드 수정** (Step 2-5)

3. **일괄 업로드 실행**
   ```typescript
   // 프론트엔드 또는 Postman에서
   const formData = new FormData();
   formData.append("csv", csvFile);
   imageFiles.forEach(file => formData.append("images", file));
   
   await fetch("/api/closet/bulk-upload", {
     method: "POST",
     body: formData,
   });
   ```

4. **추천 테스트**
   - 추천 요청
   - 이미지가 올바르게 표시되는지 확인
