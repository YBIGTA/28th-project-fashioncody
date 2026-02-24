import { NextRequest, NextResponse } from "next/server";
import { v2 as cloudinary } from "cloudinary";
import { getRepository } from "@/lib/db/repository";
import type { ClosetItemAttributes } from "@/lib/types/closet";

const repository = getRepository();

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

const IMAGE_ANALYSIS_MODEL_URL =
  process.env.IMAGE_ANALYSIS_MODEL_URL || "http://localhost:8001";
const CLIP_MODEL_URL =
  process.env.CLIP_MODEL_URL || "http://localhost:8002";

/**
 * POST /api/closet/upload
 * 이미지 업로드 및 아이템 등록
 *
 * 흐름:
 * 1. Cloudinary에 이미지 업로드
 * 2. ML 분석 서버 호출 (fallback: 기본 attributes)
 * 3. PostgreSQL에 아이템 저장
 * 4. CLIP 벡터 인코딩 (non-blocking, best-effort)
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return NextResponse.json(
        { error: "이미지 파일이 필요합니다." },
        { status: 400 }
      );
    }

    // 사용자 입력 속성 읽기
    const userCategory = formData.get("category") as string | null;
    const userColor = formData.get("color") as string | null;
    const userSubType = formData.get("sub_type") as string | null;

    // 1. Cloudinary에 이미지 업로드
    const imageUrl = await uploadImageToCloudinary(file);

    // 2. 속성 결정: 사용자 입력 우선, 없으면 ML 분석
    let attributes: ClosetItemAttributes;
    let mlSucceeded = false;

    if (userCategory && userColor) {
      // 사용자가 직접 입력한 속성 사용
      attributes = {
        category: userCategory as ClosetItemAttributes["category"],
        detection_confidence: 1.0,
        color: userColor,
        sub_type: userSubType || undefined,
      };
    } else {
      // ML 분석 서버 호출 (fallback 포함)
      const analysis = await analyzeImage(file);
      attributes = analysis.attributes;
      mlSucceeded = analysis.mlSucceeded;
    }

    // 이름 생성
    const name =
      attributes.sub_type && attributes.color
        ? `${attributes.color} ${attributes.sub_type}`
        : attributes.sub_type || attributes.category;

    // 태그 추출
    const tags: string[] = [];
    if (attributes.material) tags.push(...attributes.material.map((m) => m.value));
    if (attributes.print) tags.push(...attributes.print.map((p) => p.value));
    if (attributes.detail) tags.push(...attributes.detail.map((d) => d.value));

    // 3. PostgreSQL에 아이템 저장
    const newItem = await repository.create({
      imageUrl,
      attributes,
      name,
      tags: tags.length > 0 ? tags : undefined,
    });

    // 4. CLIP 벡터 인코딩 (non-blocking, best-effort)
    encodeAndStoreVector(newItem.id, file).catch((err) => {
      console.warn("CLIP 벡터 인코딩 실패 (무시됨):", err);
    });

    return NextResponse.json(
      {
        item: newItem,
        analysis: {
          mlAvailable: mlSucceeded,
          confidence: attributes.detection_confidence,
          fallback: !mlSucceeded,
        },
        message: "옷장 아이템이 등록되었습니다.",
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("Closet upload error:", error);
    return NextResponse.json(
      { error: "옷장 아이템 업로드에 실패했습니다." },
      { status: 500 }
    );
  }
}

/**
 * Cloudinary에 이미지 업로드
 */
async function uploadImageToCloudinary(file: File): Promise<string> {
  if (!process.env.CLOUDINARY_CLOUD_NAME) {
    console.warn("CLOUDINARY 설정이 없습니다. placeholder URL을 사용합니다.");
    return `https://picsum.photos/seed/${Date.now()}/400/500`;
  }

  const buffer = Buffer.from(await file.arrayBuffer());
  const base64 = `data:${file.type};base64,${buffer.toString("base64")}`;

  const result = await cloudinary.uploader.upload(base64, {
    folder: "closet",
    public_id: `${Date.now()}-${file.name.replace(/\.[^.]+$/, "")}`,
  });

  return result.secure_url;
}

/**
 * ML 분석 서버 호출 (fallback: 기본 attributes)
 */
async function analyzeImage(
  file: File
): Promise<{ attributes: ClosetItemAttributes; mlSucceeded: boolean }> {
  try {
    const formData = new FormData();
    formData.append("image", file);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(`${IMAGE_ANALYSIS_MODEL_URL}/analyze`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`분석 서버 응답 오류: ${response.status}`);
    }

    return { attributes: await response.json(), mlSucceeded: true };
  } catch (error) {
    console.warn("ML 분석 서버 호출 실패, fallback 사용:", error);
    return {
      attributes: {
        category: "top",
        detection_confidence: 0.5,
        sub_type: "기타",
        color: "기타",
      },
      mlSucceeded: false,
    };
  }
}

/**
 * CLIP 벡터 인코딩 후 DB에 저장 (non-blocking)
 */
async function encodeAndStoreVector(
  itemId: string,
  file: File
): Promise<void> {
  const formData = new FormData();
  formData.append("image", file);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);

  const response = await fetch(`${CLIP_MODEL_URL}/encode-image`, {
    method: "POST",
    body: formData,
    signal: controller.signal,
  });

  clearTimeout(timeout);

  if (!response.ok) {
    throw new Error(`CLIP 인코딩 오류: ${response.status}`);
  }

  const data = await response.json();
  await repository.updateVector(itemId, data.vector);
}
