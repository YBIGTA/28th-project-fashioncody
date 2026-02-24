import { NextRequest, NextResponse } from "next/server";
import { getRepository } from "@/lib/db/repository";

const repository = getRepository();

/**
 * GET /api/closet
 * 옷장 아이템 목록 조회
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const category = searchParams.get("category") || undefined;

    const items = category
      ? await repository.findByCategory(category)
      : await repository.findAll();

    return NextResponse.json({ items });
  } catch (error) {
    console.error("Closet GET error:", error);
    return NextResponse.json(
      { error: "옷장 조회에 실패했습니다." },
      { status: 500 }
    );
  }
}

/**
 * POST /api/closet
 * 옷장 아이템 생성
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { imageUrl, attributes, name, tags, season } = body;

    if (!imageUrl || !attributes) {
      return NextResponse.json(
        { error: "imageUrl과 attributes는 필수입니다." },
        { status: 400 }
      );
    }

    const newItem = await repository.create({
      imageUrl,
      attributes,
      name,
      tags,
      season,
    });

    return NextResponse.json({ item: newItem }, { status: 201 });
  } catch (error) {
    console.error("Closet POST error:", error);
    return NextResponse.json(
      { error: "옷장 아이템 생성에 실패했습니다." },
      { status: 500 }
    );
  }
}
