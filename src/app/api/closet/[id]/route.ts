import { NextRequest, NextResponse } from "next/server";
import { getRepository } from "@/lib/db/repository";

const repository = getRepository();

/**
 * GET /api/closet/:id
 * 특정 아이템 조회
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const item = await repository.findById(id);

    if (!item) {
      return NextResponse.json(
        { error: "아이템을 찾을 수 없습니다." },
        { status: 404 }
      );
    }

    return NextResponse.json({ item });
  } catch (error) {
    console.error("Closet GET by ID error:", error);
    return NextResponse.json(
      { error: "아이템 조회에 실패했습니다." },
      { status: 500 }
    );
  }
}

/**
 * PUT /api/closet/:id
 * 아이템 수정
 */
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { name, tags, season, imageVector } = body;

    if (imageVector) {
      await repository.updateVector(id, imageVector);
    }

    const updates: Record<string, unknown> = {};
    if (name !== undefined) updates.name = name;
    if (tags !== undefined) updates.tags = tags;
    if (season !== undefined) updates.season = season;

    const updated = await repository.update(id, updates);

    return NextResponse.json({ item: updated });
  } catch (error) {
    console.error("Closet PUT error:", error);
    return NextResponse.json(
      { error: "아이템 수정에 실패했습니다." },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/closet/:id
 * 아이템 삭제
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    await repository.delete(id);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Closet DELETE error:", error);
    return NextResponse.json(
      { error: "아이템 삭제에 실패했습니다." },
      { status: 500 }
    );
  }
}
