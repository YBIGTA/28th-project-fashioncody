import { NextRequest, NextResponse } from "next/server";
import {
  getHyperparams,
  updateHyperparams,
  getRecommendationHyperparams,
} from "@/lib/db/hyperparams-repository";

// 각 파라미터의 유효 범위
const BOUNDS: Record<string, [number, number]> = {
  alpha_tb: [0.2, 0.95],
  alpha_oi: [0.2, 0.95],
  mmr_lambda: [0.3, 0.95],
  beta_tb: [0.2, 0.8],
  lambda_tbset: [0.0, 0.3],
};

function clip(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * POST /api/feedback
 *
 * body:
 *   recommendation_id : string   — 추천 시 반환된 ID
 *   liked             : boolean  — true=좋아요, false=싫어요
 *
 * θ_used 방향으로 이동 (liked) 또는 반대 방향으로 이동 (disliked)
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { recommendation_id, liked } = body as {
      recommendation_id: string;
      liked: boolean;
    };

    if (!recommendation_id) {
      return NextResponse.json(
        { error: "recommendation_id가 필요합니다." },
        { status: 400 }
      );
    }
    if (typeof liked !== "boolean") {
      return NextResponse.json(
        { error: "liked는 boolean이어야 합니다." },
        { status: 400 }
      );
    }

    // 1. 이번 추천에 사용된 θ_used 조회
    const used = await getRecommendationHyperparams(recommendation_id);
    if (!used) {
      return NextResponse.json(
        { error: "해당 추천을 찾을 수 없습니다." },
        { status: 404 }
      );
    }

    // 2. 현재 기준점 θ 로드
    const baseline = await getHyperparams();
    const eta = baseline.eta;
    const sign = liked ? 1 : -1;

    // 3. θ_new = clip(θ ± η·diff, bounds)
    const keys = [
      "alpha_tb",
      "alpha_oi",
      "mmr_lambda",
      "beta_tb",
      "lambda_tbset",
    ] as const;
    const newParams = {} as Record<(typeof keys)[number], number>;

    for (const key of keys) {
      const diff = used[key] - baseline[key];
      const updated = baseline[key] + sign * eta * diff;
      const [lo, hi] = BOUNDS[key];
      newParams[key] = clip(updated, lo, hi);
    }

    // 4. DB 저장
    await updateHyperparams(newParams);

    return NextResponse.json({
      ok: true,
      updated: newParams,
    });
  } catch (error) {
    console.error("Feedback API error:", error);
    return NextResponse.json(
      {
        error: "피드백 처리에 실패했습니다.",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
