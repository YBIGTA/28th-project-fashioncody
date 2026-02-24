/**
 * 하이퍼파라미터 기반 피드백 학습 시스템 DB 레이어
 *
 * user_hyperparams  : 현재 기준점(θ) 저장 (단일 행 싱글톤)
 * recommendation_history : 추천 시 사용한 θ_used 기록
 */

import { getDb } from "./neon-client";

export type Hyperparams = {
  alpha_tb: number; // top-bottom 색상 vs 임베딩 비중 [0.20, 0.95]
  alpha_oi: number; // outer-inner 색상 vs 임베딩 비중 [0.20, 0.95]
  mmr_lambda: number; // MMR quality vs diversity       [0.30, 0.95]
  beta_tb: number; // outer-top vs outer-bottom 균형 [0.20, 0.80]
  lambda_tbset: number; // inner 응집력 반영 비중          [0.00, 0.30]
  sigma: number; // 탐색 노이즈 표준편차
  eta: number; // 학습률 (기준점 갱신 폭)
};

function rowToHyperparams(row: Record<string, unknown>): Hyperparams {
  return {
    alpha_tb: Number(row.alpha_tb),
    alpha_oi: Number(row.alpha_oi),
    mmr_lambda: Number(row.mmr_lambda),
    beta_tb: Number(row.beta_tb),
    lambda_tbset: Number(row.lambda_tbset),
    sigma: Number(row.sigma),
    eta: Number(row.eta),
  };
}

/**
 * 현재 하이퍼파라미터 기준점 조회.
 * 행이 없으면 기본값으로 초기 행 생성 후 반환.
 */
export async function getHyperparams(): Promise<Hyperparams> {
  const sql = getDb();
  const rows = await sql`SELECT * FROM user_hyperparams LIMIT 1`;
  if (rows.length > 0) {
    return rowToHyperparams(rows[0]);
  }
  const inserted =
    await sql`INSERT INTO user_hyperparams DEFAULT VALUES RETURNING *`;
  return rowToHyperparams(inserted[0]);
}

/**
 * 하이퍼파라미터 기준점 갱신.
 */
export async function updateHyperparams(
  params: Omit<Hyperparams, "sigma" | "eta">
): Promise<void> {
  const sql = getDb();
  await sql`
    UPDATE user_hyperparams
    SET
      alpha_tb     = ${params.alpha_tb},
      alpha_oi     = ${params.alpha_oi},
      mmr_lambda   = ${params.mmr_lambda},
      beta_tb      = ${params.beta_tb},
      lambda_tbset = ${params.lambda_tbset},
      updated_at   = CURRENT_TIMESTAMP
    WHERE id = (SELECT id FROM user_hyperparams LIMIT 1)
  `;
}

/**
 * 추천 결과를 recommendation_history에 저장.
 * @returns 생성된 recommendation_history.id
 */
export async function saveRecommendation(data: {
  mood: string;
  weather_data: object;
  recommended_items: object;
  hyperparams_used: Hyperparams;
}): Promise<string> {
  const sql = getDb();
  const rows = await sql`
    INSERT INTO recommendation_history
      (mood, weather_data, recommended_items, hyperparams_used)
    VALUES (
      ${data.mood},
      ${JSON.stringify(data.weather_data)},
      ${JSON.stringify(data.recommended_items)},
      ${JSON.stringify(data.hyperparams_used)}
    )
    RETURNING id
  `;
  return rows[0].id as string;
}

/**
 * 특정 추천에 사용된 하이퍼파라미터 조회.
 */
export async function getRecommendationHyperparams(
  recId: string
): Promise<Hyperparams | null> {
  const sql = getDb();
  const rows = await sql`
    SELECT hyperparams_used
    FROM recommendation_history
    WHERE id = ${recId}
  `;
  if (rows.length === 0 || !rows[0].hyperparams_used) {
    return null;
  }
  const raw = rows[0].hyperparams_used as Record<string, unknown>;
  return rowToHyperparams(raw);
}
