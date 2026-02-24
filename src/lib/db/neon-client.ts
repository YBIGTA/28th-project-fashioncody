import { neon } from "@neondatabase/serverless";

export function getDb() {
  const databaseUrl = process.env.DATABASE_URL;
  if (!databaseUrl) {
    throw new Error("DATABASE_URL 환경 변수가 설정되지 않았습니다.");
  }
  return neon(databaseUrl);
}
