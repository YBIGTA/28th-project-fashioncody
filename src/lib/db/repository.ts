import type { ClosetRepository } from "./closet-repository";
import { PostgresClosetRepository } from "./postgres-repository";

let instance: ClosetRepository | null = null;

export function getRepository(): ClosetRepository {
  if (!instance) {
    instance = new PostgresClosetRepository();
  }
  return instance;
}
