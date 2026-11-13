import type { Database } from "../db.js";

export type PendingEmbeddingDoc = { hash: string; path: string; bytes: number };

/** Point existence checks only: never combine a vec0 KNN MATCH with a join. */
export function physicalVectorExistsSql(db: Database, alias = "v"): string {
  if (!db.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vectors_vec'").get()) return "0";
  return `EXISTS (SELECT 1 FROM vectors_vec physical WHERE physical.hash_seq = ${alias}.hash || '_' || ${alias}.seq)`;
}

// Keep the candidate relation body-free. Unknown legacy chunk counts get one
// bounded rechunk; an active checkpoint is never a reason to select the corpus.
function pendingPredicate(db: Database, includeComplete: boolean): string {
  return `
    c.hash > ? AND EXISTS (
      SELECT 1 FROM documents d WHERE d.hash = c.hash AND d.active = 1
    ) AND (${includeComplete ? "1" : `
      e.hash IS NULL OR (
        SELECT COUNT(*) FROM content_vectors v
        WHERE v.hash = c.hash AND v.model = ? AND v.seq >= 0 AND v.seq < e.total_chunks
          AND ${physicalVectorExistsSql(db)}
      ) != e.total_chunks
    `})
  `;
}

export function getPendingEmbeddingStats(
  db: Database, model: string, strategy: string, includeComplete = false,
): { count: number; bytes: number } {
  const params = includeComplete ? [strategy, ""] : [strategy, "", model];
  return db.prepare(`
    SELECT COUNT(*) AS count, COALESCE(SUM(length(CAST(c.doc AS BLOB))), 0) AS bytes
    FROM content c
    LEFT JOIN content_chunk_expectations e ON e.hash = c.hash AND e.strategy = ?
    WHERE ${pendingPredicate(db, includeComplete)}
  `).get(...params) as { count: number; bytes: number };
}

/** Keyset paging: neither pending metadata nor document bodies grow with corpus size. */
export function* pendingEmbeddingBatches(
  db: Database, model: string, strategy: string, maxDocs: number, maxBytes: number,
): Generator<PendingEmbeddingDoc[]> {
  let afterHash = "";
  while (true) {
    const page = db.prepare(`
      WITH pending AS MATERIALIZED (
        SELECT c.hash, (SELECT MIN(d.path) FROM documents d WHERE d.hash = c.hash AND d.active = 1) AS path
        FROM content c
        LEFT JOIN content_chunk_expectations e ON e.hash = c.hash AND e.strategy = ?
        WHERE ${pendingPredicate(db, false)} ORDER BY c.hash LIMIT ?
      )
      SELECT p.hash, p.path, length(CAST(c.doc AS BLOB)) AS bytes
      FROM pending p JOIN content c ON c.hash = p.hash ORDER BY p.hash
    `).all(strategy, afterHash, model, maxDocs) as PendingEmbeddingDoc[];
    if (page.length === 0) return;
    afterHash = page[page.length - 1]!.hash;
    let batch: PendingEmbeddingDoc[] = [];
    let bytes = 0;
    for (const doc of page) {
      if (batch.length > 0 && bytes + doc.bytes > maxBytes) {
        yield batch;
        batch = [];
        bytes = 0;
      }
      batch.push(doc);
      bytes += doc.bytes;
    }
    if (batch.length > 0) yield batch;
  }
}

/** Existing indexes without a recorded chunker were built with the regex default. */
export function getStoredChunkStrategy(db: Database): string | undefined {
  if (!db.prepare("SELECT 1 FROM content_vectors LIMIT 1").get()) return undefined;
  const row = db.prepare("SELECT value FROM store_config WHERE key = 'embedding_chunk_strategy'").get() as { value: string } | undefined;
  return row?.value ?? "regex";
}
