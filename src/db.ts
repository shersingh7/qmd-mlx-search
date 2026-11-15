/**
 * db.ts - Cross-runtime SQLite compatibility layer
 *
 * Provides a unified Database export that works under both Bun (bun:sqlite)
 * and Node.js (better-sqlite3). The APIs are nearly identical — the main
 * difference is the import path.
 *
 * On macOS, Apple's system SQLite is compiled with SQLITE_OMIT_LOAD_EXTENSION,
 * which prevents loading native extensions like sqlite-vec. When running under
 * Bun we call Database.setCustomSQLite() to swap in Homebrew's full-featured
 * SQLite build before creating any database instances.
 */

import { AsyncLocalStorage } from "node:async_hooks";

export const isBun = typeof globalThis.Bun !== "undefined";

let _Database: any;
let _sqliteVecLoad: ((db: any) => void) | null;

if (isBun) {
  // Dynamic string prevents tsc from resolving bun:sqlite on Node.js builds
  const bunSqlite = "bun:" + "sqlite";
  const BunDatabase = (await import(/* @vite-ignore */ bunSqlite)).Database;

  // See: https://bun.com/docs/runtime/sqlite#setcustomsqlite
  if (process.platform === "darwin") {
    const homebrewPaths = [
      "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib",  // Apple Silicon
      "/usr/local/opt/sqlite/lib/libsqlite3.dylib",     // Intel
    ];
    for (const p of homebrewPaths) {
      try {
        BunDatabase.setCustomSQLite(p);
        break;
      } catch {}
    }
  }

  _Database = BunDatabase;

  // setCustomSQLite may have silently failed — test that extensions actually work.
  try {
    const { getLoadablePath } = await import("sqlite-vec");
    const vecPath = getLoadablePath();
    const testDb = new BunDatabase(":memory:");
    try {
      testDb.loadExtension(vecPath);
    } finally {
      testDb.close();
    }
    _sqliteVecLoad = (db: any) => db.loadExtension(vecPath);
  } catch {
    // Vector search won't work, but BM25 and other operations are unaffected.
    _sqliteVecLoad = null;
  }
} else {
  _Database = (await import("better-sqlite3")).default;
  const sqliteVec = await import("sqlite-vec");
  _sqliteVecLoad = (db: any) => sqliteVec.load(db);
}

/**
 * Open a SQLite database. Works with both bun:sqlite and better-sqlite3.
 */
export function openDatabase(path: string): Database {
  const raw = new _Database(path) as Database;
  // Guard execution too, not just preparation: a statement prepared before an
  // awaited rebuild must not enter its savepoint from another async caller.
  const db: Database = {
    exec(sql) { assertDatabaseAccess(db); raw.exec(sql); },
    prepare(sql) {
      assertDatabaseAccess(db);
      const statement = raw.prepare(sql);
      return {
        run(...params) { assertDatabaseAccess(db); return statement.run(...params); },
        get(...params) { assertDatabaseAccess(db); return statement.get(...params); },
        all(...params) { assertDatabaseAccess(db); return statement.all(...params); },
      };
    },
    loadExtension(extension) { assertDatabaseAccess(db); raw.loadExtension(extension); },
    close() { assertDatabaseAccess(db); raw.close(); },
  };
  return db;
}

/**
 * Common subset of the Database interface used throughout QMD.
 */
export interface Database {
  exec(sql: string): void;
  prepare(sql: string): Statement;
  loadExtension(path: string): void;
  close(): void;
}

export interface Statement {
  run(...params: any[]): { changes: number; lastInsertRowid: number | bigint };
  get(...params: any[]): any;
  all(...params: any[]): any[];
}

export class DatabaseBusyError extends Error {
  readonly code = "QMD_DATABASE_BUSY";
  constructor(operation: string) {
    super(`Database is busy with ${operation}; retry after the operation settles or use a separate connection for committed reads.`);
    this.name = "DatabaseBusyError";
  }
}

const databaseOperations = new WeakMap<Database, { token: symbol; name: string }>();
const operationContext = new AsyncLocalStorage<ReadonlyMap<Database, symbol>>();

/** Refuse foreign reads as well as writes; both can observe uncommitted vec0 DDL. */
export function assertDatabaseAccess(db: Database): void {
  const active = databaseOperations.get(db);
  if (active && operationContext.getStore()?.get(db) !== active.token) {
    throw new DatabaseBusyError(active.name);
  }
}

/**
 * Own a connection across awaits without a waiting mutex. Internal nested work
 * reuses the token; unrelated calls fail before SQLite/config side effects.
 * Separate connections retain SQLite's normal committed-read/write isolation.
 */
export async function withExclusiveDatabaseOperation<T>(db: Database, name: string, work: () => Promise<T>): Promise<T> {
  assertDatabaseAccess(db);
  if (databaseOperations.has(db)) return work();
  const token = Symbol(name);
  const context = new Map(operationContext.getStore());
  context.set(db, token);
  databaseOperations.set(db, { token, name });
  try {
    return await operationContext.run(context, work);
  } finally {
    databaseOperations.delete(db);
  }
}

/** User/backend callbacks must not inherit the owner's connection privileges. */
export function withoutDatabaseOperation<T>(db: Database, work: () => T): T {
  const context = new Map(operationContext.getStore());
  context.delete(db);
  return operationContext.run(context, work);
}

/** Bind an internal checkpoint callback before crossing an unprivileged hook. */
export function bindDatabaseOperation<Args extends unknown[], T>(work: (...args: Args) => T): (...args: Args) => T {
  const context = operationContext.getStore() ?? new Map<Database, symbol>();
  return (...args) => operationContext.run(context, () => work(...args));
}

let savepointSequence = 0;

/** A nestable transaction boundary, including when a caller owns a rebuild. */
export function createSavepoint(db: Database): { commit(): void; rollback(): void } {
  const name = `qmd_write_${++savepointSequence}`;
  db.exec(`SAVEPOINT ${name}`);
  return {
    commit: () => db.exec(`RELEASE SAVEPOINT ${name}`),
    rollback: () => {
      db.exec(`ROLLBACK TO SAVEPOINT ${name}`);
      db.exec(`RELEASE SAVEPOINT ${name}`);
    },
  };
}

export function withSavepoint<T>(db: Database, write: () => T): T {
  const transaction = createSavepoint(db);
  try {
    const result = write();
    transaction.commit();
    return result;
  } catch (error) {
    transaction.rollback();
    throw error;
  }
}

/**
 * Load the sqlite-vec extension into a database.
 *
 * Throws with platform-specific fix instructions when the extension is
 * unavailable.
 */
export function loadSqliteVec(db: Database): void {
  if (!_sqliteVecLoad) {
    const hint = isBun && process.platform === "darwin"
      ? "On macOS with Bun, install Homebrew SQLite: brew install sqlite\n" +
        "Or install qmd with npm instead: npm install -g @tobilu/qmd"
      : "Ensure the sqlite-vec native module is installed correctly.";
    throw new Error(`sqlite-vec extension is unavailable. ${hint}`);
  }
  _sqliteVecLoad(db);
}
