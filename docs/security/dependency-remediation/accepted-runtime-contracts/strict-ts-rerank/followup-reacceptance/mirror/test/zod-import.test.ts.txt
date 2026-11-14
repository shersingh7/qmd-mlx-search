import { it, expect } from 'vitest';
import { z } from 'zod';
import { JSONRPCMessageSchema } from '@modelcontextprotocol/sdk/types.js';

it('preserves the real Zod named ESM namespace and non-coercing schemas', () => {
  expect(typeof z).toBe('object');
  const schema = z.object({ limit: z.number().int().min(1).max(100) });
  expect(schema.safeParse({ limit: 1 }).success).toBe(true);
  expect(schema.safeParse({ limit: 100 }).success).toBe(true);
  for (const limit of ['1', 0, 101, 1.5, null]) {
    expect(schema.safeParse({ limit }).success).toBe(false);
  }
  expect(JSONRPCMessageSchema.safeParse({ jsonrpc: '2.0', id: 1, method: 'tools/list' }).success).toBe(true);
  expect(JSONRPCMessageSchema.safeParse({ jsonrpc: '2.0', id: 1 }).success).toBe(false);
});
