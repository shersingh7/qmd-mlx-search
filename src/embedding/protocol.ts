/** Strict little-endian framing and serving-identity validation. */
import {
  MAX_EMBEDDING_DIMENSIONS, assertEmbeddingCompatibility, validateEmbeddingDescriptor,
  type EmbeddingDescriptor,
} from "./contract.js";

export class EmbeddingProtocolError extends Error {
  constructor(message: string) { super(message); this.name = "EmbeddingProtocolError"; }
}

export const MAX_EMBEDDING_VALUES = 4_194_304;
const MAX_VECTOR_COUNT = 65_536;
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

export interface EmbeddingResponseExpectations {
  count?: number;
  dims?: number;
  descriptor?: EmbeddingDescriptor;
}

export interface DecodedBinaryPayload {
  count: number;
  dims: number;
  vectors: number[][];
  rawFloats: Float32Array;
}

function validateShape(count: number, dims: number, expected?: EmbeddingResponseExpectations): number {
  if (!Number.isSafeInteger(count) || count < 0 || count > MAX_VECTOR_COUNT) {
    throw new EmbeddingProtocolError("Invalid embedding count");
  }
  if (!Number.isSafeInteger(dims) || dims < 0 || dims > MAX_EMBEDDING_DIMENSIONS || (count > 0 && dims === 0)) {
    throw new EmbeddingProtocolError("Invalid embedding dimensions");
  }
  const values = count * dims;
  if (!Number.isSafeInteger(values) || values > MAX_EMBEDDING_VALUES) {
    throw new EmbeddingProtocolError("Embedding response exceeds value limit");
  }
  if (expected?.count !== undefined && count !== expected.count) {
    throw new EmbeddingProtocolError(`Embedding count ${count} does not match expected ${expected.count}`);
  }
  const expectedDims = expected?.dims ?? expected?.descriptor?.outputDimensions;
  if (expectedDims !== undefined && dims !== expectedDims) {
    throw new EmbeddingProtocolError(`Embedding dims ${dims} do not match expected ${expectedDims}`);
  }
  return values;
}

/** Binary body is unchanged: int32 count, int32 dims, then count*dims float32. */
export function decodeBinaryEmbeddings(buffer: ArrayBuffer, expected?: EmbeddingResponseExpectations): DecodedBinaryPayload {
  if (buffer.byteLength < 8) throw new EmbeddingProtocolError("Binary response truncated: less than 8-byte header");
  const view = new DataView(buffer);
  const count = view.getInt32(0, true);
  const dims = view.getInt32(4, true);
  const values = validateShape(count, dims, expected);
  const expectedBytes = 8 + values * 4;
  // Validate even empty frames: no early return that can hide trailing bytes.
  if (buffer.byteLength !== expectedBytes) {
    throw new EmbeddingProtocolError(`Binary payload byte length mismatch: expected ${expectedBytes}, received ${buffer.byteLength}`);
  }
  const floats = LITTLE_ENDIAN ? new Float32Array(buffer, 8, values) : new Float32Array(values);
  for (let i = 0; i < values; i++) {
    if (!LITTLE_ENDIAN) floats[i] = view.getFloat32(8 + i * 4, true);
    if (!Number.isFinite(floats[i])) throw new EmbeddingProtocolError(`Non-finite float at offset ${i}`);
  }
  const vectors = Array.from({ length: count }, (_, i) => Array.from(floats.subarray(i * dims, (i + 1) * dims)));
  return { count, dims, vectors, rawFloats: floats };
}

export function encodeBinaryEmbeddings(vectors: number[][] | Float32Array[], dims: number): Uint8Array<ArrayBuffer> {
  const count = vectors.length;
  const values = validateShape(count, dims);
  // Validate before allocation or conversion, including float32 overflow.
  for (let i = 0; i < count; i++) {
    const vector = vectors[i];
    if ((!Array.isArray(vector) && !(vector instanceof Float32Array)) || vector.length !== dims) {
      throw new EmbeddingProtocolError(`Vector at index ${i} does not have ${dims} dimensions`);
    }
    for (const value of vector) {
      if (typeof value !== "number" || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
        throw new EmbeddingProtocolError(`Non-finite float32 value in vector ${i}`);
      }
    }
  }
  const buffer = new ArrayBuffer(8 + values * 4);
  const view = new DataView(buffer);
  view.setInt32(0, count, true);
  view.setInt32(4, dims, true);
  for (let i = 0; i < count; i++) {
    for (let j = 0; j < dims; j++) view.setFloat32(8 + (i * dims + j) * 4, vectors[i]![j]!, true);
  }
  return new Uint8Array(buffer);
}

function responseDescriptor(data: unknown, expected?: EmbeddingDescriptor): EmbeddingDescriptor | undefined {
  if (data === undefined) {
    if (expected) throw new EmbeddingProtocolError("Missing serving embedding descriptor");
    return undefined; // Legacy standalone helper compatibility, never used by the production client.
  }
  try {
    const actual = validateEmbeddingDescriptor(data);
    if (actual.backend !== "mlx") throw new Error("Serving embedding backend must be mlx");
    if (expected) assertEmbeddingCompatibility(expected, actual, "MLX response");
    return actual;
  } catch (error) {
    throw new EmbeddingProtocolError(error instanceof Error ? error.message : "Invalid serving embedding descriptor");
  }
}

/** Header value is urllib.parse.quote(JSON, safe=''), not base64 or raw JSON. */
export function decodeEmbeddingDescriptorHeader(value: string | null, expected?: EmbeddingDescriptor): EmbeddingDescriptor | undefined {
  if (value === null) return responseDescriptor(undefined, expected);
  try {
    if (value.length > 262_144) throw new Error("Embedding descriptor header exceeds byte limit");
    return responseDescriptor(JSON.parse(decodeURIComponent(value)), expected);
  } catch (error) {
    if (error instanceof EmbeddingProtocolError) throw error;
    throw new EmbeddingProtocolError("Invalid percent-encoded embedding descriptor header");
  }
}

export function validateJsonEmbedResponse(data: unknown, expected?: EmbeddingResponseExpectations): {
  embeddings: number[][];
  model: string;
  dims: number;
  descriptor?: EmbeddingDescriptor;
} {
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new EmbeddingProtocolError("Expected a JSON object");
  const obj = data as Record<string, unknown>;
  if (!Array.isArray(obj.embeddings)) throw new EmbeddingProtocolError("'embeddings' must be an array");
  if (typeof obj.dims !== "number") throw new EmbeddingProtocolError("'dims' must be an integer");
  validateShape(obj.embeddings.length, obj.dims, expected);
  const descriptor = responseDescriptor(obj.descriptor, expected?.descriptor);
  if (typeof obj.model !== "string" || !obj.model.trim() || obj.model.length > 8192) throw new EmbeddingProtocolError("'model' must be a nonempty string");
  if (descriptor && (obj.model !== descriptor.model || obj.dims !== descriptor.outputDimensions)) {
    throw new EmbeddingProtocolError("Response model/dimensions disagree with serving descriptor");
  }
  const embeddings = obj.embeddings as number[][];
  for (let i = 0; i < embeddings.length; i++) {
    const vector = embeddings[i];
    if (!Array.isArray(vector) || vector.length !== obj.dims) throw new EmbeddingProtocolError(`Invalid dimensions at embedding ${i}`);
    for (let j = 0; j < vector.length; j++) {
      if (typeof vector[j] !== "number" || !Number.isFinite(vector[j]) || !Number.isFinite(Math.fround(vector[j]!))) {
        throw new EmbeddingProtocolError(`Non-finite float32 value at embedding ${i}, dim ${j}`);
      }
    }
  }
  return { embeddings, model: obj.model, dims: obj.dims, descriptor };
}
