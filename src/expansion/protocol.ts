/**
 * protocol.ts — MLX Query Expansion Protocol (Prompt Builder & Response Validation)
 *
 * Implements strict prompt construction and response parsing for fast-path MLX
 * query expansion on Apple Silicon.
 *
 * Requirements:
 * 1. Clear prompt asking ONLY for concise typed lines (lex:, vec:, hyde:)
 * 2. Delimits input query as raw data (prevents prompt injection / instruction hallucination)
 * 3. Topic & query-term relevance verification
 * 4. Lexical filtering (includeLexical)
 * 5. Rejection of prose, markdown bullet points, preamble, placeholder echo, and empty completions
 */

export type QueryType = 'lex' | 'vec' | 'hyde';

export interface Queryable {
  type: QueryType;
  text: string;
}

export interface ParsedExpansion {
  queryables: Queryable[];
  lex: string[];
  vec: string[];
  hyde: string[];
  invalidLines: string[];
  usable: boolean;
  qualityMessage: string;
  rawText: string;
}

export interface ExpansionPromptOptions {
  intent?: string;
}

export interface ExpansionParseOptions {
  query?: string;
  includeLexical?: boolean;
}

export const MAX_EXPANSION_BYTES = 65_536;
export const MAX_EXPANSION_LINES = 32;
export const MAX_EXPANSION_ITEMS = 12;
export const MAX_EXPANSION_LINE_BYTES = 4096;

/**
 * Builds a strict, structured query expansion prompt for generic / chat instruction models.
 * Delimits the search query within <query>...</query> tags to treat user input purely as data.
 */
export function buildMlxExpansionPrompt(query: string, options?: ExpansionPromptOptions): string {
  const cleanQuery = query.replace(/<\/?query>/gi, "").trim();
  const cleanIntent = options?.intent ? options.intent.replace(/<\/?intent>/gi, "").trim() : undefined;

  let prompt = `/no_think
Generate search query expansions for the query delimited below.
Output each expansion on a new line using one of these three prefixes:
lex: add specific related technical terms or synonyms, not a copy of the query
vec: a natural search reformulation using concrete related concepts
hyde: a short factual-looking hypothetical answer passage, never a question

Example for the query "SQL joins":
lex: SQL joins inner outer left foreign key
vec: combining rows from relational tables using matching columns
hyde: SQL joins combine table rows on matching keys. Inner joins keep matches, while outer joins retain unmatched rows.

Write expansions for the actual query below, NOT the example. Do not write "semantic reformulation" or other task labels in the content. Output ONLY three prefixed lines. No preamble, no explanations, no code fences.

<query>
${cleanQuery}
</query>`;

  if (cleanIntent) {
    prompt += `\n<intent>\n${cleanIntent}\n</intent>`;
  }

  return prompt;
}

/**
 * Parses and validates raw LLM completion text into typed Queryable items (lex/vec/hyde).
 * Rejects unstructured prose, conversational fluff, unstripped thinking tags, and irrelevant lines.
 */
export function parseExpansionOutput(
  text: string,
  options?: ExpansionParseOptions,
): ParsedExpansion {
  const includeLexical = options?.includeLexical ?? true;
  const query = options?.query ?? "";
  const input = text ?? "";
  const oversized = input.length > MAX_EXPANSION_BYTES || Buffer.byteLength(input, "utf8") > MAX_EXPANSION_BYTES;
  // Keep only a small diagnostic prefix for rejected oversized completions.
  const rawText = oversized ? input.slice(0, 4096) : input;

  const result: ParsedExpansion = {
    queryables: [],
    lex: [],
    vec: [],
    hyde: [],
    invalidLines: [],
    usable: false,
    qualityMessage: "",
    rawText,
  };

  if (oversized) {
    result.qualityMessage = "Completion exceeds byte limit";
    return result;
  }

  const lines = rawText.split("\n");
  if (lines.length > MAX_EXPANSION_LINES || lines.some((line) => Buffer.byteLength(line, "utf8") > MAX_EXPANSION_LINE_BYTES)) {
    result.qualityMessage = "Completion exceeds line count or line byte limit";
    return result;
  }

  if (!rawText.trim()) {
    result.qualityMessage = "Completion text is empty";
    return result;
  }

  if (rawText.includes("<think>") || rawText.includes("</think>")) {
    result.qualityMessage = "Completion contains unstripped thinking tags";
    return result;
  }

  const queryLower = query.normalize("NFKC").toLowerCase().trim();
  const queryTerms = queryLower.match(/[\p{L}\p{N}\p{M}]+/gu) ?? [];

  const hasQueryRelevance = (content: string): boolean => {
    if (!queryLower) return true;
    const lower = content.normalize("NFKC").toLowerCase();
    // Non-Latin and punctuation-only queries must never disable relevance checks.
    return lower.includes(queryLower) || queryTerms.some((term) => lower.includes(term));
  };

  const seen = new Set<string>();

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    // Reject markdown code block fence markers
    if (line.startsWith("```")) {
      result.invalidLines.push(line);
      continue;
    }

    // Match typed line: optional leading bullet/number, then lex/vec/hyde: <content>
    const match = line.match(/^[-*•\d.\s]*(lex|vec|hyde)\s*:\s*(.+)$/i);
    if (!match) {
      result.invalidLines.push(line);
      continue;
    }

    const typeStr = match[1]!.toLowerCase() as QueryType;
    let content = match[2]!.trim();

    // Strip wrapping quotes or markdown bolding around content
    content = content.replace(/^["'`*]+|["'`*]+$/g, "").trim();

    if (!content) {
      result.invalidLines.push(line);
      continue;
    }

    const contentLower = content.toLowerCase();

    // Reject echo of prompt delimiter tags or instruction placeholders
    if (
      contentLower === "<query>" ||
      contentLower === "</query>" ||
      contentLower === "<intent>" ||
      contentLower === "</intent>" ||
      contentLower.startsWith("/no_think") ||
      contentLower === "(keyword or synonym phrase)" ||
      contentLower === "(conceptual or semantic reformulation)" ||
      contentLower === "(hypothetical sentence answering the query)"
    ) {
      result.invalidLines.push(line);
      continue;
    }

    // Topic / query-term overlap check
    if (query && !hasQueryRelevance(content)) {
      result.invalidLines.push(line);
      continue;
    }

    // Deduplicate
    const key = `${typeStr}:${content}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);

    if (seen.size > MAX_EXPANSION_ITEMS) {
      result.queryables = [];
      result.lex = [];
      result.vec = [];
      result.hyde = [];
      result.qualityMessage = "Completion exceeds expansion item limit";
      return result;
    }

    // Record accepted typed items
    if (typeStr === "lex") {
      result.lex.push(content);
    } else if (typeStr === "vec") {
      result.vec.push(content);
    } else if (typeStr === "hyde") {
      result.hyde.push(content);
    }

    if (typeStr !== "lex" || includeLexical) {
      result.queryables.push({ type: typeStr, text: content });
    }
  }

  if (result.queryables.length === 0) {
    result.usable = false;
    result.qualityMessage =
      result.invalidLines.length > 0
        ? `No valid typed lines parsed; ${result.invalidLines.length} invalid lines rejected`
        : "No expansion lines generated";
    return result;
  }

  result.usable = true;
  result.qualityMessage = "Valid usable query expansion";
  return result;
}
