import {
  ChatMessage,
  isAIMessage,
  type BaseMessage,
} from "@langchain/core/messages";

export const GEMMA_TURN_OPEN = "<|turn>";
export const GEMMA_TURN_CLOSE = "<turn|>";
export const GEMMA_TOOL_TAG_OPEN = "<|tool>";
export const GEMMA_TOOL_TAG_CLOSE = "<tool|>";
export const GEMMA_TOOL_CALL_TAG_OPEN = "<|tool_call>";
export const GEMMA_TOOL_CALL_TAG_CLOSE = "<tool_call|>";
export const GEMMA_TOOL_RESPONSE_TAG_OPEN = "<|tool_response>";
export const GEMMA_TOOL_RESPONSE_TAG_CLOSE = "<tool_response|>";
export const GEMMA_STRING_DELIMITER = `<|"|>`;

export interface MediaPipeToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
  };
}

export type MediaPipeToolChoice =
  | "auto"
  | "none"
  | "required"
  | {
      type: "function";
      function: {
        name: string;
      };
    };

export interface PromptRenderInput {
  messages: BaseMessage[];
  tools: MediaPipeToolDefinition[];
  toolChoice?: MediaPipeToolChoice;
}

export interface ParsedCodecToolCall {
  id?: string;
  name?: string;
  arguments?: unknown;
  error?: string;
}

export interface ParsedModelOutput {
  content: string;
  visibleText: string;
  toolCalls: ParsedCodecToolCall[];
}

export type RenderedPromptPart =
  | string
  | { imageSource: string }
  | { audioSource: string };

export type RenderedPrompt = string | RenderedPromptPart[];

export interface MessageContentParts {
  /** All text concatenated (for text-only rendering). */
  text: string;
  /** Ordered parts preserving original interleaving of text and media. */
  orderedParts: RenderedPromptPart[];
  /** Whether any non-text media parts are present. */
  hasMedia: boolean;
}

export function extractVisibleText(rawText: string): string {
  let visibleText = removeTaggedBlocks(
    rawText,
    GEMMA_TOOL_CALL_TAG_OPEN,
    GEMMA_TOOL_CALL_TAG_CLOSE
  );
  visibleText = removeTaggedBlocks(
    visibleText,
    GEMMA_TOOL_RESPONSE_TAG_OPEN,
    GEMMA_TOOL_RESPONSE_TAG_CLOSE
  );

  if (hasTrailingToolBlock(rawText) && /(?:\r?\n)$/.test(visibleText)) {
    visibleText = visibleText.replace(/(?:\r?\n)$/, "");
  }

  return visibleText;
}

export function getMessageTextContent(message: BaseMessage): string {
  if (typeof message.content === "string") {
    return message.content;
  }
  return message.content
    .map((part) => {
      if (typeof part === "string") return part;
      if (
        typeof part === "object" &&
        part !== null &&
        "type" in part &&
        part.type === "text" &&
        "text" in part
      ) {
        return part.text as string;
      }
      return "";
    })
    .filter(Boolean)
    .join("");
}

export function getMessageContentParts(
  message: BaseMessage
): MessageContentParts {
  if (typeof message.content === "string") {
    return {
      text: message.content,
      orderedParts: [message.content],
      hasMedia: false,
    };
  }

  const textParts: string[] = [];
  const orderedParts: RenderedPromptPart[] = [];
  let hasMedia = false;

  for (const part of message.content) {
    if (typeof part === "string") {
      textParts.push(part);
      orderedParts.push(part);
      continue;
    }
    if (typeof part !== "object" || part === null || !("type" in part)) {
      continue;
    }

    if (part.type === "text" && "text" in part) {
      textParts.push(part.text as string);
      orderedParts.push(part.text as string);
    } else if (part.type === "image_url" && "image_url" in part) {
      const imageUrl = part.image_url;
      const url =
        typeof imageUrl === "string"
          ? imageUrl
          : typeof imageUrl === "object" &&
              imageUrl !== null &&
              "url" in imageUrl
            ? (imageUrl.url as string)
            : undefined;
      if (url) {
        orderedParts.push({ imageSource: url });
        hasMedia = true;
      } else {
        console.warn(
          `ChatMediaPipeGenAI: image_url content part has no extractable URL and was ignored.`
        );
      }
    } else if (part.type === "input_audio" && "input_audio" in part) {
      const audio = part.input_audio;
      if (
        typeof audio === "object" &&
        audio !== null &&
        "data" in audio &&
        typeof audio.data === "string"
      ) {
        const format =
          "format" in audio && typeof audio.format === "string"
            ? audio.format
            : "wav";
        orderedParts.push({
          audioSource: `data:audio/${format};base64,${audio.data}`,
        });
        hasMedia = true;
      } else {
        console.warn(
          `ChatMediaPipeGenAI: input_audio content part is malformed (missing "data" string) and was ignored.`
        );
      }
    } else if (part.type !== "text") {
      console.warn(
        `ChatMediaPipeGenAI: Unsupported content part type "${String(part.type)}" was ignored. ` +
          `Supported types: "text", "image_url", "input_audio".`
      );
    }
  }

  return { text: textParts.join(""), orderedParts, hasMedia };
}

export function stripHistoricalThoughts(
  content: string,
  thoughtTagName: string | undefined,
  preserveThoughts: boolean
): string {
  if (preserveThoughts || !thoughtTagName) {
    return content;
  }

  return removeTaggedBlocks(
    content,
    `<${thoughtTagName}>`,
    `</${thoughtTagName}>`
  ).trim();
}

export function getNormalizedRole(
  message: BaseMessage
): "system" | "user" | "model" {
  const messageType = message._getType();
  switch (messageType) {
    case "system":
    case "developer":
      return "system";
    case "human":
      return "user";
    case "ai":
      return "model";
    case "generic": {
      if (!ChatMessage.isInstance(message)) {
        throw new Error("Invalid generic chat message.");
      }
      if (message.role === "assistant") {
        return "model";
      }
      if (message.role === "developer" || message.role === "system") {
        return "system";
      }
      return "user";
    }
    default:
      throw new Error(
        `ChatMediaPipeGenAI does not support "${messageType}" messages in prompt rendering.`
      );
  }
}

export function getReplayToolCalls(message: BaseMessage): Array<{
  id?: string;
  name: string;
  arguments: unknown;
}> {
  if (!isAIMessage(message)) {
    return [];
  }

  if (message.tool_calls?.length) {
    return message.tool_calls.map((toolCall) => ({
      id: toolCall.id,
      name: toolCall.name,
      arguments: toolCall.args,
    }));
  }

  const rawToolCalls = message.additional_kwargs?.tool_calls;
  if (!Array.isArray(rawToolCalls)) {
    return [];
  }

  return rawToolCalls.flatMap((rawToolCall) => {
    if (
      !rawToolCall ||
      typeof rawToolCall !== "object" ||
      !("function" in rawToolCall) ||
      !rawToolCall.function ||
      typeof rawToolCall.function !== "object" ||
      !("name" in rawToolCall.function) ||
      typeof rawToolCall.function.name !== "string"
    ) {
      return [];
    }

    let parsedArguments: unknown = {};
    if (
      "arguments" in rawToolCall.function &&
      typeof rawToolCall.function.arguments === "string"
    ) {
      try {
        parsedArguments = JSON.parse(rawToolCall.function.arguments);
      } catch {
        parsedArguments = rawToolCall.function.arguments;
      }
    }

    return [
      {
        id:
          "id" in rawToolCall && typeof rawToolCall.id === "string"
            ? rawToolCall.id
            : undefined,
        name: rawToolCall.function.name,
        arguments: parsedArguments,
      },
    ];
  });
}

export function serializeGemmaToolDeclaration(
  tool: MediaPipeToolDefinition["function"]
): string {
  return `declaration:${tool.name}${serializeGemmaObject(
    omitUndefined({
      description: tool.description,
      parameters: tool.parameters,
    }),
    { schema: true }
  )}`;
}

export function serializeGemmaToolCall(toolCall: {
  id?: string;
  name: string;
  arguments: unknown;
}): string {
  const body = buildGemmaPayload(toolCall.id, toolCall.arguments, "arguments");
  return `call:${toolCall.name}${serializeGemmaObject(body)}`;
}

export function serializeGemmaToolResponse(response: {
  id?: string;
  name: string;
  content: string;
}): string {
  const parsedContent = tryParseJson(response.content);
  const body = buildGemmaPayload(
    response.id,
    parsedContent ?? response.content,
    "content"
  );
  return `response:${response.name}${serializeGemmaObject(body)}`;
}

export function parseGemmaToolCall(text: string): ParsedCodecToolCall {
  const trimmed = text.trim();

  if (trimmed.startsWith("call:")) {
    try {
      const { name, payload } = parseNamedGemmaBlock(trimmed, "call:");
      const id =
        isPlainObject(payload) && typeof payload.id === "string"
          ? payload.id
          : undefined;
      const argumentsPayload =
        isPlainObject(payload) && "arguments" in payload
          ? payload.arguments
          : isPlainObject(payload)
            ? omitUndefined({
                ...payload,
                id: undefined,
              })
            : payload;

      return {
        id,
        name,
        arguments: argumentsPayload,
      };
    } catch (error) {
      return {
        error:
          error instanceof Error ? error.message : "Invalid tool call block.",
      };
    }
  }

  try {
    const parsed = parseGemmaObject(trimmed);
    if (isPlainObject(parsed)) {
      const id = typeof parsed.id === "string" ? parsed.id : undefined;
      const name = typeof parsed.name === "string" ? parsed.name : undefined;
      const argumentsPayload =
        "arguments" in parsed
          ? parsed.arguments
          : omitUndefined({
              ...parsed,
              id: undefined,
              name: undefined,
            });

      return {
        id,
        name,
        arguments: argumentsPayload,
      };
    }

    return {
      arguments: parsed,
    };
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : "Invalid tool call block.",
    };
  }
}

export function parseGemmaObject(text: string): unknown {
  return JSON.parse(toJsonText(text));
}

export function removeTaggedBlocks(
  text: string,
  openTag: string,
  closeTag: string
): string {
  let cursor = 0;
  let output = "";

  while (cursor < text.length) {
    const start = text.indexOf(openTag, cursor);
    if (start === -1) {
      output += text.slice(cursor);
      break;
    }

    output += text.slice(cursor, start);
    const end = text.indexOf(closeTag, start + openTag.length);
    if (end === -1) {
      // No closing tag — during streaming this means the model is mid-block.
      // Drop the partial block so it doesn't appear in visible text.
      break;
    }
    cursor = end + closeTag.length;
  }

  return output;
}

export function extractTaggedBlocks(
  text: string,
  openTag: string,
  closeTag: string
): string[] {
  const blocks: string[] = [];
  let cursor = 0;

  while (cursor < text.length) {
    const start = text.indexOf(openTag, cursor);
    if (start === -1) {
      break;
    }

    const end = text.indexOf(closeTag, start + openTag.length);
    if (end === -1) {
      break;
    }

    blocks.push(text.slice(start + openTag.length, end).trim());
    cursor = end + closeTag.length;
  }

  return blocks;
}

function parseNamedGemmaBlock(
  text: string,
  prefix: string
): {
  name: string;
  payload: unknown;
} {
  const body = text.slice(prefix.length).trim();
  const objectStart = body.indexOf("{");
  if (objectStart === -1) {
    throw new Error("Gemma tool block is missing an object payload.");
  }

  const name = body.slice(0, objectStart).trim();
  if (!name) {
    throw new Error("Gemma tool block is missing a tool name.");
  }

  return {
    name,
    payload: parseGemmaObject(body.slice(objectStart)),
  };
}

function serializeGemmaObject(
  value: unknown,
  options: {
    schema?: boolean;
  } = {}
): string {
  if (!isPlainObject(value)) {
    return serializeGemmaValue(value, options);
  }

  const entries = Object.entries(value).filter(([, entryValue]) => {
    return entryValue !== undefined;
  });

  return `{${entries
    .map(([key, entryValue]) => {
      return `${key}:${serializeGemmaValue(entryValue, {
        ...options,
        key,
      })}`;
    })
    .join(",")}}`;
}

function serializeGemmaValue(
  value: unknown,
  options: {
    schema?: boolean;
    key?: string;
  } = {}
): string {
  if (value === null) {
    return "null";
  }

  if (Array.isArray(value)) {
    return `[${value
      .map((entry) => serializeGemmaValue(entry, options))
      .join(",")}]`;
  }

  if (isPlainObject(value)) {
    return serializeGemmaObject(value, options);
  }

  if (typeof value === "string") {
    const normalized =
      options.schema && options.key === "type"
        ? normalizeSchemaType(value)
        : value;
    return `${GEMMA_STRING_DELIMITER}${escapeGemmaString(
      normalized
    )}${GEMMA_STRING_DELIMITER}`;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return JSON.stringify(value);
  }

  return `${GEMMA_STRING_DELIMITER}${escapeGemmaString(
    JSON.stringify(value)
  )}${GEMMA_STRING_DELIMITER}`;
}

function normalizeSchemaType(value: string): string {
  switch (value.toLowerCase()) {
    case "string":
      return "STRING";
    case "number":
      return "NUMBER";
    case "integer":
      return "INTEGER";
    case "boolean":
      return "BOOLEAN";
    case "object":
      return "OBJECT";
    case "array":
      return "ARRAY";
    case "null":
      return "NULL";
    default:
      return value.toUpperCase();
  }
}

function escapeGemmaString(value: string): string {
  return value
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t")
    .replace(/<\|"\|>/g, '<|\\"|>')
    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, (ch) =>
      `\\u${ch.charCodeAt(0).toString(16).padStart(4, "0")}`
    );
}

function toJsonText(text: string): string {
  // Split on the Gemma string delimiter to isolate structural vs string regions.
  // Even-indexed segments are structural (keys, braces, commas).
  // Odd-indexed segments are string contents — leave them untouched.
  const segments = text.split(GEMMA_STRING_DELIMITER);
  for (let i = 0; i < segments.length; i += 2) {
    segments[i] = segments[i].replace(
      /([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)/g,
      '$1"$2"$3'
    );
  }
  return segments.join('"');
}

function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

// If data is a plain object without an "id" key, spread its fields alongside
// id at the top level (Gemma's compact form). Otherwise wrap under fallbackKey.
function buildGemmaPayload(
  id: string | undefined,
  data: unknown,
  fallbackKey: string
): Record<string, unknown> {
  if (isPlainObject(data) && !("id" in data)) {
    return omitUndefined({ id, ...data });
  }
  return omitUndefined({ id, [fallbackKey]: data });
}

function omitUndefined<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, entryValue]) => entryValue !== undefined)
  ) as T;
}

function isPlainObject(
  value: unknown
): value is Record<string, unknown> {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.getPrototypeOf(value) === Object.prototype
  );
}

function hasTrailingToolBlock(text: string): boolean {
  const trimmed = text.trimEnd();
  const toolCallOpen = trimmed.lastIndexOf(GEMMA_TOOL_CALL_TAG_OPEN);
  const toolCallClose = trimmed.lastIndexOf(GEMMA_TOOL_CALL_TAG_CLOSE);
  const toolResponseOpen = trimmed.lastIndexOf(GEMMA_TOOL_RESPONSE_TAG_OPEN);
  const toolResponseClose = trimmed.lastIndexOf(GEMMA_TOOL_RESPONSE_TAG_CLOSE);

  return (
    (toolCallOpen !== -1 &&
      (toolCallOpen > toolCallClose ||
        trimmed.endsWith(GEMMA_TOOL_CALL_TAG_CLOSE))) ||
    (toolResponseOpen !== -1 &&
      (toolResponseOpen > toolResponseClose ||
        trimmed.endsWith(GEMMA_TOOL_RESPONSE_TAG_CLOSE)))
  );
}
