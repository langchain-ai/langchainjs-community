import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type BaseChatModelParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
} from "@langchain/core/messages";
import { type ToolCallChunk } from "@langchain/core/messages/tool";
import {
  ChatGenerationChunk,
  type ChatResult,
  type ChatGeneration,
} from "@langchain/core/outputs";
import {
  makeInvalidToolCall,
  parseToolCall,
} from "@langchain/core/output_parsers/openai_tools";
import type { Runnable } from "@langchain/core/runnables";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";

import { Gemma4Codec } from "./mediapipe_genai_internal/gemma4_codec.js";
import { extractVisibleText } from "./mediapipe_genai_internal/prompt_codec.js";
import type {
  MediaPipeToolChoice,
  MediaPipeToolDefinition,
  ParsedCodecToolCall,
  RenderedPrompt,
} from "./mediapipe_genai_internal/prompt_codec.js";

interface MediaPipeFilesetResolverLike {
  forGenAiTasks(wasmRoot: string): Promise<unknown>;
}

interface MediaPipeLlmInferenceLike {
  generateResponse(
    input: RenderedPrompt,
    callback?: (partialResult: string, done: boolean) => void
  ): Promise<string | void>;
}

interface MediaPipeLlmInferenceStaticLike {
  createFromOptions(
    fileset: unknown,
    options: {
      baseOptions: {
        modelAssetPath: string;
      };
      maxTokens?: number;
      temperature?: number;
      topK?: number;
      randomSeed?: number;
      maxNumImages?: number;
      supportAudio?: boolean;
    }
  ): Promise<MediaPipeLlmInferenceLike>;
}

interface MediaPipeTasksGenAIModuleLike {
  FilesetResolver: MediaPipeFilesetResolverLike;
  LlmInference: MediaPipeLlmInferenceStaticLike;
}

export interface ChatMediaPipeGenAIInput extends BaseChatModelParams {
  wasmRoot: string;
  modelAssetPath: string;
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  randomSeed?: number;
  thoughtTagName?: string;
  /** When set > 0, enables vision modality. Required for image content parts. */
  maxNumImages?: number;
  /** When true, enables audio modality. Required for audio content parts. */
  supportAudio?: boolean;
}

export interface ChatMediaPipeGenAICallOptions extends BaseChatModelCallOptions {}

export type InitializationProgress = {
  stage: "resolving-fileset" | "creating-inference" | "ready";
};

interface ParsedResponseArtifacts {
  visibleText: string;
  message: AIMessage;
  toolCallChunks: ToolCallChunk[];
  invalidToolCalls: AIMessage["invalid_tool_calls"];
}

function isFunctionToolDefinition(
  tool: unknown
): tool is MediaPipeToolDefinition {
  return (
    !!tool &&
    typeof tool === "object" &&
    "type" in tool &&
    tool.type === "function" &&
    "function" in tool &&
    !!tool.function &&
    typeof tool.function === "object" &&
    "name" in tool.function &&
    typeof tool.function.name === "string"
  );
}

function normalizeToolDefinitions(
  tools: BindToolsInput[] | undefined
): MediaPipeToolDefinition[] {
  return (tools ?? []).map((tool) => {
    const openAITool = isFunctionToolDefinition(tool)
      ? tool
      : convertToOpenAITool(tool);

    return {
      type: "function",
      function: {
        name: openAITool.function.name,
        description: openAITool.function.description,
        parameters: openAITool.function.parameters,
      },
    };
  });
}

function normalizeToolChoice(
  toolChoice: BaseChatModelCallOptions["tool_choice"] | undefined
): MediaPipeToolChoice | undefined {
  if (toolChoice === undefined) {
    return undefined;
  }

  if (toolChoice === "auto" || toolChoice === "none") {
    return toolChoice;
  }

  if (toolChoice === "any" || toolChoice === "required") {
    return "required";
  }

  if (typeof toolChoice === "string") {
    return {
      type: "function",
      function: {
        name: toolChoice,
      },
    };
  }

  if (
    typeof toolChoice === "object" &&
    toolChoice !== null &&
    "type" in toolChoice
  ) {
    if (toolChoice.type === "none" || toolChoice.type === "auto") {
      return toolChoice.type;
    }
    if (toolChoice.type === "required" || toolChoice.type === "any") {
      return "required";
    }

    let name: string | undefined;
    if (
      toolChoice.type === "function" &&
      "function" in toolChoice &&
      toolChoice.function &&
      typeof toolChoice.function === "object" &&
      "name" in toolChoice.function &&
      typeof toolChoice.function.name === "string"
    ) {
      name = toolChoice.function.name;
    } else if (
      toolChoice.type === "tool" &&
      "name" in toolChoice &&
      typeof toolChoice.name === "string"
    ) {
      name = toolChoice.name;
    }

    if (name) {
      return { type: "function", function: { name } };
    }
  }

  throw new Error(
    `Unsupported tool_choice value for ChatMediaPipeGenAI: ${JSON.stringify(
      toolChoice
    )}`
  );
}

/**
 * Browser-only chat wrapper around MediaPipe Tasks GenAI Gemma checkpoints.
 *
 * To use this model you need to have the `@mediapipe/tasks-genai` module
 * installed. This can be installed using
 * `npm install -S @mediapipe/tasks-genai`.
 *
 * Call `initialize()` before `invoke()`, `stream()`, or `generate()`.
 * `wasmRoot` should point to the directory that serves the MediaPipe WASM
 * assets. `modelAssetPath` should point to the browser-accessible `.task`
 * model file.
 *
 * Tool calling is formatted using Gemma 4 style tool tags, so this wrapper is
 * intended for tool-capable Gemma-family checkpoints.
 *
 * Multimodal inputs (images and audio) are supported when the model checkpoint
 * supports them (e.g. Gemma 3n, Gemma 4 edge models). Enable image input by
 * setting `maxNumImages` and audio input by setting `supportAudio: true`.
 * Pass `image_url` or `input_audio` content parts in `HumanMessage` arrays.
 *
 * @example
 * ```typescript
 * import { HumanMessage } from "@langchain/core/messages";
 * import { ChatMediaPipeGenAI } from "@langchain/community/chat_models/mediapipe_genai";
 *
 * const model = new ChatMediaPipeGenAI({
 *   wasmRoot: "/vendor/mediapipe/tasks-genai/wasm",
 *   modelAssetPath: "/models/gemma/gemma-4-E2B-it-web.task",
 *   maxTokens: 2048,
 *   topK: 40,
 *   temperature: 0.2,
 *   randomSeed: 101,
 * });
 *
 * await model.initialize();
 *
 * const response = await model.invoke([
 *   new HumanMessage("Write one sentence about Berlin."),
 * ]);
 * ```
 *
 * @example
 * ```typescript
 * import {
 *   HumanMessage,
 *   ToolMessage,
 * } from "@langchain/core/messages";
 * import { ChatMediaPipeGenAI } from "@langchain/community/chat_models/mediapipe_genai";
 *
 * const getWeather = {
 *   type: "function",
 *   function: {
 *     name: "get_current_weather",
 *     description: "Get the current weather for a location.",
 *     parameters: {
 *       type: "object",
 *       properties: {
 *         location: { type: "string" },
 *       },
 *       required: ["location"],
 *     },
 *   },
 * } as const;
 *
 * const model = new ChatMediaPipeGenAI({
 *   wasmRoot: "/vendor/mediapipe/tasks-genai/wasm",
 *   modelAssetPath: "/models/gemma/gemma-4-E2B-it-web.task",
 *   maxTokens: 2048,
 *   topK: 40,
 *   temperature: 0.2,
 *   randomSeed: 101,
 * });
 *
 * await model.initialize();
 *
 * const bound = model.bindTools([getWeather]);
 * const first = await bound.invoke([
 *   new HumanMessage("What is the weather in Berlin?"),
 * ]);
 *
 * if (first.tool_calls?.length) {
 *   const toolCall = first.tool_calls[0];
 *   const toolResult = JSON.stringify({
 *     location: toolCall.args.location,
 *     temperatureC: 18,
 *   });
 *
 *   const final = await bound.invoke([
 *     new HumanMessage("What is the weather in Berlin?"),
 *     first,
 *     new ToolMessage({
 *       tool_call_id: toolCall.id ?? "call_weather_1",
 *       content: toolResult,
 *     }),
 *   ]);
 * }
 * ```
 *
 * @example
 * ```typescript
 * import { HumanMessage } from "@langchain/core/messages";
 * import { ChatMediaPipeGenAI } from "@langchain/community/chat_models/mediapipe_genai";
 *
 * const model = new ChatMediaPipeGenAI({
 *   wasmRoot: "/vendor/mediapipe/tasks-genai/wasm",
 *   modelAssetPath: "/models/gemma/gemma-4-E2B-it-web.task",
 *   maxNumImages: 1,
 * });
 *
 * await model.initialize();
 *
 * const response = await model.invoke([
 *   new HumanMessage({
 *     content: [
 *       { type: "text", text: "Describe this image." },
 *       { type: "image_url", image_url: { url: "/photos/cat.jpg" } },
 *     ],
 *   }),
 * ]);
 * ```
 */
export class ChatMediaPipeGenAI extends BaseChatModel<ChatMediaPipeGenAICallOptions> {
  static lc_name() {
    return "ChatMediaPipeGenAI";
  }

  protected llmInference?: MediaPipeLlmInferenceLike;

  protected readonly codec: Gemma4Codec;

  wasmRoot: string;

  modelAssetPath: string;

  maxTokens?: number;

  temperature?: number;

  topK?: number;

  randomSeed?: number;

  thoughtTagName?: string;

  maxNumImages?: number;

  supportAudio?: boolean;

  get callKeys() {
    return [...super.callKeys, "tools", "tool_choice"];
  }

  constructor(fields: ChatMediaPipeGenAIInput) {
    super(fields);

    this.wasmRoot = fields.wasmRoot;
    this.modelAssetPath = fields.modelAssetPath;
    this.maxTokens = fields.maxTokens;
    this.temperature = fields.temperature;
    this.topK = fields.topK;
    this.randomSeed = fields.randomSeed;
    this.thoughtTagName = fields.thoughtTagName;
    this.maxNumImages = fields.maxNumImages;
    this.supportAudio = fields.supportAudio;
    this.codec = new Gemma4Codec({
      thoughtTagName: fields.thoughtTagName,
    });
  }

  _llmType(): string {
    return "mediapipe-genai";
  }

  identifyingParams() {
    return {
      wasmRoot: this.wasmRoot,
      modelAssetPath: this.modelAssetPath,
      maxTokens: this.maxTokens,
      temperature: this.temperature,
      topK: this.topK,
      randomSeed: this.randomSeed,
      thoughtTagName: this.thoughtTagName,
      maxNumImages: this.maxNumImages,
      supportAudio: this.supportAudio,
    };
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<ChatMediaPipeGenAICallOptions>
  ): Runnable<
    BaseLanguageModelInput,
    AIMessageChunk,
    ChatMediaPipeGenAICallOptions
  > {
    return this.withConfig({
      tools,
      ...kwargs,
    });
  }

  async initialize(
    progressCallback?: (progress: InitializationProgress) => void
  ): Promise<void> {
    if (this.llmInference) {
      progressCallback?.({ stage: "ready" });
      return;
    }

    this.ensureSupportedEnvironment();

    progressCallback?.({ stage: "resolving-fileset" });
    const tasksModule = await this.importMediaPipeTasksGenAI();
    const fileset = await tasksModule.FilesetResolver.forGenAiTasks(
      this.wasmRoot
    );

    progressCallback?.({ stage: "creating-inference" });
    this.llmInference = await tasksModule.LlmInference.createFromOptions(
      fileset,
      {
        baseOptions: {
          modelAssetPath: this.modelAssetPath,
        },
        maxTokens: this.maxTokens,
        temperature: this.temperature,
        topK: this.topK,
        randomSeed: this.randomSeed,
        maxNumImages: this.maxNumImages,
        supportAudio: this.supportAudio,
      }
    );

    progressCallback?.({ stage: "ready" });
  }

  protected ensureSupportedEnvironment(): void {
    if (
      typeof window === "undefined" ||
      typeof navigator === "undefined" ||
      typeof document === "undefined"
    ) {
      throw new Error(
        "ChatMediaPipeGenAI is browser-only. Call initialize() from a WebGPU-capable browser environment."
      );
    }
  }

  protected async importMediaPipeTasksGenAI(): Promise<MediaPipeTasksGenAIModuleLike> {
    return (await import("@mediapipe/tasks-genai")) as MediaPipeTasksGenAIModuleLike;
  }

  protected getInference(): MediaPipeLlmInferenceLike {
    if (!this.llmInference) {
      throw new Error(
        "ChatMediaPipeGenAI must be initialized before invocation. Call initialize() first."
      );
    }
    return this.llmInference;
  }

  async _generate(
    messages: BaseMessage[],
    options?: this["ParsedCallOptions"]
  ): Promise<ChatResult> {
    this.validateCallOptions(options);
    const inference = this.getInference();
    const { prompt, tools } = this.renderPrompt(messages, options);

    const rawResponse = await inference.generateResponse(prompt);
    if (typeof rawResponse !== "string") {
      throw new Error(
        "MediaPipe LlmInference.generateResponse() did not return a string for non-streaming generation."
      );
    }

    const parsed = this.parseResponseArtifacts(rawResponse, tools);
    const generation: ChatGeneration = {
      text:
        typeof parsed.message.content === "string"
          ? parsed.message.content
          : "",
      message: parsed.message,
    };

    return {
      generations: [generation],
      llmOutput: {
        rawResponse,
      },
    };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.validateCallOptions(options);
    const inference = this.getInference();
    const { prompt, tools } = this.renderPrompt(messages, options);

    let rawResponse = "";
    let emittedVisibleText = "";
    let done = false;
    let rejectedError: Error | undefined;
    let wakeUp: (() => void) | undefined;
    const pendingText: string[] = [];

    const notify = () => {
      if (wakeUp) {
        const resolve = wakeUp;
        wakeUp = undefined;
        resolve();
      }
    };

    const streamPromise = inference
      .generateResponse(prompt, (partialResult, streamDone) => {
        rawResponse += partialResult;
        const visibleText = extractVisibleText(rawResponse);
        const delta = visibleText.slice(emittedVisibleText.length);
        if (delta) {
          emittedVisibleText = visibleText;
          pendingText.push(delta);
          notify();
        }
        if (streamDone) {
          done = true;
          notify();
        }
      })
      .then((maybeFullResponse) => {
        if (!rawResponse && typeof maybeFullResponse === "string") {
          rawResponse = maybeFullResponse;
        }
        done = true;
        notify();
      })
      .catch((error) => {
        rejectedError =
          error instanceof Error ? error : new Error(String(error));
        done = true;
        notify();
      });

    while (!done || pendingText.length > 0) {
      if (pendingText.length === 0) {
        await new Promise<void>((resolve) => {
          wakeUp = resolve;
        });
      }

      while (pendingText.length > 0) {
        const text = pendingText.shift();
        if (!text) {
          continue;
        }

        yield new ChatGenerationChunk({
          text,
          message: new AIMessageChunk({
            content: text,
          }),
        });
        await runManager?.handleLLMNewToken(text);
      }
    }

    await streamPromise;
    if (rejectedError) {
      throw rejectedError;
    }

    const parsed = this.parseResponseArtifacts(rawResponse, tools);
    const finalVisibleDelta = parsed.visibleText.slice(
      emittedVisibleText.length
    );

    if (finalVisibleDelta) {
      yield new ChatGenerationChunk({
        text: finalVisibleDelta,
        message: new AIMessageChunk({
          content: finalVisibleDelta,
        }),
      });
      await runManager?.handleLLMNewToken(finalVisibleDelta);
    }

    if (
      parsed.toolCallChunks.length > 0 ||
      (parsed.invalidToolCalls?.length ?? 0) > 0
    ) {
      yield new ChatGenerationChunk({
        text: "",
        message: new AIMessageChunk({
          content: "",
          tool_call_chunks: parsed.toolCallChunks,
          invalid_tool_calls: parsed.invalidToolCalls,
          additional_kwargs: parsed.message.additional_kwargs,
        }),
      });
    }
  }

  protected parseResponseArtifacts(
    rawResponse: string,
    tools: MediaPipeToolDefinition[]
  ): ParsedResponseArtifacts {
    const parsedOutput = this.codec.parse(rawResponse);
    const toolNames = new Set(tools.map((tool) => tool.function.name));
    const parsedToolCalls = parsedOutput.toolCalls.map((toolCall, index) =>
      this.parseToolCallArtifact(toolCall, toolNames, index)
    );

    const toolCalls = parsedToolCalls.flatMap((toolCall) =>
      toolCall.parsed ? [toolCall.parsed] : []
    );
    const rawToolCalls = parsedToolCalls.flatMap((toolCall) =>
      toolCall.raw ? [toolCall.raw] : []
    );
    const toolCallChunks = parsedToolCalls.flatMap((toolCall) =>
      toolCall.chunk ? [toolCall.chunk] : []
    );
    const invalidToolCalls = parsedToolCalls.flatMap((toolCall) =>
      toolCall.invalid ? [toolCall.invalid] : []
    );

    return {
      visibleText: parsedOutput.visibleText,
      toolCallChunks,
      invalidToolCalls,
      message: new AIMessage({
        content: parsedOutput.content,
        tool_calls: toolCalls,
        invalid_tool_calls: invalidToolCalls,
        additional_kwargs:
          rawToolCalls.length > 0
            ? {
                tool_calls: rawToolCalls,
              }
            : {},
      }),
    };
  }

  protected renderPrompt(
    messages: BaseMessage[],
    options?: this["ParsedCallOptions"]
  ): {
    prompt: RenderedPrompt;
    tools: MediaPipeToolDefinition[];
  } {
    const tools = normalizeToolDefinitions(options?.tools);

    return {
      prompt: this.codec.render({
        messages,
        tools,
        toolChoice: normalizeToolChoice(options?.tool_choice),
      }),
      tools,
    };
  }

  protected validateCallOptions(options?: this["ParsedCallOptions"]): void {
    if (options?.stop?.length) {
      throw new Error("ChatMediaPipeGenAI does not support stop sequences.");
    }

    if (options?.signal) {
      throw new Error(
        "ChatMediaPipeGenAI does not support AbortSignal cancellation."
      );
    }
  }

  protected parseToolCallArtifact(
    parsedToolCall: ParsedCodecToolCall,
    toolNames: Set<string>,
    index: number
  ): {
    parsed?: ReturnType<typeof parseToolCall>;
    raw?: {
      id: string;
      type: "function";
      function: {
        name: string;
        arguments: string;
      };
    };
    chunk?: ToolCallChunk;
    invalid?: ReturnType<typeof makeInvalidToolCall>;
  } {
    const toolCallId = parsedToolCall.id ?? `call_mediapipe_${index + 1}`;
    const name = parsedToolCall.name;
    const serializedArguments =
      typeof parsedToolCall.arguments === "string"
        ? parsedToolCall.arguments
        : JSON.stringify(parsedToolCall.arguments ?? {});

    const fallbackRawToolCall = {
      id: toolCallId,
      type: "function" as const,
      function: {
        name: name ?? "",
        arguments: serializedArguments,
      },
    };

    if (parsedToolCall.error) {
      return {
        invalid: makeInvalidToolCall(fallbackRawToolCall, parsedToolCall.error),
      };
    }

    if (!name) {
      return {
        invalid: makeInvalidToolCall(
          fallbackRawToolCall,
          "Tool call is missing a tool name."
        ),
      };
    }

    if (!toolNames.has(name)) {
      return {
        invalid: makeInvalidToolCall(
          fallbackRawToolCall,
          `Tool "${name}" is not bound to this ChatMediaPipeGenAI instance.`
        ),
      };
    }

    try {
      return {
        parsed: parseToolCall(fallbackRawToolCall, { returnId: true }),
        raw: fallbackRawToolCall,
        chunk: {
          id: toolCallId,
          index,
          name,
          args: serializedArguments,
        },
      };
    } catch (error) {
      return {
        invalid: makeInvalidToolCall(
          fallbackRawToolCall,
          error instanceof Error
            ? error.message
            : "Failed to parse tool call arguments."
        ),
      };
    }
  }
}
