import { describe, expect, test } from "vitest";
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { tool } from "@langchain/core/tools";
import {
  AgentExecutor,
  createToolCallingAgent,
} from "@langchain/classic/agents";
import * as z from "zod";

import {
  ChatMediaPipeGenAI,
  type ChatMediaPipeGenAIInput,
} from "../mediapipe_genai.js";
import { Gemma4Codec } from "../mediapipe_genai_internal/gemma4_codec.js";
import type { RenderedPrompt } from "../mediapipe_genai_internal/prompt_codec.js";

const weatherTool = {
  type: "function" as const,
  function: {
    name: "get_current_weather",
    description: "Get weather for a given location.",
    parameters: {
      type: "object",
      properties: {
        location: {
          type: "string",
        },
      },
      required: ["location"],
    },
  },
};

class FakeInference {
  prompts: RenderedPrompt[] = [];

  private readonly responses: Array<
    | {
        type: "full";
        value: string;
      }
    | {
        type: "stream";
        chunks: string[];
        final?: string;
      }
  >;

  constructor(
    responses: Array<
      | {
          type: "full";
          value: string;
        }
      | {
          type: "stream";
          chunks: string[];
          final?: string;
        }
    >
  ) {
    this.responses = [...responses];
  }

  async generateResponse(
    input: RenderedPrompt,
    callback?: (partialResult: string, done: boolean) => void
  ): Promise<string | void> {
    this.prompts.push(input);
    const response = this.responses.shift();
    if (!response) {
      throw new Error("No fake response configured.");
    }

    if (response.type === "full") {
      return response.value;
    }

    for (const chunk of response.chunks) {
      callback?.(chunk, false);
    }
    callback?.("", true);
    return response.final;
  }
}

class TestChatMediaPipeGenAI extends ChatMediaPipeGenAI {
  importCalls = 0;

  createInferenceCalls = 0;

  private readonly fakeInference: FakeInference;

  constructor(fields: ChatMediaPipeGenAIInput, fakeInference: FakeInference) {
    super(fields);
    this.fakeInference = fakeInference;
  }

  protected override ensureSupportedEnvironment(): void {}

  protected override async importMediaPipeTasksGenAI() {
    this.importCalls += 1;
    return {
      FilesetResolver: {
        forGenAiTasks: async () => ({ wasm: true }),
      },
      LlmInference: {
        createFromOptions: async () => {
          this.createInferenceCalls += 1;
          return this.fakeInference;
        },
      },
    };
  }
}

function createModel(fakeInference: FakeInference) {
  return new TestChatMediaPipeGenAI(
    {
      wasmRoot: "/wasm",
      modelAssetPath: "/models/gemma.task",
      temperature: 0.2,
      topK: 40,
      randomSeed: 7,
      thoughtTagName: "think",
    },
    fakeInference
  );
}

const getCurrentWeather = tool((input) => `It is 20C in ${input.location}.`, {
  name: "get_current_weather",
  description: "Get weather for a given location.",
  schema: z.object({
    location: z.string().describe("The location to get the weather for."),
  }),
});

describe("Gemma4Codec", () => {
  test("renders plain chat prompt with a final assistant turn", () => {
    const codec = new Gemma4Codec();
    const prompt = codec.render({
      tools: [],
      messages: [
        new SystemMessage({ content: "You are concise." }),
        new HumanMessage({ content: "Hello there." }),
      ],
    });

    expect(prompt).toContain("<|turn>system");
    expect(prompt).toContain("You are concise.");
    expect(prompt).toContain("<|turn>user");
    expect(prompt).toContain("Hello there.");
    expect(prompt.trimEnd().endsWith("<|turn>model")).toBe(true);
  });

  test("renders tool declarations and tool replay tags", () => {
    const codec = new Gemma4Codec({
      thoughtTagName: "think",
    });
    const prompt = codec.render({
      tools: [weatherTool],
      toolChoice: "required",
      messages: [
        new AIMessage({
          content: "<think>internal</think>Need weather.",
          tool_calls: [
            {
              id: "call_weather_1",
              type: "tool_call",
              name: "get_current_weather",
              args: {
                location: "Berlin",
              },
            },
          ],
        }),
        new ToolMessage({
          content: '{"temperature":20}',
          tool_call_id: "call_weather_1",
        }),
      ],
    });

    expect(prompt).toContain("<|tool>declaration:get_current_weather{");
    expect(prompt).toContain("description:<|\"|>Get weather for a given location.<|\"|>");
    expect(prompt).toContain(
      "<|tool_call>call:get_current_weather{id:<|\"|>call_weather_1<|\"|>,location:<|\"|>Berlin<|\"|>}<tool_call|>"
    );
    expect(prompt).toContain(
      "<|tool_response>response:get_current_weather{id:<|\"|>call_weather_1<|\"|>,temperature:20}<tool_response|>"
    );
    expect(prompt).toContain("<think>internal</think>");
  });

  test("parses single and multiple tool call blocks", () => {
    const codec = new Gemma4Codec();
    const parsed = codec.parse(
      [
        "Let me check.",
        "<|tool_call>",
        "call:weather{id:<|\"|>call_1<|\"|>,location:<|\"|>Berlin<|\"|>}",
        "<tool_call|>",
        "<|tool_call>",
        "call:population{location:<|\"|>Berlin<|\"|>}",
        "<tool_call|>",
      ].join("\n")
    );

    expect(parsed.content).toBe("Let me check.");
    expect(parsed.toolCalls).toHaveLength(2);
    expect(parsed.toolCalls[0]).toMatchObject({
      id: "call_1",
      name: "weather",
      arguments: { location: "Berlin" },
    });
    expect(parsed.toolCalls[1]).toMatchObject({
      name: "population",
      arguments: { location: "Berlin" },
    });
  });

  test("strips old thoughts on normal turns but preserves them in tool turns", () => {
    const codec = new Gemma4Codec({
      thoughtTagName: "think",
    });
    const prompt = codec.render({
      tools: [weatherTool],
      messages: [
        new AIMessage({
          content: "<think>drop me</think>Visible answer.",
        }),
        new AIMessage({
          content: "<think>keep me</think>Need tool.",
          tool_calls: [
            {
              id: "call_weather_1",
              type: "tool_call",
              name: "get_current_weather",
              args: { location: "Paris" },
            },
          ],
        }),
      ],
    });

    expect(prompt).not.toContain("<think>drop me</think>");
    expect(prompt).toContain("Visible answer.");
    expect(prompt).toContain("<think>keep me</think>");
  });

  test("renders multimodal prompt with interleaved text and media", () => {
    const codec = new Gemma4Codec();
    const prompt = codec.render({
      tools: [],
      messages: [
        new HumanMessage({
          content: [
            { type: "text", text: "Describe " },
            {
              type: "image_url",
              image_url: { url: "https://example.com/photo.jpg" },
            },
            { type: "text", text: " and transcribe " },
            {
              type: "input_audio",
              input_audio: { data: "AAAA", format: "wav" },
            },
          ],
        }),
      ],
    });

    expect(Array.isArray(prompt)).toBe(true);
    const parts = prompt as Array<unknown>;

    // Verify interleaving order: text, image, text, audio, text (close tag + model turn)
    expect(typeof parts[0]).toBe("string");
    expect(parts[0]).toContain("<|turn>user\nDescribe ");
    expect(parts[1]).toEqual({
      imageSource: "https://example.com/photo.jpg",
    });
    expect(typeof parts[2]).toBe("string");
    expect(parts[2]).toContain(" and transcribe ");
    expect(parts[3]).toEqual({
      audioSource: "data:audio/wav;base64,AAAA",
    });
    // Last part has close tag and final model turn
    const lastPart = parts[parts.length - 1];
    expect(typeof lastPart).toBe("string");
    expect(lastPart).toContain("<turn|>");
    expect(lastPart).toContain("<|turn>model");
  });

  test("renders text-only messages as a plain string even with array content", () => {
    const codec = new Gemma4Codec();
    const prompt = codec.render({
      tools: [],
      messages: [
        new HumanMessage({
          content: [{ type: "text", text: "Just text." }],
        }),
      ],
    });

    expect(typeof prompt).toBe("string");
    expect(prompt).toContain("Just text.");
  });
});

describe("ChatMediaPipeGenAI", () => {
  test("fails fast outside the browser before importing MediaPipe", async () => {
    const model = new ChatMediaPipeGenAI({
      wasmRoot: "/wasm",
      modelAssetPath: "/models/gemma.task",
    });

    await expect(model.initialize()).rejects.toThrow(
      "ChatMediaPipeGenAI is browser-only"
    );
  });

  test("initialize and invoke return plain assistant text", async () => {
    const fakeInference = new FakeInference([
      {
        type: "full",
        value: "Hello from Gemma.",
      },
    ]);
    const model = createModel(fakeInference);

    await model.initialize();
    const response = await model.invoke("Say hello.");

    expect(fakeInference.prompts).toHaveLength(1);
    expect(fakeInference.prompts[0]).toContain("Say hello.");
    expect(response.content).toBe("Hello from Gemma.");
    expect(response.tool_calls).toEqual([]);
  });

  test("initialize is idempotent", async () => {
    const model = createModel(new FakeInference([]));

    await model.initialize();
    await model.initialize();

    expect(model.importCalls).toBe(1);
    expect(model.createInferenceCalls).toBe(1);
  });

  test("bindTools renders tools into the prompt and invoke returns tool calls", async () => {
    const fakeInference = new FakeInference([
      {
        type: "full",
        value: [
          "I should call a tool.",
          "<|tool_call>",
          "call:get_current_weather{id:<|\"|>call_weather_1<|\"|>,location:<|\"|>Berlin<|\"|>}",
          "<tool_call|>",
        ].join("\n"),
      },
    ]);
    const chatModel = createModel(fakeInference);

    await chatModel.initialize();
    const model = chatModel.bindTools([weatherTool]);
    const response = await model.invoke("What is the weather in Berlin?");

    expect(fakeInference.prompts).toHaveLength(1);
    expect(fakeInference.prompts[0]).toContain("<|tool>");
    expect(fakeInference.prompts[0]).toContain("get_current_weather");
    expect(response.content).toBe("I should call a tool.");
    expect(response.tool_calls).toHaveLength(1);
    expect(response.tool_calls?.[0]).toMatchObject({
      id: "call_weather_1",
      name: "get_current_weather",
      args: {
        location: "Berlin",
      },
    });
  });

  test("tool_choice controls how tools are rendered into the prompt", async () => {
    const noneInference = new FakeInference([
      {
        type: "full",
        value: "No tools.",
      },
    ]);
    const noneChatModel = createModel(noneInference);
    await noneChatModel.initialize();
    const noneModel = noneChatModel.bindTools([weatherTool], {
      tool_choice: "none",
    });
    await noneModel.invoke("Do not call tools.");

    expect(noneInference.prompts[0]).not.toContain("<|tool>");
    expect(noneInference.prompts[0]).toContain(
      "<|turn>user\nDo not call tools."
    );

    const requiredInference = new FakeInference([
      {
        type: "full",
        value: "Calling a tool.",
      },
    ]);
    const requiredChatModel = createModel(requiredInference);
    await requiredChatModel.initialize();
    await requiredChatModel
      .bindTools([weatherTool], {
        tool_choice: "required",
      })
      .invoke("Use a tool.");

    expect(requiredInference.prompts[0]).toContain(
      "You must emit at least one tool call before providing a final answer."
    );

    const explicitInference = new FakeInference([
      {
        type: "full",
        value: "Calling get_current_weather.",
      },
    ]);
    const explicitChatModel = createModel(explicitInference);
    await explicitChatModel.initialize();
    await explicitChatModel
      .bindTools([weatherTool], {
        tool_choice: "get_current_weather",
      })
      .invoke("Call the weather tool.");

    expect(explicitInference.prompts[0]).toContain(
      "You must call the tool named get_current_weather before providing a final answer."
    );
  });

  test("replays tool responses back into the next prompt", async () => {
    const fakeInference = new FakeInference([
      {
        type: "full",
        value: "Thanks for the tool output.",
      },
    ]);
    const chatModel = createModel(fakeInference);

    await chatModel.initialize();
    const model = chatModel.bindTools([weatherTool]);
    await model.invoke([
      new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "call_weather_1",
            type: "tool_call",
            name: "get_current_weather",
            args: { location: "Berlin" },
          },
        ],
      }),
      new ToolMessage({
        content: '{"temperature":20}',
        tool_call_id: "call_weather_1",
      }),
    ]);

    expect(fakeInference.prompts[0]).toContain("<|tool_call>");
    expect(fakeInference.prompts[0]).toContain("<|tool_response>");
    expect(fakeInference.prompts[0]).toContain("call:get_current_weather{");
    expect(fakeInference.prompts[0]).toContain("response:get_current_weather{");
    expect(fakeInference.prompts[0]).toContain("call_weather_1");
  });

  test("invoke reports invalid tool calls for unbound tools", async () => {
    const fakeInference = new FakeInference([
      {
        type: "full",
        value: [
          "I should call a tool.",
          "<|tool_call>",
          "call:lookup_population{id:<|\"|>call_unknown_1<|\"|>,location:<|\"|>Berlin<|\"|>}",
          "<tool_call|>",
        ].join("\n"),
      },
    ]);
    const chatModel = createModel(fakeInference);

    await chatModel.initialize();
    const response = await chatModel
      .bindTools([weatherTool])
      .invoke("What is the population of Berlin?");

    expect(response.content).toBe("I should call a tool.");
    expect(response.tool_calls).toEqual([]);
    expect(response.invalid_tool_calls).toHaveLength(1);
  });

  test("rejects unsupported stop sequences", async () => {
    const chatModel = createModel(
      new FakeInference([
        {
          type: "full",
          value: "unused",
        },
      ])
    );

    await chatModel.initialize();

    await expect(
      chatModel.invoke("Hello", {
        stop: ["Goodbye"],
      })
    ).rejects.toThrow("does not support stop sequences");
  });

  test("rejects unsupported AbortSignal cancellation", async () => {
    const chatModel = createModel(
      new FakeInference([
        {
          type: "full",
          value: "unused",
        },
      ])
    );

    await chatModel.initialize();

    await expect(
      chatModel.invoke("Hello", {
        signal: new AbortController().signal,
      })
    ).rejects.toThrow("does not support AbortSignal cancellation");
  });

  test("works inside a LangChain tool-calling agent", async () => {
    const fakeInference = new FakeInference([
      {
        type: "full",
        value: [
          "I should call a tool.",
          "<|tool_call>",
          "call:get_current_weather{id:<|\"|>call_weather_1<|\"|>,location:<|\"|>Berlin<|\"|>}",
          "<tool_call|>",
        ].join("\n"),
      },
      {
        type: "full",
        value: "The weather in Berlin is 20C.",
      },
    ]);
    const model = createModel(fakeInference);
    await model.initialize();

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are a helpful assistant."],
      ["human", "{input}"],
      ["placeholder", "{agent_scratchpad}"],
    ]);

    const agent = await createToolCallingAgent({
      llm: model,
      tools: [getCurrentWeather],
      prompt,
    });
    const agentExecutor = new AgentExecutor({
      agent,
      tools: [getCurrentWeather],
    });

    const result = await agentExecutor.invoke({
      input: "What is the weather in Berlin?",
    });

    expect(result.output).toBe("The weather in Berlin is 20C.");
    expect(fakeInference.prompts).toHaveLength(2);
    expect(fakeInference.prompts[0]).toContain(
      "<|tool>declaration:get_current_weather{"
    );
    expect(fakeInference.prompts[0]).toContain("get_current_weather");
    expect(fakeInference.prompts[1]).toContain(
      "<|tool_response>response:get_current_weather{"
    );
    expect(fakeInference.prompts[1]).toContain("It is 20C in Berlin.");
  });

  test("stream hides raw tool tags and emits a final tool_call chunk", async () => {
    const fakeInference = new FakeInference([
      {
        type: "stream",
        chunks: [
          "Checking weather...",
          "\n<|tool_call>\n",
          "call:get_current_weather{id:<|\"|>call_weather_1<|\"|>,location:<|\"|>Berlin<|\"|>}",
          "\n<tool_call|>",
        ],
      },
    ]);
    const chatModel = createModel(fakeInference);

    await chatModel.initialize();
    const model = chatModel.bindTools([weatherTool]);

    const chunks: AIMessageChunk[] = [];
    const stream = await model.stream("Weather in Berlin?");
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks.map((chunk) => chunk.content).join("")).toBe(
      "Checking weather..."
    );
    expect(
      chunks.some(
        (chunk) =>
          chunk.tool_call_chunks?.some(
            (toolCallChunk) => toolCallChunk.name === "get_current_weather"
          ) ?? false
      )
    ).toBe(true);
  });

  test("passes image content parts as multimodal prompt", async () => {
    const fakeInference = new FakeInference([
      { type: "full", value: "This is a photo of a cat." },
    ]);
    const model = createModel(fakeInference);
    await model.initialize();

    await model.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: "What is in this image?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/cat.jpg" },
          },
        ],
      }),
    ]);

    const prompt = fakeInference.prompts[0];
    expect(Array.isArray(prompt)).toBe(true);
    const parts = prompt as Array<unknown>;

    const textParts = parts.filter((p) => typeof p === "string");
    expect(textParts.join("")).toContain("What is in this image?");
    expect(textParts.join("")).toContain("<|turn>user");

    const imageParts = parts.filter(
      (p) =>
        typeof p === "object" &&
        p !== null &&
        "imageSource" in p
    );
    expect(imageParts).toHaveLength(1);
    expect(imageParts[0]).toEqual({
      imageSource: "https://example.com/cat.jpg",
    });
  });

  test("passes audio content parts as multimodal prompt", async () => {
    const fakeInference = new FakeInference([
      { type: "full", value: "The audio says hello." },
    ]);
    const model = createModel(fakeInference);
    await model.initialize();

    await model.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: "Transcribe this." },
          {
            type: "input_audio",
            input_audio: { data: "AAAA", format: "wav" },
          },
        ],
      }),
    ]);

    const prompt = fakeInference.prompts[0];
    expect(Array.isArray(prompt)).toBe(true);
    const parts = prompt as Array<unknown>;

    const audioParts = parts.filter(
      (p) =>
        typeof p === "object" &&
        p !== null &&
        "audioSource" in p
    );
    expect(audioParts).toHaveLength(1);
    expect(audioParts[0]).toEqual({
      audioSource: "data:audio/wav;base64,AAAA",
    });
  });

  test("text-only messages still produce a string prompt", async () => {
    const fakeInference = new FakeInference([
      { type: "full", value: "Hello." },
    ]);
    const model = createModel(fakeInference);
    await model.initialize();

    await model.invoke([new HumanMessage("Hi there")]);

    expect(typeof fakeInference.prompts[0]).toBe("string");
  });
});
