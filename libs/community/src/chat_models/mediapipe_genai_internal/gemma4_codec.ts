import type { BaseMessage } from "@langchain/core/messages";

import {
  extractTaggedBlocks,
  extractVisibleText,
  GEMMA_TOOL_CALL_TAG_CLOSE,
  GEMMA_TOOL_CALL_TAG_OPEN,
  GEMMA_TOOL_RESPONSE_TAG_CLOSE,
  GEMMA_TOOL_RESPONSE_TAG_OPEN,
  GEMMA_TOOL_TAG_CLOSE,
  GEMMA_TOOL_TAG_OPEN,
  GEMMA_TURN_CLOSE,
  GEMMA_TURN_OPEN,
  getMessageContentParts,
  getMessageTextContent,
  getNormalizedRole,
  getReplayToolCalls,
  parseGemmaToolCall,
  serializeGemmaToolCall,
  serializeGemmaToolDeclaration,
  serializeGemmaToolResponse,
  stripHistoricalThoughts,
  type MediaPipeToolChoice,
  type MessageContentParts,
  type ParsedCodecToolCall,
  type ParsedModelOutput,
  type PromptRenderInput,
  type RenderedPrompt,
  type RenderedPromptPart,
} from "./prompt_codec.js";

function renderTurn(
  role: "system" | "user" | "model",
  content: string
): string {
  return `${GEMMA_TURN_OPEN}${role}\n${content}\n${GEMMA_TURN_CLOSE}`;
}

function renderToolChoiceInstruction(
  toolChoice: MediaPipeToolChoice | undefined
): string {
  if (!toolChoice || toolChoice === "auto") {
    return [
      "Use declared tools when they improve accuracy.",
      `Emit ${GEMMA_TOOL_CALL_TAG_OPEN}call:name{...}${GEMMA_TOOL_CALL_TAG_CLOSE} for tool requests.`,
      `Replay tool results as ${GEMMA_TOOL_RESPONSE_TAG_OPEN}response:name{...}${GEMMA_TOOL_RESPONSE_TAG_CLOSE} before any final answer.`,
    ].join(" ");
  }

  if (toolChoice === "none") {
    return "Do not call tools. Respond with natural language only.";
  }

  if (toolChoice === "required") {
    return "You must emit at least one tool call before providing a final answer.";
  }

  return `You must call the tool named ${toolChoice.function.name} before providing a final answer.`;
}

export interface Gemma4CodecFields {
  thoughtTagName?: string;
}

export class Gemma4Codec {
  private readonly thoughtTagName?: string;

  constructor(fields: Gemma4CodecFields = {}) {
    this.thoughtTagName = fields.thoughtTagName;
  }

  render(input: PromptRenderInput): RenderedPrompt {
    const sections: Array<string | RenderedPromptPart[]> = [];
    let hasMedia = false;

    if (input.tools.length > 0 && input.toolChoice !== "none") {
      const content = [
        ...input.tools.map((tool) => {
          return `${GEMMA_TOOL_TAG_OPEN}${serializeGemmaToolDeclaration(
            tool.function
          )}${GEMMA_TOOL_TAG_CLOSE}`;
        }),
        renderToolChoiceInstruction(input.toolChoice),
      ].join("\n");

      sections.push(renderTurn("system", content));
    }

    for (let index = 0; index < input.messages.length; index += 1) {
      const message = input.messages[index];
      if (message._getType() === "tool") {
        sections.push(this.renderOrphanToolMessage(message));
        continue;
      }

      if (message._getType() !== "ai") {
        const contentParts = getMessageContentParts(message);
        if (contentParts.hasMedia) {
          hasMedia = true;
          sections.push(
            renderMultimodalTurn(getNormalizedRole(message), contentParts)
          );
        } else {
          sections.push(
            renderTurn(getNormalizedRole(message), contentParts.text)
          );
        }
        continue;
      }

      const replayToolCalls = getReplayToolCalls(message);
      const trailingToolMessages: BaseMessage[] = [];
      let trailingIndex = index + 1;
      while (
        trailingIndex < input.messages.length &&
        input.messages[trailingIndex]._getType() === "tool"
      ) {
        trailingToolMessages.push(input.messages[trailingIndex]);
        trailingIndex += 1;
      }

      sections.push(
        renderTurn(
          "model",
          this.renderAssistantTurn(message, replayToolCalls, trailingToolMessages)
        )
      );
      index = trailingIndex - 1;
    }

    sections.push(`${GEMMA_TURN_OPEN}model\n`);

    if (!hasMedia) {
      return (sections as string[]).join("\n");
    }

    return flattenSections(sections);
  }

  parse(rawText: string): ParsedModelOutput {
    const visibleText = extractVisibleText(rawText);
    const toolCalls: ParsedCodecToolCall[] = extractTaggedBlocks(
      rawText,
      GEMMA_TOOL_CALL_TAG_OPEN,
      GEMMA_TOOL_CALL_TAG_CLOSE
    ).map((block) => parseGemmaToolCall(block));

    return {
      content: visibleText.trim(),
      visibleText,
      toolCalls,
    };
  }

  private renderAssistantTurn(
    message: BaseMessage,
    replayToolCalls: Array<{
      id?: string;
      name: string;
      arguments: unknown;
    }>,
    trailingToolMessages: BaseMessage[]
  ): string {
    const preserveThoughts =
      replayToolCalls.length > 0 || trailingToolMessages.length > 0;
    const content = stripHistoricalThoughts(
      getMessageTextContent(message),
      this.thoughtTagName,
      preserveThoughts
    );
    const sections = [
      content,
      ...replayToolCalls.map((toolCall) => {
        return `${GEMMA_TOOL_CALL_TAG_OPEN}${serializeGemmaToolCall(
          toolCall
        )}${GEMMA_TOOL_CALL_TAG_CLOSE}`;
      }),
      ...trailingToolMessages.map((toolMessage) => {
        const toolCallId =
          "tool_call_id" in toolMessage &&
          typeof toolMessage.tool_call_id === "string"
            ? toolMessage.tool_call_id
            : undefined;
        const matchedToolCall = replayToolCalls.find(
          (toolCall) => toolCall.id === toolCallId
        );

        return this.renderToolResponseBlock(toolMessage, matchedToolCall?.name);
      }),
    ].filter(Boolean);

    return sections.join("\n");
  }

  private renderOrphanToolMessage(message: BaseMessage): string {
    return renderTurn("model", this.renderToolResponseBlock(message));
  }

  private renderToolResponseBlock(
    message: BaseMessage,
    fallbackName?: string
  ): string {
    const content = getMessageTextContent(message);
    const toolCallId =
      "tool_call_id" in message && typeof message.tool_call_id === "string"
        ? message.tool_call_id
        : undefined;
    const toolName =
      typeof message.name === "string" ? message.name : fallbackName ?? "tool";

    return `${GEMMA_TOOL_RESPONSE_TAG_OPEN}${serializeGemmaToolResponse({
      id: toolCallId,
      name: toolName,
      content,
    })}${GEMMA_TOOL_RESPONSE_TAG_CLOSE}`;
  }
}

function renderMultimodalTurn(
  role: "system" | "user" | "model",
  contentParts: MessageContentParts
): RenderedPromptPart[] {
  const parts: RenderedPromptPart[] = [`${GEMMA_TURN_OPEN}${role}\n`];
  for (const part of contentParts.orderedParts) {
    if (typeof part === "string") {
      appendText(parts, part);
    } else {
      parts.push(part);
    }
  }
  appendText(parts, `\n${GEMMA_TURN_CLOSE}`);
  return parts;
}

function appendText(parts: RenderedPromptPart[], text: string): void {
  if (parts.length > 0 && typeof parts[parts.length - 1] === "string") {
    (parts as string[])[parts.length - 1] += text;
  } else {
    parts.push(text);
  }
}

function flattenSections(
  sections: Array<string | RenderedPromptPart[]>
): RenderedPromptPart[] {
  const result: RenderedPromptPart[] = [];
  for (let i = 0; i < sections.length; i++) {
    if (i > 0) {
      appendText(result, "\n");
    }
    const section = sections[i];
    if (typeof section === "string") {
      appendText(result, section);
    } else {
      for (const part of section) {
        if (typeof part === "string") {
          appendText(result, part);
        } else {
          result.push(part);
        }
      }
    }
  }
  return result;
}
