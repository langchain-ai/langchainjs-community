import { expect, describe, test, beforeEach, afterEach } from "vitest";
import { NotteClient, Session } from "notte-sdk";
import { ChatOpenAI } from "@langchain/openai";
import { NotteBrowserToolkit } from "../notte_browser.js";

describe("NotteBrowserToolkit Integration Tests", () => {
  let client: NotteClient;
  let session: Session;
  let toolkit: NotteBrowserToolkit;

  beforeEach(async () => {
    client = new NotteClient();
    session = client.Session({ headless: true, idle_timeout_minutes: 1 });
    await session.start();
    toolkit = await NotteBrowserToolkit.fromClient(client, session);
  });

  afterEach(async () => {
    await session.stop().catch(() => {});
  });

  test("should expose the four expected tools", () => {
    const names = toolkit.tools.map((t) => t.name).sort();
    expect(names).toEqual([
      "notte_act",
      "notte_extract",
      "notte_navigate",
      "notte_observe",
    ]);
  });

  test("should perform basic navigation", async () => {
    const navigateTool = toolkit.tools.find((t) => t.name === "notte_navigate");
    if (!navigateTool) {
      throw new Error("Navigate tool not found");
    }
    const result = (await navigateTool.invoke(
      "https://www.ecosia.org"
    )) as string;
    expect(result).toContain("Successfully navigated");
  });

  test("should extract structured data from a webpage", async () => {
    const navigateTool = toolkit.tools.find((t) => t.name === "notte_navigate");
    if (!navigateTool) {
      throw new Error("Navigate tool not found");
    }
    await navigateTool.invoke("https://example.com");

    const extractTool = toolkit.tools.find((t) => t.name === "notte_extract");
    if (!extractTool) {
      throw new Error("Extract tool not found");
    }
    const result = (await extractTool.invoke({
      instruction: "Extract the page heading and the body paragraph text.",
      schema: {
        type: "object",
        properties: {
          heading: { type: "string" },
          paragraph: { type: "string" },
        },
        required: ["heading", "paragraph"],
      },
    })) as string;
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  test("should use observe tool to list interactive elements", async () => {
    const navigateTool = toolkit.tools.find((t) => t.name === "notte_navigate");
    if (!navigateTool) {
      throw new Error("Navigate tool not found");
    }
    await navigateTool.invoke("https://github.com/nottelabs/notte");

    const observeTool = toolkit.tools.find((t) => t.name === "notte_observe");
    if (!observeTool) {
      throw new Error("Observe tool not found");
    }
    const raw = (await observeTool.invoke("")) as string;
    const observation = JSON.parse(raw);
    expect(observation.space).toBeDefined();
    expect(observation.space.interaction_actions).toBeDefined();
  });

  test("should bind tools to an LLM", async () => {
    const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

    if (!llm.bindTools) {
      throw new Error("Language model does not support tools.");
    }

    const llmWithTools = llm.bindTools(toolkit.tools);
    const result = await llmWithTools.invoke(
      "Navigate to https://www.ecosia.org"
    );

    expect(result.tool_calls).toBeDefined();
    expect(result.tool_calls?.length).toBe(1);
    const toolCall = result.tool_calls?.[0];
    expect(toolCall?.name).toBe("notte_navigate");
  });
});
