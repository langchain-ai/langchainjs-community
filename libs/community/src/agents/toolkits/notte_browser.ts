import {
  Tool,
  BaseToolkit as Toolkit,
  StructuredTool,
  StructuredToolInterface,
} from "@langchain/core/tools";
import { NotteClient, Session } from "notte-sdk";
import { z } from "zod/v3";

//  Documentation:
//  https://notte.cc — product
//  https://docs.notte.cc — API + SDK reference

function isErrorWithMessage(error: unknown): error is { message: string } {
  return (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof (error as { message: unknown }).message === "string"
  );
}

abstract class NotteBrowserToolBase extends Tool {
  protected client: NotteClient;

  protected session?: Session;

  private localSession?: Session;

  constructor(client: NotteClient, session?: Session) {
    super();
    this.client = client;
    this.session = session;
  }

  protected async getSession(): Promise<Session> {
    if (this.session) return this.session;

    if (!this.localSession) {
      this.localSession = this.client.Session({ headless: true });
      await this.localSession.start();
    }
    return this.localSession;
  }

  protected async closeLocalSession(): Promise<void> {
    if (this.localSession) {
      try {
        await this.localSession.stop();
      } finally {
        this.localSession = undefined;
      }
    }
  }
}

export class NotteNavigateTool extends NotteBrowserToolBase {
  name = "notte_navigate";

  description =
    "Use this tool to navigate the Notte browser session to a specific URL. The input should be a single, valid URL as a string.";

  async _call(input: string): Promise<string> {
    const session = await this.getSession();
    try {
      const result = await session.execute({ type: "goto", url: input });
      if (result.success) {
        return `Successfully navigated to ${input}.`;
      }
      return `Failed to navigate to ${input}: ${result.message}`;
    } catch (error: unknown) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      return `Failed to navigate to ${input}: ${message}`;
    }
  }
}

export class NotteActTool extends NotteBrowserToolBase {
  name = "notte_act";

  description =
    "Use this tool to perform a natural-language action on the current web page using Notte's perception model (e.g. 'click the sign-up button', 'fill the search box with cats'). The input should be a single string describing the action to perform. The action is interpreted and executed against the live page; the tool returns a short summary of the outcome.";

  async _call(input: string): Promise<string> {
    const session = await this.getSession();
    try {
      const agent = this.client.Agent({ session, max_steps: 3 });
      const result = await agent.run({ task: input });
      const summary = result.answer ?? "no summary returned";
      if (result.success) {
        return `Action performed successfully: ${summary}`;
      }
      return `Failed to perform action: ${summary}`;
    } catch (error: unknown) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      return `Failed to perform action: ${message}`;
    }
  }
}

export class NotteObserveTool extends NotteBrowserToolBase {
  name = "notte_observe";

  description =
    "Use this tool to observe the current web page and retrieve the list of interactive elements (buttons, inputs, links) that an agent can act on. Useful for planning the next action. The input is unused; pass an empty string.";

  async _call(_input: string): Promise<string> {
    const session = await this.getSession();
    try {
      const observation = await session.observe("fast");
      return JSON.stringify(observation);
    } catch (error: unknown) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      return `Failed to observe: ${message}`;
    }
  }
}

export class NotteExtractTool extends StructuredTool {
  name = "notte_extract";

  description =
    "Use this tool to extract structured information from the current web page. The input should include an 'instruction' string describing what to extract, and a 'schema' object representing the expected shape of the output in JSON Schema format. Returns the structured extraction as a JSON string.";

  // Match the Stagehand convention: a JSON Schema object passed as a record.
  schema = z.object({
    instruction: z.string().describe("Instruction on what to extract from the page"),
    schema: z
      .record(z.any())
      .describe("Extraction schema in JSON Schema format"),
  });

  protected client: NotteClient;

  protected session?: Session;

  private localSession?: Session;

  constructor(client: NotteClient, session?: Session) {
    super();
    this.client = client;
    this.session = session;
  }

  async _call(input: {
    instruction: string;
    schema: Record<string, unknown>;
  }): Promise<string> {
    const session = await this.getSession();
    const { instruction, schema } = input;

    try {
      // Notte's SDK accepts either a Zod schema (which it converts) or a
      // plain JSON Schema object passed straight through. We send the JSON
      // Schema we received from the LLM verbatim via the `any` overload.
      const result = await session.scrape({
        instructions: instruction,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        response_format: schema as any,
      });
      return typeof result === "string" ? result : JSON.stringify(result);
    } catch (error: unknown) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      return `Failed to extract information: ${message}`;
    }
  }

  protected async getSession(): Promise<Session> {
    if (this.session) return this.session;

    if (!this.localSession) {
      this.localSession = this.client.Session({ headless: true });
      await this.localSession.start();
    }
    return this.localSession;
  }
}

export class NotteBrowserToolkit extends Toolkit {
  tools: StructuredToolInterface[];

  client: NotteClient;

  session?: Session;

  constructor(client: NotteClient, session?: Session) {
    super();
    this.client = client;
    this.session = session;
    this.tools = this.initializeTools();
  }

  private initializeTools(): StructuredToolInterface[] {
    return [
      new NotteNavigateTool(this.client, this.session),
      new NotteActTool(this.client, this.session),
      new NotteExtractTool(this.client, this.session),
      new NotteObserveTool(this.client, this.session),
    ];
  }

  /**
   * Create a toolkit bound to an existing NotteClient. If `session` is
   * provided, every tool reuses it. If not, each tool will lazily create its
   * own session on first call.
   */
  static async fromClient(
    client: NotteClient,
    session?: Session,
  ): Promise<NotteBrowserToolkit> {
    return new NotteBrowserToolkit(client, session);
  }
}
