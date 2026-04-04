import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { Tool } from "@langchain/core/tools";

/**
 * Interface for Suwappu tool parameters.
 */
export interface SuwappuParams {
  apiKey?: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function createSuwappuClient(apiKey: string): Promise<any> {
  try {
    const { createClient } = await import("@suwappu/sdk");
    return createClient({ apiKey });
  } catch {
    throw new Error(
      'The "@suwappu/sdk" package is required for Suwappu tools. ' +
        "Install it with: npm install @suwappu/sdk"
    );
  }
}

/**
 * Tool for getting real-time token prices via the Suwappu DEX aggregator.
 *
 * Setup:
 * Install `@langchain/community` and `@suwappu/sdk`.
 *
 * ```bash
 * npm install @langchain/community @suwappu/sdk
 * ```
 *
 * Set your API key:
 * ```bash
 * export SUWAPPU_API_KEY=your_key_here
 * ```
 *
 * ## [Constructor args](https://api.js.langchain.com/classes/_langchain_community.tools_suwappu.SuwappuGetPricesTool.html)
 *
 * <details open>
 * <summary><strong>Instantiate</strong></summary>
 *
 * ```typescript
 * import { SuwappuGetPricesTool } from "@langchain/community/tools/suwappu";
 *
 * const tool = new SuwappuGetPricesTool();
 * ```
 * </details>
 *
 * <details>
 * <summary><strong>Invocation</strong></summary>
 *
 * ```typescript
 * await tool.invoke("ETH");
 * // or with chain filter:
 * await tool.invoke(JSON.stringify({ token: "ETH", chain: "base" }));
 * ```
 * </details>
 */
export class SuwappuGetPricesTool extends Tool {
  static lc_name() {
    return "SuwappuGetPricesTool";
  }

  name = "suwappu_get_prices";

  description =
    "Get the current USD price and 24-hour change for a cryptocurrency token. " +
    'Input: token symbol (e.g. "ETH") or JSON with token and optional chain ' +
    '(e.g. {"token": "ETH", "chain": "base"}).';

  apiKey: string;

  constructor(fields?: SuwappuParams) {
    super();
    const apiKey =
      fields?.apiKey ?? getEnvironmentVariable("SUWAPPU_API_KEY");
    if (!apiKey) {
      throw new Error(
        'Suwappu API key not set. Pass it in or set the "SUWAPPU_API_KEY" environment variable.'
      );
    }
    this.apiKey = apiKey;
  }

  /** @ignore */
  async _call(input: string): Promise<string> {
    let token: string;
    let chain: string | undefined;

    try {
      const parsed = JSON.parse(input);
      token = parsed.token;
      chain = parsed.chain;
    } catch {
      token = input.trim();
    }

    if (!token) {
      return JSON.stringify({ error: "Missing token symbol." });
    }

    const client = await createSuwappuClient(this.apiKey);
    try {
      const prices = await client.getPrices(token, chain);
      return JSON.stringify(prices);
    } finally {
      await client.close();
    }
  }
}

/**
 * Tool for getting swap quotes via the Suwappu DEX aggregator.
 *
 * Returns a detailed quote for swapping one token to another, including
 * the best execution route, estimated gas costs, and protocol fees.
 *
 * Setup:
 * ```bash
 * npm install @langchain/community @suwappu/sdk
 * export SUWAPPU_API_KEY=your_key_here
 * ```
 */
export class SuwappuGetQuoteTool extends Tool {
  static lc_name() {
    return "SuwappuGetQuoteTool";
  }

  name = "suwappu_get_quote";

  description =
    "Get a swap quote for trading one token for another. Returns price impact, " +
    "route, estimated gas, and fees. Input: JSON with from_token, to_token, amount, " +
    'and chain (e.g. {"from_token": "ETH", "to_token": "USDC", "amount": 1.0, "chain": "base"}).';

  apiKey: string;

  constructor(fields?: SuwappuParams) {
    super();
    const apiKey =
      fields?.apiKey ?? getEnvironmentVariable("SUWAPPU_API_KEY");
    if (!apiKey) {
      throw new Error(
        'Suwappu API key not set. Pass it in or set the "SUWAPPU_API_KEY" environment variable.'
      );
    }
    this.apiKey = apiKey;
  }

  /** @ignore */
  async _call(input: string): Promise<string> {
    const { from_token, to_token, amount, chain } = JSON.parse(input);

    if (!from_token || !to_token || !amount || !chain) {
      return JSON.stringify({
        error:
          "Missing required fields. Provide from_token, to_token, amount, and chain.",
      });
    }

    const client = await createSuwappuClient(this.apiKey);
    try {
      const quote = await client.getQuote(
        from_token,
        to_token,
        Number(amount),
        chain
      );
      return JSON.stringify(quote);
    } finally {
      await client.close();
    }
  }
}

/**
 * Tool for executing a previously quoted swap via the Suwappu DEX aggregator.
 *
 * Takes a quote_id returned from SuwappuGetQuoteTool and executes the swap.
 *
 * Setup:
 * ```bash
 * npm install @langchain/community @suwappu/sdk
 * export SUWAPPU_API_KEY=your_key_here
 * ```
 */
export class SuwappuExecuteSwapTool extends Tool {
  static lc_name() {
    return "SuwappuExecuteSwapTool";
  }

  name = "suwappu_execute_swap";

  description =
    "Execute a previously quoted swap. Input: quote_id string " +
    "(the ID returned from suwappu_get_quote).";

  apiKey: string;

  constructor(fields?: SuwappuParams) {
    super();
    const apiKey =
      fields?.apiKey ?? getEnvironmentVariable("SUWAPPU_API_KEY");
    if (!apiKey) {
      throw new Error(
        'Suwappu API key not set. Pass it in or set the "SUWAPPU_API_KEY" environment variable.'
      );
    }
    this.apiKey = apiKey;
  }

  /** @ignore */
  async _call(input: string): Promise<string> {
    const quoteId = input.trim();
    if (!quoteId) {
      return JSON.stringify({ error: "Missing quote_id." });
    }

    const client = await createSuwappuClient(this.apiKey);
    try {
      const result = await client.executeSwap(quoteId);
      return JSON.stringify(result);
    } finally {
      await client.close();
    }
  }
}

/**
 * Tool for checking wallet token balances via the Suwappu DEX aggregator.
 *
 * Returns current token holdings and balances for the connected wallet,
 * optionally filtered to a specific chain.
 *
 * Setup:
 * ```bash
 * npm install @langchain/community @suwappu/sdk
 * export SUWAPPU_API_KEY=your_key_here
 * ```
 */
export class SuwappuPortfolioTool extends Tool {
  static lc_name() {
    return "SuwappuPortfolioTool";
  }

  name = "suwappu_get_portfolio";

  description =
    "Check wallet token balances across blockchain networks. " +
    'Input: optional chain name to filter (e.g. "base"). Leave empty for all chains.';

  apiKey: string;

  constructor(fields?: SuwappuParams) {
    super();
    const apiKey =
      fields?.apiKey ?? getEnvironmentVariable("SUWAPPU_API_KEY");
    if (!apiKey) {
      throw new Error(
        'Suwappu API key not set. Pass it in or set the "SUWAPPU_API_KEY" environment variable.'
      );
    }
    this.apiKey = apiKey;
  }

  /** @ignore */
  async _call(input: string): Promise<string> {
    const chain = input.trim() || undefined;

    const client = await createSuwappuClient(this.apiKey);
    try {
      const portfolio = await client.getPortfolio(chain);
      return JSON.stringify(portfolio);
    } finally {
      await client.close();
    }
  }
}

/**
 * Tool for listing supported blockchain networks via the Suwappu DEX aggregator.
 *
 * Returns the full list of supported chains (Ethereum, Base, Arbitrum,
 * Solana, etc.) with their chain IDs and metadata.
 *
 * Setup:
 * ```bash
 * npm install @langchain/community @suwappu/sdk
 * export SUWAPPU_API_KEY=your_key_here
 * ```
 */
export class SuwappuChainsTool extends Tool {
  static lc_name() {
    return "SuwappuChainsTool";
  }

  name = "suwappu_list_chains";

  description =
    "List all blockchain networks supported by the Suwappu DEX aggregator. " +
    "No input required.";

  apiKey: string;

  constructor(fields?: SuwappuParams) {
    super();
    const apiKey =
      fields?.apiKey ?? getEnvironmentVariable("SUWAPPU_API_KEY");
    if (!apiKey) {
      throw new Error(
        'Suwappu API key not set. Pass it in or set the "SUWAPPU_API_KEY" environment variable.'
      );
    }
    this.apiKey = apiKey;
  }

  /** @ignore */
  async _call(_input: string): Promise<string> {
    const client = await createSuwappuClient(this.apiKey);
    try {
      const chains = await client.listChains();
      return JSON.stringify(chains);
    } finally {
      await client.close();
    }
  }
}

/**
 * Tool for listing available tokens on a specific chain via the Suwappu DEX aggregator.
 *
 * Returns all tradeable tokens on the given chain, including symbols,
 * addresses, and decimals.
 *
 * Setup:
 * ```bash
 * npm install @langchain/community @suwappu/sdk
 * export SUWAPPU_API_KEY=your_key_here
 * ```
 */
export class SuwappuTokensTool extends Tool {
  static lc_name() {
    return "SuwappuTokensTool";
  }

  name = "suwappu_list_tokens";

  description =
    "List available tokens for trading on a specific blockchain. " +
    'Input: chain name (e.g. "base", "ethereum"). Leave empty for all tokens.';

  apiKey: string;

  constructor(fields?: SuwappuParams) {
    super();
    const apiKey =
      fields?.apiKey ?? getEnvironmentVariable("SUWAPPU_API_KEY");
    if (!apiKey) {
      throw new Error(
        'Suwappu API key not set. Pass it in or set the "SUWAPPU_API_KEY" environment variable.'
      );
    }
    this.apiKey = apiKey;
  }

  /** @ignore */
  async _call(input: string): Promise<string> {
    const chain = input.trim() || undefined;

    const client = await createSuwappuClient(this.apiKey);
    try {
      const tokens = await client.listTokens(chain);
      return JSON.stringify(tokens);
    } finally {
      await client.close();
    }
  }
}
