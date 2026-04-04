import { vi, afterEach, beforeEach, describe, expect, test } from "vitest";
import {
  SuwappuGetPricesTool,
  SuwappuGetQuoteTool,
  SuwappuExecuteSwapTool,
  SuwappuPortfolioTool,
  SuwappuChainsTool,
  SuwappuTokensTool,
} from "../suwappu.js";

const MOCK_API_KEY = "test-key-123";

// Mock the @suwappu/sdk module
const mockClose = vi.fn().mockResolvedValue(undefined);
const mockGetPrices = vi.fn();
const mockGetQuote = vi.fn();
const mockExecuteSwap = vi.fn();
const mockGetPortfolio = vi.fn();
const mockListChains = vi.fn();
const mockListTokens = vi.fn();

vi.mock("@suwappu/sdk", () => ({
  createClient: () => ({
    getPrices: mockGetPrices,
    getQuote: mockGetQuote,
    executeSwap: mockExecuteSwap,
    getPortfolio: mockGetPortfolio,
    listChains: mockListChains,
    listTokens: mockListTokens,
    close: mockClose,
  }),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("SuwappuGetPricesTool", () => {
  test("initializes with correct name and description", () => {
    const tool = new SuwappuGetPricesTool({ apiKey: MOCK_API_KEY });
    expect(tool.name).toBe("suwappu_get_prices");
    expect(tool.description).toContain("price");
  });

  test("throws without API key", () => {
    expect(() => new SuwappuGetPricesTool()).toThrow("API key not set");
  });

  test("fetches prices for a token string", async () => {
    mockGetPrices.mockResolvedValue({
      symbol: "ETH",
      price_usd: 3200.5,
      change_24h: 2.5,
    });

    const tool = new SuwappuGetPricesTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call("ETH");
    const parsed = JSON.parse(result);

    expect(parsed.symbol).toBe("ETH");
    expect(parsed.price_usd).toBe(3200.5);
    expect(mockGetPrices).toHaveBeenCalledWith("ETH", undefined);
    expect(mockClose).toHaveBeenCalled();
  });

  test("fetches prices with chain filter via JSON input", async () => {
    mockGetPrices.mockResolvedValue({
      symbol: "ETH",
      price_usd: 3200.5,
    });

    const tool = new SuwappuGetPricesTool({ apiKey: MOCK_API_KEY });
    await tool._call(JSON.stringify({ token: "ETH", chain: "base" }));

    expect(mockGetPrices).toHaveBeenCalledWith("ETH", "base");
  });

  test("returns error for empty input", async () => {
    const tool = new SuwappuGetPricesTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call("");
    const parsed = JSON.parse(result);
    expect(parsed.error).toContain("Missing token");
  });

  test("static lc_name returns correct value", () => {
    expect(SuwappuGetPricesTool.lc_name()).toBe("SuwappuGetPricesTool");
  });
});

describe("SuwappuGetQuoteTool", () => {
  test("initializes with correct name", () => {
    const tool = new SuwappuGetQuoteTool({ apiKey: MOCK_API_KEY });
    expect(tool.name).toBe("suwappu_get_quote");
    expect(tool.description).toContain("quote");
  });

  test("fetches a swap quote", async () => {
    mockGetQuote.mockResolvedValue({
      quote_id: "q-123",
      from_token: "ETH",
      to_token: "USDC",
      amount_in: 1.0,
      amount_out: 3200.0,
    });

    const tool = new SuwappuGetQuoteTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call(
      JSON.stringify({
        from_token: "ETH",
        to_token: "USDC",
        amount: 1.0,
        chain: "base",
      })
    );
    const parsed = JSON.parse(result);

    expect(parsed.quote_id).toBe("q-123");
    expect(mockGetQuote).toHaveBeenCalledWith("ETH", "USDC", 1.0, "base");
    expect(mockClose).toHaveBeenCalled();
  });

  test("returns error for missing fields", async () => {
    const tool = new SuwappuGetQuoteTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call(
      JSON.stringify({ from_token: "ETH" })
    );
    const parsed = JSON.parse(result);
    expect(parsed.error).toContain("Missing required fields");
  });
});

describe("SuwappuExecuteSwapTool", () => {
  test("initializes with correct name", () => {
    const tool = new SuwappuExecuteSwapTool({ apiKey: MOCK_API_KEY });
    expect(tool.name).toBe("suwappu_execute_swap");
  });

  test("executes a swap with quote_id", async () => {
    mockExecuteSwap.mockResolvedValue({
      tx_hash: "0xabc123",
      status: "confirmed",
    });

    const tool = new SuwappuExecuteSwapTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call("q-123");
    const parsed = JSON.parse(result);

    expect(parsed.tx_hash).toBe("0xabc123");
    expect(mockExecuteSwap).toHaveBeenCalledWith("q-123");
    expect(mockClose).toHaveBeenCalled();
  });

  test("returns error for empty input", async () => {
    const tool = new SuwappuExecuteSwapTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call("");
    const parsed = JSON.parse(result);
    expect(parsed.error).toContain("Missing quote_id");
  });
});

describe("SuwappuPortfolioTool", () => {
  test("initializes with correct name", () => {
    const tool = new SuwappuPortfolioTool({ apiKey: MOCK_API_KEY });
    expect(tool.name).toBe("suwappu_get_portfolio");
  });

  test("fetches portfolio for a specific chain", async () => {
    mockGetPortfolio.mockResolvedValue([
      { token: "ETH", balance: 1.5, value_usd: 4800.0 },
      { token: "USDC", balance: 1000.0, value_usd: 1000.0 },
    ]);

    const tool = new SuwappuPortfolioTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call("base");
    const parsed = JSON.parse(result);

    expect(parsed).toHaveLength(2);
    expect(parsed[0].token).toBe("ETH");
    expect(mockGetPortfolio).toHaveBeenCalledWith("base");
    expect(mockClose).toHaveBeenCalled();
  });

  test("fetches portfolio for all chains with empty input", async () => {
    mockGetPortfolio.mockResolvedValue([]);

    const tool = new SuwappuPortfolioTool({ apiKey: MOCK_API_KEY });
    await tool._call("");

    expect(mockGetPortfolio).toHaveBeenCalledWith(undefined);
  });
});

describe("SuwappuChainsTool", () => {
  test("initializes with correct name", () => {
    const tool = new SuwappuChainsTool({ apiKey: MOCK_API_KEY });
    expect(tool.name).toBe("suwappu_list_chains");
  });

  test("lists supported chains", async () => {
    mockListChains.mockResolvedValue([
      { name: "Ethereum", chain_id: 1 },
      { name: "Base", chain_id: 8453 },
    ]);

    const tool = new SuwappuChainsTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call("");
    const parsed = JSON.parse(result);

    expect(parsed).toHaveLength(2);
    expect(parsed[0].name).toBe("Ethereum");
    expect(mockClose).toHaveBeenCalled();
  });
});

describe("SuwappuTokensTool", () => {
  test("initializes with correct name", () => {
    const tool = new SuwappuTokensTool({ apiKey: MOCK_API_KEY });
    expect(tool.name).toBe("suwappu_list_tokens");
  });

  test("lists tokens for a chain", async () => {
    mockListTokens.mockResolvedValue([
      { symbol: "ETH", address: "0x...", decimals: 18 },
      { symbol: "USDC", address: "0x...", decimals: 6 },
    ]);

    const tool = new SuwappuTokensTool({ apiKey: MOCK_API_KEY });
    const result = await tool._call("base");
    const parsed = JSON.parse(result);

    expect(parsed).toHaveLength(2);
    expect(parsed[0].symbol).toBe("ETH");
    expect(mockListTokens).toHaveBeenCalledWith("base");
    expect(mockClose).toHaveBeenCalled();
  });
});
