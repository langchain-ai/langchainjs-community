import { vi, afterEach, beforeEach, describe, expect } from "vitest";
import { UniRateAPIWrapper, UniRateExchangeTool } from "../unirate.js";

const MOCK_API_KEY = "test-api-key";

function jsonResponse(body: unknown, init: { status?: number } = {}): Response {
  return {
    ok: (init.status ?? 200) < 400,
    status: init.status ?? 200,
    json: () => Promise.resolve(body),
  } as Response;
}

describe("UniRate tool", () => {
  // oxlint-disable-next-line typescript/no-explicit-any
  let fetchMock: any;

  beforeEach(() => {
    fetchMock = vi.spyOn(global, "fetch");
  });

  afterEach(() => {
    fetchMock.mockRestore();
  });

  test("convert sends api_key + Accept header and parses result", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ result: "92.50" }));

    const wrapper = new UniRateAPIWrapper({ apiKey: MOCK_API_KEY });
    const result = await wrapper.convert("usd", "eur", 100);

    expect(result).toBe(92.5);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [calledUrl, calledInit] = fetchMock.mock.calls[0];
    const url = calledUrl instanceof URL ? calledUrl : new URL(calledUrl);
    expect(url.origin + url.pathname).toBe(
      "https://api.unirateapi.com/api/convert"
    );
    expect(url.searchParams.get("api_key")).toBe(MOCK_API_KEY);
    expect(url.searchParams.get("from")).toBe("USD");
    expect(url.searchParams.get("to")).toBe("EUR");
    expect(url.searchParams.get("amount")).toBe("100");

    const headers = (calledInit?.headers ?? {}) as Record<string, string>;
    expect(headers.Accept).toBe("application/json");
  });

  test("getRate without `to` returns the full map of rates", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ rates: { EUR: "0.92", GBP: "0.79" } })
    );

    const wrapper = new UniRateAPIWrapper({ apiKey: MOCK_API_KEY });
    const rates = await wrapper.getRate("USD");

    expect(rates).toEqual({ EUR: 0.92, GBP: 0.79 });

    const [calledUrl] = fetchMock.mock.calls[0];
    const url = calledUrl instanceof URL ? calledUrl : new URL(calledUrl);
    expect(url.searchParams.get("from")).toBe("USD");
    expect(url.searchParams.has("to")).toBe(false);
  });

  test("getRate with `to` returns a single number", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ rate: "0.92" }));

    const wrapper = new UniRateAPIWrapper({ apiKey: MOCK_API_KEY });
    const rate = await wrapper.getRate("USD", "EUR");

    expect(rate).toBe(0.92);
  });

  test("getSupportedCurrencies returns the currency code list", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ currencies: ["USD", "EUR", "GBP"] })
    );

    const wrapper = new UniRateAPIWrapper({ apiKey: MOCK_API_KEY });
    const codes = await wrapper.getSupportedCurrencies();

    expect(codes).toEqual(["USD", "EUR", "GBP"]);
  });

  test("401 response is mapped to a clear authentication error", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}, { status: 401 }));

    const wrapper = new UniRateAPIWrapper({ apiKey: MOCK_API_KEY });
    await expect(wrapper.convert("USD", "EUR", 1)).rejects.toThrow(
      "Missing or invalid UniRate API key"
    );
  });

  test("403 response surfaces the Pro-subscription requirement", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}, { status: 403 }));

    const wrapper = new UniRateAPIWrapper({ apiKey: MOCK_API_KEY });
    await expect(wrapper.convert("USD", "EUR", 1)).rejects.toThrow(
      "This UniRate endpoint requires a Pro subscription"
    );
  });

  test("429 response surfaces a rate-limit error", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}, { status: 429 }));

    const wrapper = new UniRateAPIWrapper({ apiKey: MOCK_API_KEY });
    await expect(wrapper.convert("USD", "EUR", 1)).rejects.toThrow(
      "UniRate API rate limit exceeded"
    );
  });

  test("constructor throws when no API key is available", () => {
    const previous = process.env.UNIRATE_API_KEY;
    delete process.env.UNIRATE_API_KEY;
    try {
      expect(() => new UniRateAPIWrapper()).toThrow(/UNIRATE_API_KEY/);
    } finally {
      if (previous !== undefined) {
        process.env.UNIRATE_API_KEY = previous;
      }
    }
  });

  test("UniRateExchangeTool.invoke formats the result for the LLM", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ result: "92.50" }));

    const tool = new UniRateExchangeTool({ apiKey: MOCK_API_KEY });
    const out = await tool.invoke({
      fromCurrency: "USD",
      toCurrency: "EUR",
      amount: 100,
    });

    expect(out).toBe("100 USD = 92.5 EUR (UniRate latest rate)");
  });
});
