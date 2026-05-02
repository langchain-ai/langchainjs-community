import { z } from "zod/v3";
import { StructuredTool } from "@langchain/core/tools";
import { InferInteropZodOutput } from "@langchain/core/utils/types";
import { getEnvironmentVariable } from "@langchain/core/utils/env";

/**
 * Configuration parameters shared by {@link UniRateAPIWrapper} and
 * {@link UniRateExchangeTool}. The API key falls back to the
 * `UNIRATE_API_KEY` environment variable.
 */
export interface UniRateAPIWrapperParams {
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
}

/**
 * Thin wrapper around the UniRate REST API
 * (https://unirateapi.com). Handles authentication, the required
 * `Accept: application/json` header, and HTTP error mapping. The
 * wrapper exposes the three endpoints that are free on every UniRate
 * tier: `/api/rates`, `/api/convert`, and `/api/currencies`.
 *
 * Setup:
 *
 * 1. Sign up at https://unirateapi.com to obtain an API key.
 * 2. Save the key into the `UNIRATE_API_KEY` environment variable, or
 *    pass `apiKey` when constructing the wrapper.
 *
 * @example
 * ```typescript
 * const unirate = new UniRateAPIWrapper();
 * const usdToEur = await unirate.convert("USD", "EUR", 100);
 * ```
 */
export class UniRateAPIWrapper {
  apiKey: string;

  baseUrl: string;

  timeout: number;

  constructor(fields: UniRateAPIWrapperParams = {}) {
    const apiKey = fields.apiKey ?? getEnvironmentVariable("UNIRATE_API_KEY");
    if (!apiKey) {
      throw new Error(
        `UniRate API key not set. Pass it in or set the environment variable named "UNIRATE_API_KEY".`
      );
    }
    this.apiKey = apiKey;
    this.baseUrl = fields.baseUrl ?? "https://api.unirateapi.com";
    this.timeout = fields.timeout ?? 30_000;
  }

  /** @ignore */
  protected async _request(
    path: string,
    params: Record<string, string | number | undefined>
  ): Promise<Record<string, unknown>> {
    const url = new URL(`${this.baseUrl}${path}`);
    url.searchParams.set("api_key", this.apiKey);
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) {
        url.searchParams.set(key, String(value));
      }
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    try {
      const response = await fetch(url, {
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      if (!response.ok) {
        if (response.status === 401) {
          throw new Error("Missing or invalid UniRate API key");
        }
        if (response.status === 403) {
          throw new Error("This UniRate endpoint requires a Pro subscription");
        }
        if (response.status === 404) {
          throw new Error("Currency not found or no data available");
        }
        if (response.status === 429) {
          throw new Error("UniRate API rate limit exceeded");
        }
        throw new Error(`UniRate API error: HTTP ${response.status}`);
      }
      return (await response.json()) as Record<string, unknown>;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Get the latest exchange rate for a currency pair, or all rates for a
   * base currency.
   *
   * @param fromCurrency ISO 4217 base currency code. Defaults to `"USD"`.
   * @param toCurrency ISO 4217 target currency code. If omitted, every
   *  supported target rate is returned.
   * @returns A single rate when `toCurrency` is provided, otherwise a
   *  mapping of every target currency code to its rate.
   */
  async getRate(
    fromCurrency = "USD",
    toCurrency?: string
  ): Promise<number | Record<string, number>> {
    const params: Record<string, string> = {
      from: fromCurrency.toUpperCase(),
    };
    if (toCurrency !== undefined) {
      params.to = toCurrency.toUpperCase();
    }
    const data = await this._request("/api/rates", params);
    if (toCurrency !== undefined) {
      return parseFloat(data.rate as string);
    }
    const rates = data.rates as Record<string, string>;
    return Object.fromEntries(
      Object.entries(rates).map(([code, rate]) => [code, parseFloat(rate)])
    );
  }

  /**
   * Convert an amount from one currency to another at the latest rate.
   */
  async convert(
    fromCurrency: string,
    toCurrency: string,
    amount = 1
  ): Promise<number> {
    const data = await this._request("/api/convert", {
      from: fromCurrency.toUpperCase(),
      to: toCurrency.toUpperCase(),
      amount,
    });
    return parseFloat(data.result as string);
  }

  /** Return every currency code supported by the UniRate API. */
  async getSupportedCurrencies(): Promise<string[]> {
    const data = await this._request("/api/currencies", {});
    return data.currencies as string[];
  }

  /** Convert an amount and format the result for an LLM agent. */
  async run(
    fromCurrency: string,
    toCurrency: string,
    amount = 1
  ): Promise<string> {
    const result = await this.convert(fromCurrency, toCurrency, amount);
    return `${amount} ${fromCurrency.toUpperCase()} = ${result} ${toCurrency.toUpperCase()} (UniRate latest rate)`;
  }
}

const UniRateExchangeSchema = z.object({
  fromCurrency: z
    .string()
    .describe(
      "The ISO 4217 source currency code (e.g. 'USD', 'EUR'). Crypto and commodity tickers (e.g. 'BTC', 'XAU') are also supported."
    ),
  toCurrency: z
    .string()
    .describe("The ISO 4217 target currency code to convert into."),
  amount: z
    .number()
    .default(1)
    .describe("The amount of the source currency to convert. Defaults to 1."),
});

/**
 * Constructor parameters for {@link UniRateExchangeTool}. Either pass an
 * existing {@link UniRateAPIWrapper} via `apiWrapper`, or pass the wrapper
 * fields directly and the tool will instantiate one for you.
 */
export interface UniRateExchangeToolParams extends UniRateAPIWrapperParams {
  apiWrapper?: UniRateAPIWrapper;
}

/**
 * A LangChain tool that converts an amount between currencies using the
 * UniRate API. UniRate provides 593+ fiat, crypto, and commodity exchange
 * rates with a permissive free tier. Latest rates and conversion are
 * free; historical endpoints require a Pro plan.
 *
 * Setup:
 *
 * 1. Sign up at https://unirateapi.com to get an API key.
 * 2. Save the key into the `UNIRATE_API_KEY` environment variable, or
 *    pass `apiKey` when constructing the tool.
 *
 * @example
 * ```typescript
 * import { UniRateExchangeTool } from "@langchain/community/tools/unirate";
 *
 * const tool = new UniRateExchangeTool();
 * const result = await tool.invoke({
 *   fromCurrency: "USD",
 *   toCurrency: "EUR",
 *   amount: 100,
 * });
 * console.log(result);
 * // 100 USD = 92.5 EUR (UniRate latest rate)
 * ```
 */
export class UniRateExchangeTool extends StructuredTool {
  static lc_name() {
    return "UniRateExchangeTool";
  }

  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      apiKey: "UNIRATE_API_KEY",
    };
  }

  name = "unirate_exchange";

  description =
    "A wrapper around the UniRate currency exchange API. Useful for converting an amount from one currency to another at the current market rate, or for looking up the latest exchange rate between two currencies. Input requires a source currency code, a target currency code, and an amount (defaults to 1).";

  schema: typeof UniRateExchangeSchema = UniRateExchangeSchema;

  apiWrapper: UniRateAPIWrapper;

  constructor(fields: UniRateExchangeToolParams = {}) {
    super(...arguments);
    this.apiWrapper =
      fields.apiWrapper ??
      new UniRateAPIWrapper({
        apiKey: fields.apiKey,
        baseUrl: fields.baseUrl,
        timeout: fields.timeout,
      });
  }

  /** @ignore */
  async _call(
    input: InferInteropZodOutput<typeof UniRateExchangeTool.prototype.schema>
  ): Promise<string> {
    return this.apiWrapper.run(
      input.fromCurrency,
      input.toCurrency,
      input.amount
    );
  }
}
