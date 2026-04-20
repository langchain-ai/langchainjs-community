import { test, vi, expect, afterEach } from "vitest";
import { VoyageEmbeddings } from "../voyage.js";

const FAKE_API_KEY = "voyage-test-key";

function makeFetchMock(body: unknown, status = 200) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  }) as unknown as typeof fetch;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

test("embedDocuments surfaces Voyage API error with detail field (issue #10031)", async () => {
  vi.stubGlobal(
    "fetch",
    makeFetchMock({ detail: "Provided API key is invalid." }, 401)
  );

  const embeddings = new VoyageEmbeddings({ apiKey: FAKE_API_KEY });

  await expect(embeddings.embedDocuments(["Hello"])).rejects.toThrow(
    "Voyage AI API error (HTTP 401): Provided API key is invalid."
  );
});

test("embedQuery surfaces Voyage API error with detail field", async () => {
  vi.stubGlobal(
    "fetch",
    makeFetchMock({ detail: "Provided API key is invalid." }, 401)
  );

  const embeddings = new VoyageEmbeddings({ apiKey: FAKE_API_KEY });

  await expect(embeddings.embedQuery("Hello")).rejects.toThrow(
    "Voyage AI API error (HTTP 401): Provided API key is invalid."
  );
});

test("embedDocuments surfaces generic Voyage API error via error.message field", async () => {
  // 400 is in AsyncCaller's STATUS_NO_RETRY list, so it won't be retried.
  vi.stubGlobal(
    "fetch",
    makeFetchMock(
      { error: { message: "Input exceeds maximum token length" } },
      400
    )
  );

  const embeddings = new VoyageEmbeddings({ apiKey: FAKE_API_KEY });

  await expect(embeddings.embedDocuments(["Hello"])).rejects.toThrow(
    "Voyage AI API error (HTTP 400): Input exceeds maximum token length"
  );
});

test("embedDocuments succeeds on a valid 200 response", async () => {
  const fakeEmbedding = [0.1, 0.2, 0.3];
  vi.stubGlobal(
    "fetch",
    makeFetchMock({
      data: [{ embedding: fakeEmbedding }],
      model: "voyage-01",
      usage: { total_tokens: 3 },
    })
  );

  const embeddings = new VoyageEmbeddings({ apiKey: FAKE_API_KEY });
  const result = await embeddings.embedDocuments(["Hello"]);

  expect(result).toEqual([fakeEmbedding]);
});
