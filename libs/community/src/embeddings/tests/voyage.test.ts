import { afterEach, describe, expect, test, vi } from "vitest";
import { VoyageEmbeddings } from "../voyage";

describe("VoyageEmbeddings", () => {
  const fetchMock = vi.fn<typeof fetch>();

  afterEach(() => {
    vi.unstubAllGlobals();
    fetchMock.mockReset();
  });

  test("uses basePath provided in constructor", async () => {
    fetchMock.mockResolvedValue({
      json: async () => ({
        data: [{ embedding: [0.1, 0.2, 0.3] }],
      }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);

    const embeddings = new VoyageEmbeddings({
      apiKey: "test-key",
      basePath: "https://ai.mongodb.com/v1",
    });

    expect(embeddings.apiUrl).toBe("https://ai.mongodb.com/v1/embeddings");

    await embeddings.embedQuery("Hello world");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "https://ai.mongodb.com/v1/embeddings"
    );
  });
});
