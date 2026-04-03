import { test, expect } from "vitest";
import * as url from "node:url";
import * as path from "node:path";
import * as fs from "node:fs/promises";
import { PDFLoader } from "../fs/pdf.js";

test("Test PDF loader from blob", async () => {
  const filePath = path.resolve(
    path.dirname(url.fileURLToPath(import.meta.url)),
    "./example_data/1706.03762.pdf"
  );
  const loader = new PDFLoader(
    new Blob([await fs.readFile(filePath)], {
      type: "application/pdf",
    })
  );
  const docs = await loader.load();

  expect(docs.length).toBe(15);
  expect(docs[0].pageContent).toContain("Attention Is All You Need");
  expect(docs[0].metadata).toMatchObject({
    blobType: "application/pdf",
    loc: {
      pageNumber: 1,
    },
    pdf: {
      totalPages: 15,
      version: expect.any(String),
    },
    source: "blob",
  });
  expect(docs[0].metadata.pdf.info).toBeTruthy();
});

test("Test PDF loader with custom pdf-parse v2 implementation", async () => {
  let destroyed = false;

  class MockPDFParse {
    constructor(_: { data: Uint8Array }) {}

    async getText() {
      return {
        pages: [
          { num: 1, text: "Mock page 1" },
          { num: 2, text: "Mock page 2" },
        ],
        total: 2,
      };
    }

    async getInfo() {
      return {
        info: { Title: "Mock PDF" },
        metadata: { format: "mock-v2" },
      };
    }

    async destroy() {
      destroyed = true;
    }
  }

  const loader = new PDFLoader(
    new Blob([Buffer.from("mock pdf")], {
      type: "application/pdf",
    }),
    {
      pdfjs: async () =>
        ({
          isV2: true as const,
          PDFParse: MockPDFParse,
        }) as any,
    }
  );

  const docs = await loader.load();

  expect(docs).toHaveLength(2);
  expect(docs[0].pageContent).toBe("Mock page 1");
  expect(docs[1].metadata).toMatchObject({
    blobType: "application/pdf",
    loc: {
      pageNumber: 2,
    },
    pdf: {
      info: { Title: "Mock PDF" },
      metadata: { format: "mock-v2" },
      totalPages: 2,
      version: "mock-v2",
    },
    source: "blob",
  });
  expect(destroyed).toBe(true);
});
