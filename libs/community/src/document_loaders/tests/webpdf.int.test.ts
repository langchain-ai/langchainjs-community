// oxlint-disable typescript/no-explicit-any
import { test, expect } from "vitest";
import * as url from "node:url";
import * as path from "node:path";
import * as fs from "node:fs/promises";
import { WebPDFLoader } from "../web/pdf.js";

test("Test Web PDF loader from blob", async () => {
  const filePath = path.resolve(
    path.dirname(url.fileURLToPath(import.meta.url)),
    "./example_data/1706.03762.pdf"
  );
  const loader = new WebPDFLoader(
    new Blob([await fs.readFile(filePath)], {
      type: "application/pdf",
    })
  );
  const docs = await loader.load();

  expect(docs.length).toBe(15);
  expect(docs[0].pageContent).toContain("Attention Is All You Need");
  expect(docs[0].metadata).toMatchObject({
    loc: {
      pageNumber: 1,
    },
    pdf: {
      totalPages: 15,
      version: expect.any(String),
    },
  });
  expect(docs[0].metadata.pdf.info).toBeTruthy();
});

test("Test Web PDF loader with custom pdf-parse v1 implementation", async () => {
  const loader = new WebPDFLoader(new Blob([Buffer.from("mock pdf")]), {
    pdfjs: async () =>
      ({
        isV2: false as const,
        version: "1.10.100",
        getDocument: () =>
          ({
            promise: Promise.resolve({
              numPages: 1,
              getMetadata: async () => ({
                info: { Title: "Mock PDF" },
                metadata: null,
              }),
              getPage: async () => ({
                getTextContent: async () => ({
                  items: [
                    {
                      str: "Mock page 1",
                      transform: [0, 0, 0, 0, 0, 0],
                    },
                  ],
                }),
              }),
            }),
          }) as any,
      }) as any,
  });
  const docs = await loader.load();

  expect(docs).toHaveLength(1);
  expect(docs[0].pageContent).toBe("Mock page 1");
  expect(docs[0].metadata).toMatchObject({
    loc: {
      pageNumber: 1,
    },
    pdf: {
      info: { Title: "Mock PDF" },
      metadata: null,
      totalPages: 1,
      version: "1.10.100",
    },
  });
});

test("Test Web PDF loader lines", async () => {
  const filePath = path.resolve(
    path.dirname(url.fileURLToPath(import.meta.url)),
    "./example_data/Jacob_Lee_Resume_2023.pdf"
  );
  const loader = new WebPDFLoader(
    new Blob([await fs.readFile(filePath)], {
      type: "application/pdf",
    }),
    { splitPages: false }
  );
  const docs = await loader.load();

  expect(docs.length).toBe(1);
  expect(docs[0].pageContent.split("\n").length).toBeLessThan(100);
});
