import { Document } from "@langchain/core/documents";
import { BaseDocumentLoader } from "@langchain/core/document_loaders/base";

type PDFLoaderV1Imports = {
  isV2: false;
  getDocument: typeof import("pdf-parse/lib/pdf.js/v1.10.100/build/pdf.js").getDocument;
  version: typeof import("pdf-parse/lib/pdf.js/v1.10.100/build/pdf.js").version;
};

type PDFLoaderV2Imports = {
  isV2: true;
  PDFParse: typeof import("pdf-parse").PDFParse;
};

type PDFLoaderImportsResult = PDFLoaderV1Imports | PDFLoaderV2Imports;

const PDF_PARSE_V1_IMPORT_PATH = "pdf-parse/lib/pdf.js/v1.10.100/build/pdf.js";

/**
 * A document loader for loading data from PDFs.
 * @example
 * ```typescript
 * const loader = new WebPDFLoader(new Blob());
 * const docs = await loader.load();
 * console.log({ docs });
 * ```
 */
export class WebPDFLoader extends BaseDocumentLoader {
  protected blob: Blob;

  protected splitPages = true;

  private pdfjs: typeof PDFLoaderImports;

  protected parsedItemSeparator: string;

  constructor(
    blob: Blob,
    {
      splitPages = true,
      pdfjs = PDFLoaderImports,
      parsedItemSeparator = "",
    } = {}
  ) {
    super();
    this.blob = blob;
    this.splitPages = splitPages ?? this.splitPages;
    this.pdfjs = pdfjs;
    this.parsedItemSeparator = parsedItemSeparator;
  }

  /**
   * Loads the contents of the PDF as documents.
   * @returns An array of Documents representing the retrieved data.
   */
  async load(): Promise<Document[]> {
    const raw = new Uint8Array(await this.blob.arrayBuffer());
    const pdfjsResult = await this.pdfjs();

    if (pdfjsResult.isV2) {
      return this.parseWithV2(raw, pdfjsResult.PDFParse);
    }

    const { getDocument, version } = pdfjsResult;
    const parsedPdf = await getDocument({
      data: raw,
      useWorkerFetch: false,
      isEvalSupported: false,
      useSystemFonts: true,
    }).promise;
    const meta = await parsedPdf.getMetadata().catch(() => null);

    const documents: Document[] = [];

    for (let i = 1; i <= parsedPdf.numPages; i += 1) {
      const page = await parsedPdf.getPage(i);
      const content = await page.getTextContent();

      if (content.items.length === 0) {
        continue;
      }

      // Eliminate excessive newlines
      // Source: https://github.com/albertcui/pdf-parse/blob/7086fc1cc9058545cdf41dd0646d6ae5832c7107/lib/pdf-parse.js#L16
      let lastY;
      const textItems = [];
      for (const item of content.items) {
        if ("str" in item) {
          if (lastY === item.transform[5] || !lastY) {
            textItems.push(item.str);
          } else {
            textItems.push(`\n${item.str}`);
          }
          lastY = item.transform[5];
        }
      }
      const text = textItems.join(this.parsedItemSeparator);

      documents.push(
        new Document({
          pageContent: text,
          metadata: {
            pdf: {
              version,
              info: meta?.info,
              metadata: meta?.metadata,
              totalPages: parsedPdf.numPages,
            },
            loc: {
              pageNumber: i,
            },
          },
        })
      );
    }

    if (this.splitPages) {
      return documents;
    }

    if (documents.length === 0) {
      return [];
    }

    return [
      new Document({
        pageContent: documents.map((doc) => doc.pageContent).join("\n\n"),
        metadata: {
          pdf: {
            version,
            info: meta?.info,
            metadata: meta?.metadata,
            totalPages: parsedPdf.numPages,
          },
        },
      }),
    ];
  }

  private async parseWithV2(
    raw: Uint8Array,
    PDFParseClass: typeof import("pdf-parse").PDFParse
  ): Promise<Document[]> {
    const parser = new PDFParseClass({ data: raw });

    try {
      const textResult = await parser.getText();
      const infoResult = await parser.getInfo();

      const documents: Document[] = [];

      for (const page of textResult.pages) {
        if (!page.text || page.text.trim().length === 0) {
          continue;
        }

        documents.push(
          new Document({
            pageContent: page.text,
            metadata: {
              pdf: {
                version: infoResult.metadata?.format || "unknown",
                info: infoResult.info,
                metadata: infoResult.metadata,
                totalPages: textResult.total,
              },
              loc: {
                pageNumber: page.num,
              },
            },
          })
        );
      }

      if (this.splitPages) {
        return documents;
      }

      if (documents.length === 0) {
        return [];
      }

      return [
        new Document({
          pageContent: documents.map((doc) => doc.pageContent).join("\n\n"),
          metadata: {
            pdf: {
              version: infoResult.metadata?.format || "unknown",
              info: infoResult.info,
              metadata: infoResult.metadata,
              totalPages: textResult.total,
            },
          },
        }),
      ];
    } finally {
      await parser.destroy();
    }
  }
}

async function PDFLoaderImports(): Promise<PDFLoaderImportsResult> {
  try {
    const pdfParseModule = await import("pdf-parse");
    if ("PDFParse" in pdfParseModule) {
      return { isV2: true as const, PDFParse: pdfParseModule.PDFParse };
    }
  } catch {
    // Fall back to the pdf-parse v1 import path below.
  }

  try {
    const { default: mod } = await import(
      /* @vite-ignore */ PDF_PARSE_V1_IMPORT_PATH
    );
    const { getDocument, version } = mod;
    return { isV2: false as const, getDocument, version };
  } catch (e) {
    console.error(e);
    throw new Error(
      "Failed to load pdf-parse. Please install pdf-parse v1 or v2, e.g. `npm install pdf-parse@^1` or `npm install pdf-parse@^2`."
    );
  }
}
