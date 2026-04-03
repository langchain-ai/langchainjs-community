import { Document } from "@langchain/core/documents";
import { BufferLoader } from "@langchain/classic/document_loaders/fs/buffer";

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

/**
 * A class that extends the `BufferLoader` class. It represents a document
 * loader that loads documents from PDF files.
 * @example
 * ```typescript
 * const loader = new PDFLoader("path/to/bitcoin.pdf");
 * const docs = await loader.load();
 * console.log({ docs });
 * ```
 */
export class PDFLoader extends BufferLoader {
  private splitPages: boolean;

  private pdfjs: typeof PDFLoaderImports;

  protected parsedItemSeparator: string;

  constructor(
    filePathOrBlob: string | Blob,
    {
      splitPages = true,
      pdfjs = PDFLoaderImports,
      parsedItemSeparator = "",
    } = {}
  ) {
    super(filePathOrBlob);
    this.splitPages = splitPages;
    this.pdfjs = pdfjs;
    this.parsedItemSeparator = parsedItemSeparator;
  }

  /**
   * A method that takes a `raw` buffer and `metadata` as parameters and
   * returns a promise that resolves to an array of `Document` instances. It
   * uses the `getDocument` function from the PDF.js library to load the PDF
   * from the buffer. It then iterates over each page of the PDF, retrieves
   * the text content using the `getTextContent` method, and joins the text
   * items to form the page content. It creates a new `Document` instance
   * for each page with the extracted text content and metadata, and adds it
   * to the `documents` array. If `splitPages` is `true`, it returns the
   * array of `Document` instances. Otherwise, if there are no documents, it
   * returns an empty array. Otherwise, it concatenates the page content of
   * all documents and creates a single `Document` instance with the
   * concatenated content.
   * @param raw The buffer to be parsed.
   * @param metadata The metadata of the document.
   * @returns A promise that resolves to an array of `Document` instances.
   */
  public async parse(
    raw: Buffer,
    metadata: Document["metadata"]
  ): Promise<Document[]> {
    const pdfjsResult = await this.pdfjs();

    if (pdfjsResult.isV2) {
      return this.parseWithV2(raw, metadata, pdfjsResult.PDFParse);
    }

    const { getDocument, version } = pdfjsResult;
    const pdf = await getDocument({
      data: new Uint8Array(raw.buffer),
      useWorkerFetch: false,
      isEvalSupported: false,
      useSystemFonts: true,
    }).promise;
    const meta = await pdf.getMetadata().catch(() => null);

    const documents: Document[] = [];

    for (let i = 1; i <= pdf.numPages; i += 1) {
      const page = await pdf.getPage(i);
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
            ...metadata,
            pdf: {
              version,
              info: meta?.info,
              metadata: meta?.metadata,
              totalPages: pdf.numPages,
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
          ...metadata,
          pdf: {
            version,
            info: meta?.info,
            metadata: meta?.metadata,
            totalPages: pdf.numPages,
          },
        },
      }),
    ];
  }

  private async parseWithV2(
    raw: Buffer,
    metadata: Document["metadata"],
    PDFParseClass: typeof import("pdf-parse").PDFParse
  ): Promise<Document[]> {
    const parser = new PDFParseClass({ data: new Uint8Array(raw.buffer) });

    try {
      const [textResult, infoResult] = await Promise.all([
        parser.getText(),
        parser.getInfo(),
      ]);

      const documents: Document[] = [];

      for (const page of textResult.pages) {
        if (!page.text || page.text.trim().length === 0) {
          continue;
        }

        documents.push(
          new Document({
            pageContent: page.text,
            metadata: {
              ...metadata,
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
            ...metadata,
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
    const { default: mod } =
      await import("pdf-parse/lib/pdf.js/v1.10.100/build/pdf.js");
    const { getDocument, version } = mod;
    return { isV2: false as const, getDocument, version };
  } catch (e) {
    console.error(e);
    throw new Error(
      "Failed to load pdf-parse. Please install pdf-parse v1 or v2, e.g. `npm install pdf-parse@^1` or `npm install pdf-parse@^2`."
    );
  }
}
