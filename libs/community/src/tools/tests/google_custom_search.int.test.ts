import { test } from "vitest";
import { GoogleCustomSearch } from "../google_custom_search.js";

test.skip("GoogleCustomSearchTool", async () => {
  const tool = new GoogleCustomSearch();
// @ts-expect-error unused var
  const result = await tool.invoke("What is Langchain?");

  // console.log({ result });
});
