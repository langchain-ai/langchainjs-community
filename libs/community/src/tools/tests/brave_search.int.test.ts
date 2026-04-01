import { test } from "vitest";
import { BraveSearch } from "../brave_search.js";

test.skip("BraveSearchTool", async () => {
  const tool = new BraveSearch();
// @ts-expect-error unused var
  const result = await tool.invoke("What is Langchain?");

  // console.log({ result });
});
