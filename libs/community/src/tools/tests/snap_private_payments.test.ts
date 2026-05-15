import { test, expect } from "@jest/globals";
import { SNAPListPoolsTool, SNAPDepositTool, SNAPWithdrawTool, SNAPEstimateFeeTool, createSNAPTools } from "../snap_private_payments.js";

test("SNAPListPoolsTool returns mainnet pools", async () => {
  const tool = new SNAPListPoolsTool();
  const result = await tool.invoke({});
  const parsed = JSON.parse(result);
  expect(parsed.pools).toHaveLength(3);
  expect(parsed.pools[0].asset).toBe("SOL");
  expect(parsed.pools[0].denomination).toBe("0.1 SOL");
  expect(parsed.pools[1].asset).toBe("USDC");
  expect(parsed.pools[2].denomination).toBe("10 USDC");
});

test("createSNAPTools returns 4 tools", () => {
  const mockClient = {
    deposit: async () => ({ depositIndex: 0 }),
    withdraw: async () => "txhash",
    withdrawViaRelayer: async () => ({
      txSignature: "txhash",
      fee: 0.00025,
      recipientReceived: 0.09975,
    }),
    getPoolInfo: async () => ({
      depositAmount: 0.1,
      assetType: "sol" as const,
      tokenMint: null,
    }),
  };

  const tools = createSNAPTools({ snapClient: mockClient });
  expect(tools).toHaveLength(4);
  expect(tools.map((t) => t.name)).toEqual([
    "snap_list_pools",
    "snap_deposit",
    "snap_withdraw",
    "snap_estimate_fee",
  ]);
});

test("SNAPEstimateFeeTool calculates protocol fee", async () => {
  const mockClient = {
    deposit: async () => ({ depositIndex: 0 }),
    withdraw: async () => "txhash",
    withdrawViaRelayer: async () => ({
      txSignature: "txhash",
      fee: 0,
      recipientReceived: 0,
    }),
    getPoolInfo: async () => ({
      depositAmount: 0.1,
      assetType: "sol" as const,
      tokenMint: null,
    }),
  };

  const tool = new SNAPEstimateFeeTool({ snapClient: mockClient });
  const result = await tool.invoke({});
  const parsed = JSON.parse(result);
  expect(parsed.protocolFeeBps).toBe(25);
  expect(parsed.protocolFee).toBe(0.00025);
  expect(parsed.recipientReceives).toBe(0.09975);
});
