import { StructuredTool } from "@langchain/core/tools";
import { z } from "zod";

const SNAP_DEFAULT_POOL = "B8SyffZKt8LABKogWjH9rZcjY5PV2hyYRCbTxxbcrpFf";
const SNAP_DEFAULT_RPC = "https://api.mainnet-beta.solana.com";
const SNAP_PROTOCOL_FEE_BPS = 25;

const SNAP_MAINNET_POOLS = [
  { address: "B8SyffZKt8LABKogWjH9rZcjY5PV2hyYRCbTxxbcrpFf", denomination: "0.1 SOL", asset: "SOL" },
  { address: "5LeuHrPBgHNhgbCy996MEjcsBk5gNHhVj6AiuuCHZ8od", denomination: "1 USDC", asset: "USDC" },
  { address: "ECuHf8kgiWfmL3Q6id4WGBQWvuukhzqvF5vsxuPAKZBv", denomination: "10 USDC", asset: "USDC" },
];

/**
 * Parameters for creating SNAP tools.
 *
 * Requires `snap-solana-sdk` and `@solana/web3.js` as peer dependencies.
 * Install with: `npm install snap-solana-sdk @solana/web3.js`
 */
export interface SNAPToolParams {
  /** Solana RPC URL. Defaults to mainnet-beta public RPC. */
  rpcUrl?: string;
  /** Default pool address for deposits/withdrawals. */
  poolAddress?: string;
  /** Optional relayer URL for private withdrawals. */
  relayerUrl?: string;
  /**
   * Pre-configured SNAPClient instance. If provided, rpcUrl and wallet
   * are ignored and this client is used directly.
   */
  snapClient?: SNAPClientLike;
  /** Solana Connection instance. Required if snapClient is not provided. */
  connection?: unknown;
  /** Wallet or Keypair for signing. Required if snapClient is not provided. */
  wallet?: unknown;
}

interface SNAPClientLike {
  deposit(pool: unknown, amount?: number): Promise<{ depositIndex: number }>;
  withdraw(pool: unknown, note: unknown, recipient: unknown): Promise<string>;
  withdrawViaRelayer(
    pool: unknown,
    note: unknown,
    recipient: unknown,
    relayerUrl?: string
  ): Promise<{ txSignature: string; fee: number; recipientReceived?: number }>;
  getPoolInfo(
    pool: unknown
  ): Promise<{
    depositAmount: number;
    assetType: "sol" | "spl";
    tokenMint: { toBase58(): string } | null;
  }>;
}

function resolveClient(params: SNAPToolParams): SNAPClientLike {
  if (params.snapClient) return params.snapClient;

  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { SNAPClient } = require("snap-solana-sdk");
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { Connection, PublicKey } = require("@solana/web3.js");

  const connection =
    params.connection ??
    new Connection(params.rpcUrl ?? SNAP_DEFAULT_RPC, "confirmed");

  if (!params.wallet) {
    throw new Error(
      "SNAP tools require either a snapClient or a wallet parameter."
    );
  }

  const programId = new PublicKey(
    "9uePoqdgaXpqFLQM2ED1GGQrwSEiqe3r6tW1AfsnrrbS"
  );
  return new SNAPClient(connection, params.wallet, { programId });
}

function toPublicKey(address: string): unknown {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { PublicKey } = require("@solana/web3.js");
  return new PublicKey(address);
}

function stringify(value: Record<string, unknown>): string {
  return JSON.stringify(value, null, 2);
}

/**
 * Tool that lists available SNAP shielded payment pools on Solana mainnet.
 */
export class SNAPListPoolsTool extends StructuredTool {
  static lc_name() {
    return "SNAPListPoolsTool";
  }

  name = "snap_list_pools";

  description =
    "List available SNAP shielded payment pools on Solana mainnet. Returns pool addresses, denominations, and asset types.";

  schema = z.object({});

  protected async _call(): Promise<string> {
    return stringify({ pools: SNAP_MAINNET_POOLS });
  }
}

/**
 * Tool that deposits into a SNAP shielded pool.
 * Returns a serialized note that must be sent to the recipient off-chain.
 */
export class SNAPDepositTool extends StructuredTool {
  static lc_name() {
    return "SNAPDepositTool";
  }

  name = "snap_deposit";

  description =
    "Deposit into a SNAP shielded pool on Solana. Returns a secret note that must be sent to the recipient off-chain. The note is a bearer instrument — anyone with it can withdraw.";

  schema = z.object({
    poolAddress: z
      .string()
      .min(32)
      .optional()
      .describe("Pool address. Defaults to the 0.1 SOL pool."),
    amount: z
      .number()
      .positive()
      .optional()
      .describe("Deposit amount. Must match the pool denomination."),
  });

  private client: SNAPClientLike;

  private defaultPool: string;

  constructor(params: SNAPToolParams = {}) {
    super();
    this.client = resolveClient(params);
    this.defaultPool = params.poolAddress ?? SNAP_DEFAULT_POOL;
  }

  protected async _call(input: {
    poolAddress?: string;
    amount?: number;
  }): Promise<string> {
    const pool = toPublicKey(input.poolAddress ?? this.defaultPool);
    const deposit = await this.client.deposit(pool, input.amount);
    const poolInfo = await this.client.getPoolInfo(pool);

    let noteSerialized: string | null = null;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { SNAPClient } = require("snap-solana-sdk");
      noteSerialized = SNAPClient.serializeNote(deposit);
    } catch {
      // SDK may not support serialization in all contexts
    }

    return stringify({
      success: true,
      depositIndex: deposit.depositIndex,
      poolAddress: input.poolAddress ?? this.defaultPool,
      amount: poolInfo.depositAmount,
      assetType: poolInfo.assetType,
      note: noteSerialized,
    });
  }
}

/**
 * Tool that withdraws from a SNAP shielded pool using a ZK proof.
 * Supports direct withdrawal or relayed withdrawal for enhanced privacy.
 */
export class SNAPWithdrawTool extends StructuredTool {
  static lc_name() {
    return "SNAPWithdrawTool";
  }

  name = "snap_withdraw";

  description =
    "Withdraw from a SNAP shielded pool to a recipient address using a zero-knowledge proof. Provide relayerUrl for relayed withdrawal (recipient doesn't pay gas).";

  schema = z.object({
    poolAddress: z
      .string()
      .min(32)
      .optional()
      .describe("Pool address. Defaults to the 0.1 SOL pool."),
    note: z.any().describe("The secret note from the deposit (string or object)."),
    recipient: z
      .string()
      .min(32)
      .describe("Recipient Solana address."),
    relayerUrl: z
      .string()
      .url()
      .optional()
      .describe("Optional relayer URL for private withdrawal."),
  });

  private client: SNAPClientLike;

  private defaultPool: string;

  private defaultRelayerUrl?: string;

  constructor(params: SNAPToolParams = {}) {
    super();
    this.client = resolveClient(params);
    this.defaultPool = params.poolAddress ?? SNAP_DEFAULT_POOL;
    this.defaultRelayerUrl = params.relayerUrl;
  }

  protected async _call(input: {
    poolAddress?: string;
    note: unknown;
    recipient: string;
    relayerUrl?: string;
  }): Promise<string> {
    const pool = toPublicKey(input.poolAddress ?? this.defaultPool);
    const recipient = toPublicKey(input.recipient);
    const relayerUrl = input.relayerUrl ?? this.defaultRelayerUrl;

    let note = input.note;
    if (typeof note === "string") {
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { SNAPClient } = require("snap-solana-sdk");
        note = SNAPClient.deserializeNote(note);
      } catch {
        // pass through as-is
      }
    }

    if (relayerUrl) {
      const result = await this.client.withdrawViaRelayer(
        pool,
        note,
        recipient,
        relayerUrl
      );
      return stringify({
        success: true,
        transaction: result.txSignature,
        fee: result.fee,
        recipientReceived: result.recipientReceived,
        relayed: true,
      });
    }

    const tx = await this.client.withdraw(pool, note, recipient);
    return stringify({ success: true, transaction: tx, relayed: false });
  }
}

/**
 * Tool that estimates withdrawal fees for a SNAP pool.
 */
export class SNAPEstimateFeeTool extends StructuredTool {
  static lc_name() {
    return "SNAPEstimateFeeTool";
  }

  name = "snap_estimate_fee";

  description =
    "Estimate the protocol fee for withdrawing from a SNAP shielded pool. Returns fee breakdown including protocol fee percentage.";

  schema = z.object({
    poolAddress: z
      .string()
      .min(32)
      .optional()
      .describe("Pool address. Defaults to the 0.1 SOL pool."),
  });

  private client: SNAPClientLike;

  private defaultPool: string;

  constructor(params: SNAPToolParams = {}) {
    super();
    this.client = resolveClient(params);
    this.defaultPool = params.poolAddress ?? SNAP_DEFAULT_POOL;
  }

  protected async _call(input: { poolAddress?: string }): Promise<string> {
    const pool = toPublicKey(input.poolAddress ?? this.defaultPool);
    const poolInfo = await this.client.getPoolInfo(pool);
    const protocolFee =
      (poolInfo.depositAmount * SNAP_PROTOCOL_FEE_BPS) / 10000;

    return stringify({
      poolAddress: input.poolAddress ?? this.defaultPool,
      denomination: poolInfo.depositAmount,
      assetType: poolInfo.assetType,
      protocolFeeBps: SNAP_PROTOCOL_FEE_BPS,
      protocolFee,
      recipientReceives: poolInfo.depositAmount - protocolFee,
    });
  }
}

/**
 * Create all four SNAP tools with shared configuration.
 *
 * @example
 * ```typescript
 * import { createSNAPTools } from "@langchain/community/tools/snap_private_payments";
 * import { Connection, Keypair } from "@solana/web3.js";
 *
 * const tools = createSNAPTools({
 *   connection: new Connection("https://api.mainnet-beta.solana.com"),
 *   wallet: Keypair.generate(),
 * });
 * ```
 */
export function createSNAPTools(params: SNAPToolParams = {}) {
  return [
    new SNAPListPoolsTool(),
    new SNAPDepositTool(params),
    new SNAPWithdrawTool(params),
    new SNAPEstimateFeeTool(params),
  ];
}
