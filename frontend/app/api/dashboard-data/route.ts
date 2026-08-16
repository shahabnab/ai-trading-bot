import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

import {
  getMarketKlines,
  getMarketQuote,
  getPaperModels,
  getPaperModelTrades,
} from "../../../lib/api";

export const dynamic = "force-dynamic";

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeEpochMs(value: unknown): number | null {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  return Math.trunc(n < 10_000_000_000 ? n * 1000 : n);
}

function finiteNumber(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function sanitizeMarket(market: unknown): unknown {
  if (!isRecord(market)) return market;
  const rawCandles = Array.isArray(market.candles) ? market.candles : [];
  const byTimestamp = new Map<number, JsonRecord>();

  for (const raw of rawCandles) {
    if (!isRecord(raw)) continue;
    const createdAt = normalizeEpochMs(raw.created_at);
    const open = finiteNumber(raw.open);
    const close = finiteNumber(raw.close);
    const high = finiteNumber(raw.high);
    const low = finiteNumber(raw.low);
    if (createdAt === null || open === null || close === null || high === null || low === null || open <= 0 || close <= 0 || high <= 0 || low <= 0) continue;
    byTimestamp.set(createdAt, { ...raw, created_at: createdAt, open: String(open), close: String(close), high: String(high), low: String(low) });
  }

  return {
    ...market,
    candles: [...byTimestamp.entries()].sort((a, b) => a[0] - b[0]).map(([, candle]) => candle),
  };
}

function repoRoot(): string {
  return path.resolve(process.cwd(), "..");
}

function loadForwardRecords(limit = 720): JsonRecord[] {
  const file = path.join(repoRoot(), "state", "forward_v3", "predictions.jsonl");
  if (!existsSync(file)) return [];
  try {
    const rows: JsonRecord[] = [];
    for (const line of readFileSync(file, "utf8").split(/\r?\n/)) {
      if (!line.trim()) continue;
      try {
        const parsed = JSON.parse(line) as unknown;
        if (isRecord(parsed)) rows.push(parsed);
      } catch {
        // Ignore a partially-written/corrupt telemetry line and keep the dashboard alive.
      }
    }
    rows.sort((a, b) => Number(a.feature_timestamp ?? 0) - Number(b.feature_timestamp ?? 0));
    return rows.slice(-limit * 2);
  } catch {
    return [];
  }
}

function artifactReady(modelId: string): boolean {
  const root = path.join(repoRoot(), "artifacts", "ml", "forward_deployment", "v3-paper", modelId);
  return ["model.keras", "manifest.json", "standardizer.json"].every((name) => existsSync(path.join(root, name)));
}

export async function GET() {
  const [quote, market, paperModels] = await Promise.all([
    getMarketQuote("BTCUSDT"),
    getMarketKlines("BTCUSDT", "1hour", 720),
    getPaperModels(),
  ]);

  const records = loadForwardRecords(720);
  const models = paperModels?.models ?? [];
  const tradesByModel = await Promise.all(models.map((model) => getPaperModelTrades(model.model_id, 1000)));
  const forwardModels = models.map((model, index) => ({
    ...model,
    artifact_ready: model.driver === "frozen_v3" ? artifactReady(model.model_id) : true,
    history: model.driver === "frozen_v3" ? records.filter((row) => row.model_id === model.model_id).slice(-720) : [],
    trades: tradesByModel[index] ?? [],
  }));

  return NextResponse.json(
    {
      generated_at: new Date().toISOString(),
      quote,
      market: sanitizeMarket(market),
      forward: {
        mode: "prospective_forward_paper",
        paper_only: true,
        real_orders_enabled: false,
        models: forwardModels,
      },
    },
    { headers: { "Cache-Control": "no-store, max-age=0" } },
  );
}
