import { NextResponse } from "next/server";

import {
  getMarketKlines,
  getMarketQuote,
  getModelPredictionHistory,
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

    if (
      createdAt === null ||
      open === null ||
      close === null ||
      high === null ||
      low === null ||
      open <= 0 ||
      close <= 0 ||
      high <= 0 ||
      low <= 0
    ) {
      continue;
    }

    byTimestamp.set(createdAt, {
      ...raw,
      created_at: createdAt,
      open: String(open),
      close: String(close),
      high: String(high),
      low: String(low),
    });
  }

  return {
    ...market,
    candles: [...byTimestamp.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, candle]) => candle),
  };
}

function sanitizePredictionSeries(series: unknown): unknown {
  if (!isRecord(series)) return series;

  const rawPredictions = Array.isArray(series.predictions) ? series.predictions : [];
  const byTarget = new Map<number, JsonRecord>();

  for (const raw of rawPredictions) {
    if (!isRecord(raw)) continue;

    const targetTimestamp = normalizeEpochMs(raw.target_timestamp);
    const sourceTimestamp = normalizeEpochMs(raw.source_timestamp);
    const predictedPrice = finiteNumber(raw.predicted_price);
    const predictedLogReturn = finiteNumber(raw.predicted_log_return);
    const errorLogReturn = finiteNumber(raw.error_log_return);

    if (
      targetTimestamp === null ||
      sourceTimestamp === null ||
      predictedPrice === null ||
      predictedLogReturn === null ||
      errorLogReturn === null
    ) {
      continue;
    }

    byTarget.set(targetTimestamp, {
      ...raw,
      source_timestamp: sourceTimestamp,
      target_timestamp: targetTimestamp,
      predicted_price: predictedPrice,
      predicted_log_return: predictedLogReturn,
      error_log_return: errorLogReturn,
    });
  }

  return {
    ...series,
    predictions: [...byTarget.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, point]) => point),
  };
}

function sanitizePredictions(predictions: unknown): unknown {
  if (!isRecord(predictions) || !isRecord(predictions.models)) return predictions;

  const models = predictions.models;
  return {
    ...predictions,
    models: {
      ...models,
      xgboost: sanitizePredictionSeries(models.xgboost),
      lstm: sanitizePredictionSeries(models.lstm),
    },
  };
}

export async function GET() {
  const [quote, market, predictions] = await Promise.all([
    getMarketQuote("BTCUSDT"),
    getMarketKlines("BTCUSDT", "1hour", 720),
    getModelPredictionHistory(720),
  ]);

  return NextResponse.json(
    {
      generated_at: new Date().toISOString(),
      quote,
      market: sanitizeMarket(market),
      predictions: sanitizePredictions(predictions),
    },
    {
      headers: {
        "Cache-Control": "no-store, max-age=0",
      },
    },
  );
}
