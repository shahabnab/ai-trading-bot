import { NextResponse } from "next/server";

import {
  getMarketKlines,
  getMarketQuote,
  getModelPredictionHistory,
} from "../../../lib/api";

export const dynamic = "force-dynamic";

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
      market,
      predictions,
    },
    {
      headers: {
        "Cache-Control": "no-store, max-age=0",
      },
    },
  );
}
