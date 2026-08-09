export type HealthStatus = {
  status: string;
  environment: string;
  trading_mode: string;
  coinex_configured?: boolean;
  paper_engine?: boolean;
};

export type MarketQuote = {
  exchange: string;
  read_only: boolean;
  symbol: string;
  last: string;
  open: string;
  high: string;
  low: string;
  volume: string;
  value: string;
};

export type PaperPosition = {
  symbol: string;
  quantity: string;
  avg_entry_price: string;
  last_price: string;
  market_value_usdt: string;
  unrealized_pnl_usdt: string;
  price_source: string;
};

export type PaperPortfolio = {
  currency: string;
  initial_cash_usdt: string;
  cash_usdt: string;
  positions_value_usdt: string;
  portfolio_value_usdt: string;
  unrealized_pnl_usdt: string;
  total_pnl_usdt: string;
  total_return: string;
  positions: PaperPosition[];
};

export type PaperDecision = {
  id: number;
  created_at: string;
  symbol: string;
  signal: string;
  confidence: number | null;
  approved: number;
  reason: string;
  model_version: string;
  strategy_version: string;
  market_price: string | null;
};

export type PaperTrade = {
  id: number;
  created_at: string;
  symbol: string;
  side: string;
  quantity: string;
  market_price: string;
  execution_price: string;
  gross_value_usdt: string;
  fee_usdt: string;
  realized_pnl_usdt: string;
  model_version: string;
  strategy_version: string;
  confidence: number | null;
};

export type PaperPerformance = {
  trade_count: number;
  decision_count: number;
  closed_trades: number;
  winning_trades: number;
  win_rate: number | null;
  total_fees_usdt: string;
  realized_pnl_usdt: string;
};

const API_BASE_URL = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";

async function safeGet<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

export function getHealth(): Promise<HealthStatus | null> {
  return safeGet<HealthStatus>("/health");
}

export function getMarketQuote(symbol: string): Promise<MarketQuote | null> {
  return safeGet<MarketQuote>(`/api/market/${encodeURIComponent(symbol)}`);
}

export function getPaperPortfolio(): Promise<PaperPortfolio | null> {
  return safeGet<PaperPortfolio>("/api/paper/portfolio");
}

export async function getPaperDecisions(limit = 8): Promise<PaperDecision[]> {
  const result = await safeGet<{ decisions: PaperDecision[] }>(`/api/paper/decisions?limit=${limit}`);
  return result?.decisions ?? [];
}

export async function getPaperTrades(limit = 8): Promise<PaperTrade[]> {
  const result = await safeGet<{ trades: PaperTrade[] }>(`/api/paper/trades?limit=${limit}`);
  return result?.trades ?? [];
}

export function getPaperPerformance(): Promise<PaperPerformance | null> {
  return safeGet<PaperPerformance>("/api/paper/performance");
}
