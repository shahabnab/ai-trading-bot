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

export type MarketCandle = {
  symbol: string;
  created_at: number | string | null;
  open: string;
  close: string;
  high: string;
  low: string;
  volume: string;
  value: string;
};

export type MarketKlinesResponse = {
  exchange: string;
  read_only: boolean;
  symbol: string;
  period: string;
  candles: MarketCandle[];
};

export type ModelPredictionPoint = {
  source_timestamp: number;
  target_timestamp: number;
  fold: number;
  reference_price: number;
  predicted_price: number;
  actual_price: number;
  predicted_log_return: number;
  actual_log_return: number;
  predicted_simple_return: number;
  actual_simple_return: number;
  error_log_return: number;
  direction_correct: boolean;
};

export type ModelPredictionMetrics = {
  forecast?: {
    direction_accuracy?: number;
    rmse_log_return?: number;
    mae_log_return?: number;
    correlation?: number;
  };
  strategy?: {
    cumulative_return?: number;
    annualized_return?: number;
    sharpe?: number;
    sortino?: number;
    max_drawdown?: number;
    turnover?: number;
    trade_count?: number;
  };
  fold_count?: number;
  oos_prediction_count?: number;
  feature_version?: string;
};

export type ModelPredictionSeries = {
  run_id: string;
  started_at?: string | null;
  finished_at?: string | null;
  mode: "walk_forward_oos" | "forward_live" | string;
  metrics: ModelPredictionMetrics;
  predictions: ModelPredictionPoint[];
};

export type ModelPredictionHistory = {
  generated_at: string;
  source: string;
  models: Partial<Record<"xgboost" | "lstm", ModelPredictionSeries>>;
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
  ledger_currency?: string;
  initial_cash_usdt: string;
  cash_usdt: string;
  positions_value_usdt: string;
  portfolio_value_usdt: string;
  unrealized_pnl_usdt: string;
  total_pnl_usdt: string;
  total_return: string;
  positions: PaperPosition[];
  fx_note?: string;
};

export type PaperDecision = {
  id: number;
  model_id?: string;
  created_at: string;
  symbol: string;
  signal: string;
  confidence: number | null;
  approved: number;
  reason: string;
  model_version?: string;
  strategy_version: string;
  market_price: string | null;
};

export type PaperTrade = {
  id: number;
  model_id?: string;
  created_at: string;
  symbol: string;
  side: string;
  quantity: string;
  market_price: string;
  execution_price: string;
  gross_value_usdt: string;
  fee_usdt: string;
  realized_pnl_usdt: string;
  model_version?: string;
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

export type PaperModelAccount = {
  model_id: string;
  display_name: string;
  role: "paper_strategy" | "research_control" | string;
  target_bps: number;
  horizon_hours: number;
  feature_set: string;
  research_auc: number;
  research_median_auc: number;
  research_sharpe_25bps: number;
  research_return_25bps: number;
  research_trades: number;
  research_gate_passed: boolean;
  portfolio: PaperPortfolio;
  performance: PaperPerformance;
  latest_decision: PaperDecision | null;
  live_status: "paper_running" | "waiting_for_signal" | string;
};

export type PaperModelsResponse = {
  mode: string;
  starting_capital_eur_equiv_per_model: string;
  real_orders_enabled: boolean;
  models: PaperModelAccount[];
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

export function getMarketKlines(
  symbol: string,
  period = "1hour",
  limit = 720,
): Promise<MarketKlinesResponse | null> {
  return safeGet<MarketKlinesResponse>(
    `/api/market/${encodeURIComponent(symbol)}/klines?period=${encodeURIComponent(period)}&limit=${limit}`,
  );
}

export function getModelPredictionHistory(limit = 720): Promise<ModelPredictionHistory | null> {
  return safeGet<ModelPredictionHistory>(`/api/ml/predictions?limit=${limit}`);
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

export function getPaperModels(): Promise<PaperModelsResponse | null> {
  return safeGet<PaperModelsResponse>("/api/paper/models");
}

export async function getPaperModelTrades(modelId: string, limit = 50): Promise<PaperTrade[]> {
  const result = await safeGet<{ trades: PaperTrade[] }>(
    `/api/paper/models/${encodeURIComponent(modelId)}/trades?limit=${limit}`,
  );
  return result?.trades ?? [];
}

export async function getPaperModelDecisions(modelId: string, limit = 20): Promise<PaperDecision[]> {
  const result = await safeGet<{ decisions: PaperDecision[] }>(
    `/api/paper/models/${encodeURIComponent(modelId)}/decisions?limit=${limit}`,
  );
  return result?.decisions ?? [];
}
