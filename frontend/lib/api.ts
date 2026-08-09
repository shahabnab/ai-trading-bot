export type HealthStatus = {
  status: string;
  environment: string;
  trading_mode: string;
  coinex_configured?: boolean;
};

export type CoinExBalance = {
  ccy: string;
  available: string;
  frozen: string;
  total: string;
};

export type CoinExBalancesResponse = {
  exchange: string;
  account: string;
  read_only: boolean;
  balances: CoinExBalance[];
};

const API_BASE_URL = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";

export async function getHealth(): Promise<HealthStatus | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as HealthStatus;
  } catch {
    return null;
  }
}

export async function getCoinExBalances(): Promise<CoinExBalancesResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/coinex/balances`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as CoinExBalancesResponse;
  } catch {
    return null;
  }
}
