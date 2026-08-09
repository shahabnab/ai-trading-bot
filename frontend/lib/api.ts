export type HealthStatus = {
  status: string;
  environment: string;
  trading_mode: string;
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
