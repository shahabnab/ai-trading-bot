import LivePredictionTerminal from "../components/LivePredictionTerminal";
import ModelComparisonTabs, { type PaperDecision, type PaperModel, type PaperTrade } from "../components/ModelComparisonTabs";
import ShortTermDiagnostics from "../components/ShortTermDiagnostics";
import SystemOverview from "../components/SystemOverview";

const API_BASE = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";
const BRAIN_IDS = ["trader-brain-v1", "trader-brain-bandit-v1"] as const;

async function safeGet<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

async function loadModels(): Promise<PaperModel[]> {
  const base = await safeGet<{ models: PaperModel[] }>("/api/paper/models");
  const models = [...(base?.models ?? [])];
  const known = new Set(models.map((model) => model.model_id));
  for (const id of BRAIN_IDS) {
    if (known.has(id)) continue;
    const model = await safeGet<PaperModel>(`/api/paper/models/${id}`);
    if (model) models.push(model);
  }
  return models;
}

export default async function Home() {
  const [health, quote, models] = await Promise.all([
    safeGet<{ status: string; trading_mode: string }>("/health"),
    safeGet<{ last: string; high: string; low: string; volume: string }>("/api/market/BTCUSDT"),
    loadModels(),
  ]);
  const tradeEntries = await Promise.all(models.map(async (model) => {
    const value = await safeGet<{ trades: PaperTrade[] }>(`/api/paper/models/${model.model_id}/trades?limit=50`);
    return [model.model_id, value?.trades ?? []] as const;
  }));
  const decisionEntries = await Promise.all(models.map(async (model) => {
    const value = await safeGet<{ decisions: PaperDecision[] }>(`/api/paper/models/${model.model_id}/decisions?limit=20`);
    return [model.model_id, value?.decisions ?? []] as const;
  }));
  const tradesByModel = Object.fromEntries(tradeEntries) as Record<string, PaperTrade[]>;
  const decisionsByModel = Object.fromEntries(decisionEntries) as Record<string, PaperDecision[]>;

  const shortTermModels = models.filter((model) => model.driver === "short_term");
  const coreModels = models.filter((model) => model.driver !== "short_term");

  const panel: React.CSSProperties = { border: "1px solid rgba(148,163,184,.22)", borderRadius: 14, padding: 18, background: "rgba(15,23,42,.52)" };

  return (
    <main className="app-shell" style={{ display: "grid", gap: 20 }}>
      <section className="hero">
        <div>
          <p className="eyebrow">AI TRADING OBSERVATORY</p>
          <h1>Every algorithm, one scoreboard. <span>Every approach stays auditable.</span></h1>
          <p className="hero-copy">
            Frozen V3 and Trader Brain remain the medium/long-horizon forward experiment. The 15-minute lab now keeps the conservative momentum/mean-reversion benchmarks and separate PAPER-only exploration ledgers that intentionally trade more often without contaminating the original results.
          </p>
        </div>
        <div className="hero-badges">
          <span className={health?.status === "ok" ? "status-badge on" : "status-badge off"}>{health?.status === "ok" ? "API live" : "API offline"}</span>
          <span className="status-badge on">PAPER ONLY</span>
          <span className="status-badge on">REAL ORDERS DISABLED</span>
        </div>
      </section>

      <SystemOverview
        models={models}
        apiOk={health?.status === "ok"}
        tradingMode={health?.trading_mode}
        btcPrice={quote ? Number(quote.last) : null}
      />

      {coreModels.length > 0 && (
        <ModelComparisonTabs
          models={coreModels}
          tradesByModel={tradesByModel}
          decisionsByModel={decisionsByModel}
          eyebrow="CORE FORWARD EXPERIMENT · UNCHANGED"
          title="Medium / long-horizon performance"
        />
      )}

      {shortTermModels.length > 0 && (
        <>
          <ModelComparisonTabs
            models={shortTermModels}
            tradesByModel={tradesByModel}
            decisionsByModel={decisionsByModel}
            eyebrow="SHORT-TERM LAB · OFFICIAL + EXPLORATION POLICY LEDGERS"
            title="Intraday performance comparison"
            decisionScoreLabel="Setup score"
          />
          <ShortTermDiagnostics />
        </>
      )}

      <LivePredictionTerminal />

      <section style={panel}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div>
            <p className="eyebrow">DAILY GIT AUDIT</p>
            <h2>Performance report pushed every day</h2>
            <p>The server exports the side-by-side leaderboard plus per-algorithm trades, decisions, positions, and Trader-Brain experience/reward rows. The existing daily job commits the snapshot to <code>paper-live-results</code>, with no API keys or .env values.</p>
          </div>
          <span className="placeholder-chip">23:55 UTC</span>
        </div>
      </section>
      <footer className="footer"><p>Research/PAPER trading only. Forward performance is evidence, not a guarantee of profitability.</p></footer>
    </main>
  );
}
