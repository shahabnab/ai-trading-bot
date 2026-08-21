import type { PaperModel } from "./ModelComparisonTabs";

type Props = {
  models: PaperModel[];
  apiOk: boolean;
  tradingMode?: string | null;
  btcPrice?: number | null;
};

function n(value: string | number | null | undefined): number {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function money(value: string | number | null | undefined): string {
  return n(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function pct(value: string | number | null | undefined): string {
  const v = n(value) * 100;
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function statusFor(model: PaperModel): string {
  if (model.portfolio.positions.length > 0) return "POSITION OPEN";
  if (model.experimental) return "EXPLORE";
  if (model.performance.closed_trades > 0) return "ACTIVE";
  if ((model.driver ?? "").startsWith("trader_brain")) return "COLLECTING";
  return "WAITING";
}

function positionFor(model: PaperModel): string {
  const btc = model.portfolio.positions.find((row) => row.symbol === "BTCUSDT");
  return btc ? `LONG ${Number(btc.quantity).toFixed(5)} BTC` : "FLAT";
}

function why(model: PaperModel): string {
  const raw = model.latest_decision?.reason?.trim();
  if (!raw) return model.performance.decision_count > 0 ? "No explanatory reason stored." : "Waiting for the first decision.";
  const withoutRisk = raw.split("RiskManager=")[0]?.trim() || raw;
  const stripped = withoutRisk.replace(/^15m\s+[^;]+;\s*/i, "");
  return stripped.length > 220 ? `${stripped.slice(0, 217)}…` : stripped;
}

function statusColor(status: string): string {
  if (status === "POSITION OPEN" || status === "ACTIVE") return "#2dd4bf";
  if (status === "EXPLORE" || status === "COLLECTING") return "#fbbf24";
  return "#94a3b8";
}

const panel: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,.22)",
  borderRadius: 14,
  padding: 18,
  background: "rgba(15,23,42,.52)",
};

const metric: React.CSSProperties = {
  ...panel,
  padding: 14,
};

const cell: React.CSSProperties = {
  padding: "10px 11px",
  borderBottom: "1px solid rgba(148,163,184,.15)",
  textAlign: "left",
  verticalAlign: "top",
};

export default function SystemOverview({ models, apiOk, tradingMode, btcPrice }: Props) {
  const totalEquity = models.reduce((sum, model) => sum + n(model.portfolio.portfolio_value_usdt), 0);
  const totalPnl = models.reduce((sum, model) => sum + n(model.portfolio.total_pnl_usdt), 0);
  const closedTrades = models.reduce((sum, model) => sum + model.performance.closed_trades, 0);
  const fees = models.reduce((sum, model) => sum + n(model.performance.total_fees_usdt), 0);
  const openPositions = models.filter((model) => model.portfolio.positions.length > 0).length;
  const ranked = [...models].sort((a, b) => n(b.portfolio.total_return) - n(a.portfolio.total_return));

  return (
    <section style={{ display: "grid", gap: 14 }}>
      <article style={panel}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap", alignItems: "end" }}>
          <div>
            <p className="eyebrow">WHAT IS HAPPENING RIGHT NOW?</p>
            <h2 style={{ margin: "4px 0" }}>System and algorithm status</h2>
            <p style={{ marginBottom: 0 }}>
              Read this section first. It separates system health, actual PAPER performance, current position, last action and the reason each algorithm did or did not trade.
            </p>
          </div>
          <div style={{ textAlign: "right" }}>
            <strong style={{ color: apiOk ? "#2dd4bf" : "#fb7185" }}>{apiOk ? "SYSTEM ONLINE" : "API OFFLINE"}</strong><br />
            <small>{String(tradingMode ?? "paper").toUpperCase()} · real orders disabled</small>
          </div>
        </div>
      </article>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 10 }}>
        <div style={metric}><small>BTCUSDT</small><h3>{btcPrice == null ? "—" : `$${btcPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}</h3><span>Live reference</span></div>
        <div style={metric}><small>TOTAL EQUITY</small><h3>€{money(totalEquity)}</h3><span>{models.length} independent ledgers</span></div>
        <div style={metric}><small>NET P/L</small><h3 style={{ color: totalPnl > 0 ? "#2dd4bf" : totalPnl < 0 ? "#fb7185" : "inherit" }}>€{money(totalPnl)}</h3><span>Sum across ledgers</span></div>
        <div style={metric}><small>CLOSED TRADES</small><h3>{closedTrades}</h3><span>Round trips, not executions</span></div>
        <div style={metric}><small>OPEN POSITIONS</small><h3>{openPositions}</h3><span>Across all ledgers</span></div>
        <div style={metric}><small>FEES PAID</small><h3>€{money(fees)}</h3><span>Simulated costs</span></div>
      </div>

      <article style={panel}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
          <div><p className="eyebrow">ALGORITHM SCOREBOARD</p><h3 style={{ margin: "4px 0" }}>Performance + current action + why</h3></div>
          <small>EXPLORE = separate PAPER-only relaxed policy ledger</small>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={cell}>Status</th>
                <th style={cell}>Algorithm</th>
                <th style={cell}>Equity</th>
                <th style={cell}>Net P/L</th>
                <th style={cell}>Return</th>
                <th style={cell}>Closed trades</th>
                <th style={cell}>Win rate</th>
                <th style={cell}>Position</th>
                <th style={cell}>Last action</th>
                <th style={{ ...cell, minWidth: 310 }}>Why?</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((model) => {
                const status = statusFor(model);
                const pnl = n(model.portfolio.total_pnl_usdt);
                return (
                  <tr key={model.model_id}>
                    <td style={{ ...cell, color: statusColor(status), fontWeight: 700 }}>{status}</td>
                    <td style={cell}>
                      <strong>{model.display_name}</strong><br />
                      <small>{model.experimental ? "PAPER exploration policy" : model.algorithm_family ?? model.driver ?? model.model_id}</small>
                    </td>
                    <td style={cell}>€{money(model.portfolio.portfolio_value_usdt)}</td>
                    <td style={{ ...cell, color: pnl > 0 ? "#2dd4bf" : pnl < 0 ? "#fb7185" : "inherit" }}>€{money(model.portfolio.total_pnl_usdt)}</td>
                    <td style={{ ...cell, color: n(model.portfolio.total_return) > 0 ? "#2dd4bf" : n(model.portfolio.total_return) < 0 ? "#fb7185" : "inherit" }}>{pct(model.portfolio.total_return)}</td>
                    <td style={cell}>{model.performance.closed_trades}</td>
                    <td style={cell}>{model.performance.win_rate == null ? "—" : `${(model.performance.win_rate * 100).toFixed(1)}%`}</td>
                    <td style={cell}>{positionFor(model)}</td>
                    <td style={cell}><strong>{model.latest_decision?.signal ?? "WAIT"}</strong></td>
                    <td style={{ ...cell, whiteSpace: "normal" }}>{why(model)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </article>

      <article style={{ ...panel, borderColor: "rgba(251,191,36,.35)" }}>
        <strong>How to read HOLD:</strong> HOLD is a real model decision, not automatically a system failure. The “Why?” column tells you whether the setup was weak, expected edge was below costs, the model forecast was bearish/uncertain, or an existing position was being managed. Exploration ledgers deliberately relax only the short-term entry policy; the frozen V3 controls remain untouched.
      </article>
    </section>
  );
}
