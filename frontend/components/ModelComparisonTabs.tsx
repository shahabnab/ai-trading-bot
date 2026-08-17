"use client";

import { useMemo, useState } from "react";

export type PaperDecision = {
  id?: number;
  created_at?: string;
  signal?: string;
  confidence?: number | null;
  approved?: number;
  reason?: string;
};

export type PaperTrade = {
  id?: number;
  created_at?: string;
  side?: string;
  quantity?: string;
  execution_price?: string;
  fee_usdt?: string;
  realized_pnl_usdt?: string;
  confidence?: number | null;
};

export type PaperModel = {
  model_id: string;
  display_name: string;
  algorithm_family?: string;
  driver?: string;
  description?: string;
  horizon_hours: number;
  feature_set: string;
  adaptive?: boolean;
  research_auc?: number;
  research_sharpe_25bps?: number;
  research_return_25bps?: number;
  research_trades?: number;
  portfolio: {
    portfolio_value_usdt: string;
    total_pnl_usdt: string;
    total_return: string;
    positions: Array<{ symbol: string; quantity: string; avg_entry_price: string; unrealized_pnl_usdt: string }>;
  };
  performance: {
    trade_count: number;
    decision_count: number;
    closed_trades: number;
    winning_trades: number;
    win_rate: number | null;
    total_fees_usdt: string;
    realized_pnl_usdt: string;
  };
  latest_decision?: PaperDecision | null;
  live_status?: string;
};

type Props = {
  models: PaperModel[];
  tradesByModel: Record<string, PaperTrade[]>;
  decisionsByModel: Record<string, PaperDecision[]>;
  eyebrow?: string;
  title?: string;
  decisionScoreLabel?: string;
};

function n(value: string | number | null | undefined): number {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}
function money(value: string | number | null | undefined): string {
  return n(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function pct(value: string | number | null | undefined, signed = false): string {
  const v = n(value) * 100;
  return `${signed && v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}
function signalColor(signal?: string | null): string {
  return signal === "BUY" ? "#2dd4bf" : signal === "SELL" ? "#fb7185" : "#94a3b8";
}
function pnlColor(value: string | number | null | undefined): string {
  const v = n(value);
  return v > 0 ? "#2dd4bf" : v < 0 ? "#fb7185" : "inherit";
}

const card: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,.22)", borderRadius: 14, padding: 16,
  background: "rgba(15,23,42,.52)", minWidth: 0,
};
const thtd: React.CSSProperties = { padding: "10px 12px", borderBottom: "1px solid rgba(148,163,184,.15)", textAlign: "left" };

export default function ModelComparisonTabs({
  models,
  tradesByModel,
  decisionsByModel,
  eyebrow = "ALL ALGORITHMS · SAME PAPER CONDITIONS",
  title = "Forward performance comparison",
  decisionScoreLabel = "Confidence",
}: Props) {
  const ranked = useMemo(() => [...models].sort((a, b) => n(b.portfolio.total_return) - n(a.portfolio.total_return)), [models]);
  const [activeId, setActiveId] = useState(ranked[0]?.model_id ?? "");
  const active = ranked.find((model) => model.model_id === activeId) ?? ranked[0];
  if (!active) return null;
  const trades = tradesByModel[active.model_id] ?? [];
  const decisions = decisionsByModel[active.model_id] ?? [];

  return (
    <section style={{ display: "grid", gap: 20 }}>
      <article style={card}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "end", marginBottom: 12 }}>
          <div><small>{eyebrow}</small><h2 style={{ margin: "4px 0" }}>{title}</h2></div>
          <span>{models.length} independent ledgers</span>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead><tr><th style={thtd}>#</th><th style={thtd}>Algorithm</th><th style={thtd}>Family</th><th style={thtd}>Equity</th><th style={thtd}>Net P/L</th><th style={thtd}>Return</th><th style={thtd}>Fills</th><th style={thtd}>Win rate</th><th style={thtd}>Fees</th><th style={thtd}>Latest</th></tr></thead>
            <tbody>{ranked.map((model, idx) => (
              <tr key={model.model_id}>
                <td style={thtd}>{idx + 1}</td>
                <td style={thtd}><strong>{model.display_name}</strong><br /><small>{model.model_id}</small></td>
                <td style={thtd}>{model.algorithm_family ?? model.driver ?? "—"}</td>
                <td style={thtd}>€{money(model.portfolio.portfolio_value_usdt)}</td>
                <td style={{ ...thtd, color: pnlColor(model.portfolio.total_pnl_usdt) }}>€{money(model.portfolio.total_pnl_usdt)}</td>
                <td style={{ ...thtd, color: pnlColor(model.portfolio.total_return) }}>{pct(model.portfolio.total_return, true)}</td>
                <td style={thtd}>{model.performance.trade_count}</td>
                <td style={thtd}>{model.performance.win_rate == null ? "—" : pct(model.performance.win_rate)}</td>
                <td style={thtd}>€{money(model.performance.total_fees_usdt)}</td>
                <td style={{ ...thtd, color: signalColor(model.latest_decision?.signal) }}>{model.latest_decision?.signal ?? "WAIT"}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </article>

      <article style={card}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 18 }}>
          {ranked.map((model) => <button key={model.model_id} onClick={() => setActiveId(model.model_id)} style={{ borderRadius: 10, padding: "9px 12px", border: `1px solid ${active.model_id === model.model_id ? "#60a5fa" : "rgba(148,163,184,.25)"}`, background: active.model_id === model.model_id ? "rgba(96,165,250,.14)" : "transparent", color: "inherit", cursor: "pointer" }}>{model.display_name}</button>)}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))", gap: 12, marginBottom: 18 }}>
          <div style={card}><small>ARCHITECTURE</small><h3>{active.algorithm_family ?? active.driver}</h3><p>{active.description ?? "No description."}</p><p><b>Horizon:</b> {active.horizon_hours}h · <b>Features:</b> {active.feature_set} · <b>Adaptive:</b> {active.adaptive ? "YES" : "NO"}</p></div>
          <div style={card}><small>FORWARD PAPER</small><h3>€{money(active.portfolio.portfolio_value_usdt)}</h3><p style={{ color: pnlColor(active.portfolio.total_pnl_usdt) }}>Net P/L €{money(active.portfolio.total_pnl_usdt)} · {pct(active.portfolio.total_return, true)}</p><p>{active.performance.closed_trades} closed · {active.performance.winning_trades} wins · fees €{money(active.performance.total_fees_usdt)}</p></div>
          <div style={card}><small>HISTORICAL RESEARCH · NOT FORWARD</small><h3>{active.research_auc ? `AUC ${active.research_auc.toFixed(3)}` : "Collecting"}</h3><p>Sharpe {active.research_trades ? active.research_sharpe_25bps?.toFixed(3) : "—"} · Return {active.research_trades ? pct(active.research_return_25bps, true) : "—"} · Trades {active.research_trades ?? 0}</p></div>
        </div>

        <h3>Recent decisions</h3>
        <div style={{ overflowX: "auto", marginBottom: 18 }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th style={thtd}>Time</th><th style={thtd}>Signal</th><th style={thtd}>{decisionScoreLabel}</th><th style={thtd}>Approved</th><th style={thtd}>Audit reason</th></tr></thead><tbody>{decisions.length ? decisions.slice(0,20).map((row, idx) => <tr key={row.id ?? idx}><td style={thtd}>{row.created_at ? new Date(row.created_at).toLocaleString() : "—"}</td><td style={{ ...thtd, color: signalColor(row.signal) }}>{row.signal ?? "—"}</td><td style={thtd}>{row.confidence == null ? "—" : pct(row.confidence)}</td><td style={thtd}>{row.approved ? "yes" : "no"}</td><td style={{ ...thtd, whiteSpace: "normal", maxWidth: 700 }}>{row.reason ?? "—"}</td></tr>) : <tr><td style={thtd} colSpan={5}>No decisions yet.</td></tr>}</tbody></table></div>

        <h3>Recent simulated fills</h3>
        <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th style={thtd}>Time</th><th style={thtd}>Side</th><th style={thtd}>Qty</th><th style={thtd}>Execution</th><th style={thtd}>Fee</th><th style={thtd}>Realized P/L</th></tr></thead><tbody>{trades.length ? trades.slice(0,20).map((row, idx) => <tr key={row.id ?? idx}><td style={thtd}>{row.created_at ? new Date(row.created_at).toLocaleString() : "—"}</td><td style={{ ...thtd, color: signalColor(row.side) }}>{row.side}</td><td style={thtd}>{n(row.quantity).toFixed(6)}</td><td style={thtd}>{money(row.execution_price)}</td><td style={thtd}>€{money(row.fee_usdt)}</td><td style={{ ...thtd, color: pnlColor(row.realized_pnl_usdt) }}>€{money(row.realized_pnl_usdt)}</td></tr>) : <tr><td style={thtd} colSpan={6}>No fills yet.</td></tr>}</tbody></table></div>
      </article>
    </section>
  );
}
