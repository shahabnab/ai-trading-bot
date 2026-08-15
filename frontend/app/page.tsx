import LivePredictionTerminal from "../components/LivePredictionTerminal";
import {
  getHealth,
  getMarketQuote,
  getPaperModels,
  getPaperModelTrades,
  type PaperModelAccount,
  type PaperTrade,
} from "../lib/api";

const watchlist = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "CETUSDT"];

function toNumber(value: string | number | undefined | null): number {
  const n = Number(value ?? 0);
  return Number.isFinite(n) ? n : 0;
}

function money(value: string | number | undefined | null): string {
  return toNumber(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function signedMoney(value: string | number | undefined | null): string {
  const n = toNumber(value);
  return `${n > 0 ? "+" : ""}${money(n)}`;
}

function percent(value: string | number | undefined | null, signed = false): string {
  const n = toNumber(value) * 100;
  return `${signed && n > 0 ? "+" : ""}${n.toFixed(2)}%`;
}

function price(value: string | number | undefined | null): string {
  const n = toNumber(value);
  if (!n) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: n >= 1000 ? 2 : 8 });
}

function signClass(value: string | number | undefined | null): string {
  const n = toNumber(value);
  return n > 0 ? "pos" : n < 0 ? "neg" : "neutral";
}

function relativeTime(iso: string | undefined | null): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return "—";
  const sec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (sec < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hrs = Math.round(min / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

function signalClass(signal: string | undefined | null): string {
  if (signal === "BUY") return "signal-pill buy";
  if (signal === "SELL") return "signal-pill sell";
  return "signal-pill hold";
}

function roleLabel(model: PaperModelAccount): string {
  return model.role === "paper_strategy" ? "PRIMARY PAPER MODEL" : "RESEARCH CONTROL";
}

function ModelCard({ model }: { model: PaperModelAccount }) {
  const p = model.portfolio;
  const perf = model.performance;
  const latest = model.latest_decision;
  const position = p.positions[0];

  return (
    <article className={`strategy-card ${model.role === "paper_strategy" ? "primary" : "control"}`}>
      <div className="strategy-card-head">
        <div>
          <span className="strategy-role">{roleLabel(model)}</span>
          <h2>{model.display_name}</h2>
          <p>{model.horizon_hours}h horizon · +{model.target_bps} bps event · {model.feature_set.replaceAll("_", " ")}</p>
        </div>
        <div className="strategy-state">
          <span className={`status-dot ${model.live_status === "paper_running" ? "online" : ""}`} />
          {model.live_status === "paper_running" ? "Forward paper" : "Waiting for signal"}
        </div>
      </div>

      <div className="capital-row">
        <div>
          <span>Paper equity</span>
          <strong>€{money(p.portfolio_value_usdt)}</strong>
          <small>Started €{money(p.initial_cash_usdt)} equivalent</small>
        </div>
        <div className="capital-pnl">
          <span>Net P/L</span>
          <strong className={signClass(p.total_pnl_usdt)}>€{signedMoney(p.total_pnl_usdt)}</strong>
          <small className={signClass(p.total_return)}>{percent(p.total_return, true)}</small>
        </div>
      </div>

      <div className="strategy-kpis">
        <div><span>Current signal</span><strong><span className={signalClass(latest?.signal)}>{latest?.signal ?? "WAIT"}</span></strong></div>
        <div><span>Confidence</span><strong>{latest?.confidence != null ? `${(latest.confidence * 100).toFixed(1)}%` : "—"}</strong></div>
        <div><span>Open position</span><strong>{position ? position.symbol : "CASH"}</strong></div>
        <div><span>Completed trades</span><strong>{perf.closed_trades}</strong></div>
        <div><span>Win rate</span><strong>{perf.win_rate == null ? "—" : percent(perf.win_rate)}</strong></div>
        <div><span>Fees</span><strong>€{money(perf.total_fees_usdt)}</strong></div>
      </div>

      <div className="research-strip">
        <div><span>V3 OOS AUC</span><strong>{model.research_auc.toFixed(3)}</strong></div>
        <div><span>Median AUC</span><strong>{model.research_median_auc.toFixed(3)}</strong></div>
        <div><span>Sharpe @25bps</span><strong className={signClass(model.research_sharpe_25bps)}>{model.research_sharpe_25bps > 0 ? "+" : ""}{model.research_sharpe_25bps.toFixed(3)}</strong></div>
        <div><span>Research return</span><strong className={signClass(model.research_return_25bps)}>{percent(model.research_return_25bps, true)}</strong></div>
      </div>

      <div className="strategy-footer">
        <span className="warning-chip">RESEARCH GATE FAILED</span>
        <span>{latest ? `Last decision ${relativeTime(latest.created_at)}` : "No forward decisions recorded yet"}</span>
      </div>
    </article>
  );
}

function TradesPanel({ model, trades }: { model: PaperModelAccount; trades: PaperTrade[] }) {
  return (
    <article className="panel model-trades-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{roleLabel(model)}</p>
          <h2>{model.display_name} trades</h2>
        </div>
        <span className="placeholder-chip">{model.performance.trade_count} fills · {model.performance.closed_trades} completed</span>
      </div>
      {trades.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Time</th><th>Side</th><th>Symbol</th><th>Qty</th><th>Execution</th><th>Fee</th><th>Realized P/L</th><th>Confidence</th></tr>
            </thead>
            <tbody>
              {trades.map((trade) => (
                <tr key={`${model.model_id}-${trade.id}`}>
                  <td className="muted">{relativeTime(trade.created_at)}</td>
                  <td><span className={signalClass(trade.side)}>{trade.side}</span></td>
                  <td><strong>{trade.symbol}</strong></td>
                  <td>{price(trade.quantity)}</td>
                  <td>{price(trade.execution_price)}</td>
                  <td>€{money(trade.fee_usdt)}</td>
                  <td className={signClass(trade.realized_pnl_usdt)}>{trade.side === "SELL" ? `€${signedMoney(trade.realized_pnl_usdt)}` : "—"}</td>
                  <td>{trade.confidence == null ? "—" : `${(trade.confidence * 100).toFixed(0)}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-state">No paper fills yet for this model. Its trades will be recorded here independently from every other model.</div>
      )}
    </article>
  );
}

export default async function DashboardPage() {
  const [health, paperModels, ...quotes] = await Promise.all([
    getHealth(),
    getPaperModels(),
    ...watchlist.map((symbol) => getMarketQuote(symbol)),
  ]);

  const models = paperModels?.models ?? [];
  const tradesByModel = await Promise.all(models.map((model) => getPaperModelTrades(model.model_id, 50)));
  const backendOnline = health?.status === "ok";
  const btc = quotes[0];
  const btcChange = btc && toNumber(btc.open) ? (toNumber(btc.last) - toNumber(btc.open)) / toNumber(btc.open) : 0;
  const currentPaperCapital = models.reduce((sum, model) => sum + toNumber(model.portfolio.portfolio_value_usdt), 0);
  const startingPaperCapital = models.reduce((sum, model) => sum + toNumber(model.portfolio.initial_cash_usdt), 0);

  return (
    <main className="trading-page">
      <nav className="exchange-nav">
        <div className="brand-lockup">
          <span className="brand-mark">AI</span>
          <div><strong>AI Trading Bot</strong><small>Forward Research Terminal</small></div>
        </div>
        <div className="exchange-links"><span className="active">Overview</span><span>Models</span><span>Paper trades</span><span>Research</span></div>
        <div className="nav-status"><span className={`status-dot ${backendOnline ? "online" : "offline"}`} /><span>{backendOnline ? "System online" : "Backend offline"}</span><span className="paper-chip">PAPER ONLY</span></div>
      </nav>

      <section className="market-bar">
        <div><span>BTC / USDT</span><strong>{btc ? `$${price(btc.last)}` : "—"}</strong><small className={signClass(btcChange)}>{btc ? percent(btcChange, true) : "Market unavailable"}</small></div>
        <div><span>Comparison capital</span><strong>€{money(paperModels?.starting_capital_eur_equiv_per_model ?? 1000)}</strong><small>per model · independent accounts</small></div>
        <div><span>Combined virtual equity</span><strong>€{money(currentPaperCapital)}</strong><small className={signClass(currentPaperCapital - startingPaperCapital)}>€{signedMoney(currentPaperCapital - startingPaperCapital)} vs starts</small></div>
        <div><span>Execution</span><strong>Disabled</strong><small>Real orders are hard-disabled</small></div>
      </section>

      <section className="paper-notice">
        <div><strong>Forward paper experiment</strong><span>Each model receives its own €1,000-equivalent ledger so performance and trades remain directly comparable.</span></div>
        <span>USDT ledger · EUR/USDT FX not modeled</span>
      </section>

      <section className="section-heading">
        <div><p className="eyebrow">MODEL ARENA</p><h1>Independent model performance</h1><p>Live paper results are kept separate from the historical V3 research metrics.</p></div>
      </section>

      <section className="strategy-grid">
        {models.length ? models.map((model) => <ModelCard key={model.model_id} model={model} />) : (
          <div className="empty-state">Model paper accounts are unavailable. Start the updated FastAPI backend and refresh this page.</div>
        )}
      </section>

      <section className="section-heading compact-heading">
        <div><p className="eyebrow">MARKET + PREDICTIONS</p><h2>BTC analysis terminal</h2></div>
      </section>
      <LivePredictionTerminal />

      <section className="market-and-watchlist">
        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">LIVE MARKET</p><h2>Watchlist</h2></div><span className="placeholder-chip">CoinEx public data</span></div>
          <div className="watchlist">
            {watchlist.map((symbol, i) => {
              const quote = quotes[i];
              const change = quote && toNumber(quote.open) ? (toNumber(quote.last) - toNumber(quote.open)) / toNumber(quote.open) : 0;
              return <div className="watch-row" key={symbol}><strong>{symbol}</strong><span className="watch-quote"><span>{quote ? `$${price(quote.last)}` : "—"}</span><span className={`change-pill ${change >= 0 ? "pos" : "neg"}`}>{quote ? percent(change, true) : "—"}</span></span></div>;
            })}
          </div>
        </article>
        <article className="panel research-note-panel">
          <div className="panel-heading"><div><p className="eyebrow">EXPERIMENT RULES</p><h2>What stays fixed</h2></div></div>
          <div className="rule-list">
            <div><span>01</span><p><strong>€1,000 equivalent per model.</strong> No shared cash or shared positions.</p></div>
            <div><span>02</span><p><strong>No leverage and no real orders.</strong> CoinEx is market-data only.</p></div>
            <div><span>03</span><p><strong>Trades remain model-specific.</strong> Results cannot be mixed between strategies.</p></div>
            <div><span>04</span><p><strong>Historical research stays visible.</strong> A model is not relabeled successful because of a short forward streak.</p></div>
          </div>
        </article>
      </section>

      <section className="section-heading compact-heading">
        <div><p className="eyebrow">AUDITABLE PAPER LEDGER</p><h2>Completed and open trade activity by model</h2><p>Every fill, fee and realized result is stored under the model that generated it.</p></div>
      </section>
      <section className="model-ledgers">
        {models.map((model, index) => <TradesPanel key={model.model_id} model={model} trades={tradesByModel[index] ?? []} />)}
      </section>
    </main>
  );
}
