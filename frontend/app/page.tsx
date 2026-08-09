import { getHealth, getMarketQuote, getPaperDecisions, getPaperPortfolio } from "../lib/api";

const watchlist = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "CETUSDT"];

function formatMoney(value: string | undefined) {
  const number = Number(value ?? "0");
  if (!Number.isFinite(number)) return value ?? "—";
  return number.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatPrice(value: string | undefined) {
  const number = Number(value ?? "0");
  if (!Number.isFinite(number)) return value ?? "—";
  return number.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

function formatPercent(value: string | undefined) {
  const number = Number(value ?? "0") * 100;
  if (!Number.isFinite(number)) return "—";
  return `${number.toFixed(2)}%`;
}

export default async function DashboardPage() {
  const [health, portfolio, decisions, ...quotes] = await Promise.all([
    getHealth(),
    getPaperPortfolio(),
    getPaperDecisions(6),
    ...watchlist.map((symbol) => getMarketQuote(symbol)),
  ]);

  const backendOnline = health?.status === "ok";
  const tradingMode = health?.trading_mode?.toUpperCase() ?? "PAPER";

  return (
    <main className="page-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">PERSONAL TRADING SYSTEM</p>
          <h1>AI Trading Bot</h1>
        </div>
        <div className="status-row">
          <span className={`status-dot ${backendOnline ? "online" : "offline"}`} />
          <span>{backendOnline ? "Backend online" : "Backend offline"}</span>
          <span className="mode-badge">{tradingMode} TRADING</span>
        </div>
      </header>

      <section className="warning-card">
        <strong>Paper mode only.</strong> Prices come from CoinEx, but every order, position and P/L value is simulated locally. No live-order execution exists in this phase.
      </section>

      <section className="metrics-grid">
        <article className="metric-card">
          <span>Paper portfolio</span>
          <strong>{portfolio ? `${formatMoney(portfolio.portfolio_value_usdt)} USDT` : "—"}</strong>
          <small>Started with {portfolio ? formatMoney(portfolio.initial_cash_usdt) : "10,000.00"} USDT</small>
        </article>
        <article className="metric-card">
          <span>Total P/L</span>
          <strong>{portfolio ? `${formatMoney(portfolio.total_pnl_usdt)} USDT` : "—"}</strong>
          <small>{portfolio ? formatPercent(portfolio.total_return) : "Waiting for backend"}</small>
        </article>
        <article className="metric-card">
          <span>Open positions</span>
          <strong>{portfolio?.positions.length ?? 0}</strong>
          <small>Simulated positions only</small>
        </article>
        <article className="metric-card">
          <span>Risk state</span>
          <strong>PAPER</strong>
          <small>Risk checks enabled · live execution disabled</small>
        </article>
      </section>

      <section className="dashboard-grid">
        <article className="panel large-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">PAPER PORTFOLIO</p><h2>Positions</h2></div>
            <span className="placeholder-chip">SIMULATED</span>
          </div>
          {portfolio && portfolio.positions.length > 0 ? (
            <div className="watchlist">
              {portfolio.positions.map((position) => (
                <div className="watch-row" key={position.symbol}>
                  <strong>{position.symbol}</strong>
                  <span>
                    {formatPrice(position.quantity)} units · {formatMoney(position.market_value_usdt)} USDT · P/L {formatMoney(position.unrealized_pnl_usdt)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              No paper positions yet. Send a BUY / SELL / HOLD signal to the paper API to start recording model performance.
            </div>
          )}
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">COINEX MARKET DATA</p><h2>Live watchlist</h2></div></div>
          <div className="watchlist">
            {watchlist.map((symbol, index) => (
              <div className="watch-row" key={symbol}>
                <strong>{symbol}</strong>
                <span>{quotes[index] ? `${formatPrice(quotes[index]?.last)} USDT` : "Market data unavailable"}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">MODEL DECISIONS</p><h2>Recent signals</h2></div></div>
          {decisions.length > 0 ? (
            <div className="watchlist">
              {decisions.map((decision) => (
                <div className="watch-row" key={decision.id}>
                  <strong>{decision.signal} {decision.symbol}</strong>
                  <span>{decision.approved ? "accepted" : "rejected"} · {decision.model_version} · {decision.confidence == null ? "no confidence" : `${(decision.confidence * 100).toFixed(0)}%`}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">No model decisions have been recorded yet.</div>
          )}
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">BOT ACTIVITY</p><h2>System state</h2></div></div>
          <div className="activity-item"><span className="activity-time">Market</span><p>CoinEx public ticker and candlestick data are read-only.</p></div>
          <div className="activity-item"><span className="activity-time">Broker</span><p>Paper fills include configurable fees and slippage and are logged to SQLite.</p></div>
          <div className="activity-item"><span className="activity-time">Safety</span><p>There is no CoinEx order-placement or withdrawal code.</p></div>
        </article>
      </section>
    </main>
  );
}
