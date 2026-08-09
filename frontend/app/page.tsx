import {
  getHealth,
  getMarketQuote,
  getPaperDecisions,
  getPaperPerformance,
  getPaperPortfolio,
  getPaperTrades,
} from "../lib/api";

const watchlist = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "CETUSDT"];

function toNumber(value: string | number | undefined | null): number {
  const number = Number(value ?? "0");
  return Number.isFinite(number) ? number : 0;
}

function formatMoney(value: string | number | undefined | null) {
  return toNumber(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatSignedMoney(value: string | number | undefined | null) {
  const number = toNumber(value);
  const sign = number > 0 ? "+" : "";
  return `${sign}${formatMoney(number)}`;
}

function formatPrice(value: string | undefined) {
  const number = Number(value ?? "0");
  if (!Number.isFinite(number)) return value ?? "—";
  return number.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

function formatPercent(value: string | number | undefined | null, signed = false) {
  const number = toNumber(value) * 100;
  const sign = signed && number > 0 ? "+" : "";
  return `${sign}${number.toFixed(2)}%`;
}

function signClass(value: string | number | undefined | null) {
  const number = toNumber(value);
  if (number > 0) return "pos";
  if (number < 0) return "neg";
  return "neutral";
}

function formatRelativeTime(iso: string | undefined | null) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const diffSeconds = Math.round((Date.now() - then) / 1000);
  if (diffSeconds < 5) return "just now";
  if (diffSeconds < 60) return `${diffSeconds}s ago`;
  const diffMinutes = Math.round(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.round(diffHours / 24);
  return `${diffDays}d ago`;
}

function badgeClassForSignal(signal: string) {
  if (signal === "BUY") return "badge buy";
  if (signal === "SELL") return "badge sell";
  return "badge hold";
}

export default async function DashboardPage() {
  const [health, portfolio, decisions, trades, performance, ...quotes] = await Promise.all([
    getHealth(),
    getPaperPortfolio(),
    getPaperDecisions(6),
    getPaperTrades(6),
    getPaperPerformance(),
    ...watchlist.map((symbol) => getMarketQuote(symbol)),
  ]);

  const backendOnline = health?.status === "ok";
  const tradingMode = health?.trading_mode?.toUpperCase() ?? "PAPER";
  const generatedAt = new Date().toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

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
          <span className="refreshed-at">Updated {generatedAt}</span>
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
          <strong className={portfolio ? signClass(portfolio.total_pnl_usdt) : ""}>
            {portfolio ? `${formatSignedMoney(portfolio.total_pnl_usdt)} USDT` : "—"}
          </strong>
          <small className={portfolio ? signClass(portfolio.total_return) : ""}>
            {portfolio ? formatPercent(portfolio.total_return, true) : "Waiting for backend"}
          </small>
        </article>
        <article className="metric-card">
          <span>Realized P/L</span>
          <strong className={performance ? signClass(performance.realized_pnl_usdt) : ""}>
            {performance ? `${formatSignedMoney(performance.realized_pnl_usdt)} USDT` : "—"}
          </strong>
          <small>{performance ? `${formatMoney(performance.total_fees_usdt)} USDT paid in fees` : "No trades yet"}</small>
        </article>
        <article className="metric-card">
          <span>Win rate</span>
          <strong>{performance?.win_rate != null ? formatPercent(performance.win_rate) : "—"}</strong>
          <small>
            {performance
              ? `${performance.winning_trades}/${performance.closed_trades} closed trades won`
              : "No closed trades yet"}
          </small>
        </article>
        <article className="metric-card">
          <span>Open positions</span>
          <strong>{portfolio?.positions.length ?? 0}</strong>
          <small>Simulated positions only</small>
        </article>
        <article className="metric-card">
          <span>Risk state</span>
          <strong>PAPER</strong>
          <small>Risk checks enabled &middot; live execution disabled</small>
        </article>
      </section>

      <section className="dashboard-grid">
        <div className="grid-column">
          <article className="panel large-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">PAPER PORTFOLIO</p>
                <h2>Positions</h2>
              </div>
              <span className="placeholder-chip">SIMULATED</span>
            </div>
            {portfolio && portfolio.positions.length > 0 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Quantity</th>
                      <th>Avg entry</th>
                      <th>Last price</th>
                      <th>Market value</th>
                      <th>Unrealized P/L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.positions.map((position) => (
                      <tr key={position.symbol}>
                        <td><strong>{position.symbol}</strong></td>
                        <td>{formatPrice(position.quantity)}</td>
                        <td>{formatPrice(position.avg_entry_price)}</td>
                        <td>
                          {formatPrice(position.last_price)}
                          {position.price_source !== "coinex" && (
                            <span className="inline-chip">fallback</span>
                          )}
                        </td>
                        <td>{formatMoney(position.market_value_usdt)} USDT</td>
                        <td className={signClass(position.unrealized_pnl_usdt)}>
                          {formatSignedMoney(position.unrealized_pnl_usdt)} USDT
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                No paper positions yet. Send a BUY / SELL / HOLD signal to the paper API to start recording model performance.
              </div>
            )}
          </article>

          <article className="panel large-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">TRADE LOG</p>
                <h2>Recent fills</h2>
              </div>
              <span className="placeholder-chip">{performance ? `${performance.trade_count} total` : "0 total"}</span>
            </div>
            {trades.length > 0 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Side</th>
                      <th>Symbol</th>
                      <th>Quantity</th>
                      <th>Exec price</th>
                      <th>Fee</th>
                      <th>Realized P/L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.map((trade) => (
                      <tr key={trade.id}>
                        <td className="muted">{formatRelativeTime(trade.created_at)}</td>
                        <td><span className={badgeClassForSignal(trade.side)}>{trade.side}</span></td>
                        <td><strong>{trade.symbol}</strong></td>
                        <td>{formatPrice(trade.quantity)}</td>
                        <td>{formatPrice(trade.execution_price)}</td>
                        <td className="muted">{formatMoney(trade.fee_usdt)} USDT</td>
                        <td className={signClass(trade.realized_pnl_usdt)}>
                          {trade.side === "SELL" ? `${formatSignedMoney(trade.realized_pnl_usdt)} USDT` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">No paper fills yet. Filled trades will appear here with fees and realized P/L.</div>
            )}
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">BOT ACTIVITY</p>
                <h2>System state</h2>
              </div>
            </div>
            <div className="activity-item"><span className="activity-time">Market</span><p>CoinEx public ticker and candlestick data are read-only.</p></div>
            <div className="activity-item"><span className="activity-time">Broker</span><p>Paper fills include configurable fees and slippage and are logged to SQLite.</p></div>
            <div className="activity-item"><span className="activity-time">Safety</span><p>There is no CoinEx order-placement or withdrawal code.</p></div>
          </article>
        </div>

        <div className="grid-column">
          <article className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">COINEX MARKET DATA</p>
                <h2>Live watchlist</h2>
              </div>
            </div>
            <div className="watchlist">
              {watchlist.map((symbol, index) => {
                const quote = quotes[index];
                const changePct = quote ? ((Number(quote.last) - Number(quote.open)) / Number(quote.open)) * 100 : null;
                return (
                  <div className="watch-row" key={symbol}>
                    <strong>{symbol}</strong>
                    {quote ? (
                      <span className="watch-quote">
                        <span>{formatPrice(quote.last)} USDT</span>
                        {changePct !== null && Number.isFinite(changePct) && (
                          <span className={`change-pill ${changePct >= 0 ? "pos" : "neg"}`}>
                            {changePct >= 0 ? "▲" : "▼"} {Math.abs(changePct).toFixed(2)}%
                          </span>
                        )}
                      </span>
                    ) : (
                      <span>Market data unavailable</span>
                    )}
                  </div>
                );
              })}
            </div>
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">MODEL DECISIONS</p>
                <h2>Recent signals</h2>
              </div>
            </div>
            {decisions.length > 0 ? (
              <div className="decision-list">
                {decisions.map((decision) => (
                  <div className="decision-row" key={decision.id}>
                    <div className="decision-top">
                      <span className={badgeClassForSignal(decision.signal)}>{decision.signal}</span>
                      <strong>{decision.symbol}</strong>
                      <span className={`badge ${decision.approved ? "approved" : "rejected"}`}>
                        {decision.approved ? "approved" : "rejected"}
                      </span>
                      <span className="decision-time">{formatRelativeTime(decision.created_at)}</span>
                    </div>
                    <p className="decision-reason">{decision.reason}</p>
                    <p className="decision-meta">
                      {decision.model_version} &middot; {decision.strategy_version}
                      {decision.confidence != null && ` · ${(decision.confidence * 100).toFixed(0)}% confidence`}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">No model decisions have been recorded yet.</div>
            )}
          </article>
        </div>
      </section>
    </main>
  );
}
