import { getHealth } from "../lib/api";

const watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"];

export default async function DashboardPage() {
  const health = await getHealth();
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
        <strong>Paper mode only.</strong> No live-order execution is enabled in this project phase.
      </section>

      <section className="metrics-grid">
        <article className="metric-card"><span>Portfolio value</span><strong>€10,000.00</strong><small>Demo value</small></article>
        <article className="metric-card"><span>Today&apos;s P/L</span><strong>€0.00</strong><small>No trades yet</small></article>
        <article className="metric-card"><span>Open positions</span><strong>0</strong><small>Paper account</small></article>
        <article className="metric-card"><span>Risk state</span><strong>LOCKED</strong><small>Execution disabled</small></article>
      </section>

      <section className="dashboard-grid">
        <article className="panel large-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">PERFORMANCE</p><h2>Portfolio curve</h2></div>
            <span className="placeholder-chip">Coming in Step 4</span>
          </div>
          <div className="chart-placeholder">
            <div className="chart-line" />
            <p>Real backtest and portfolio data will appear here.</p>
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">WATCHLIST</p><h2>Market universe</h2></div></div>
          <div className="watchlist">
            {watchlist.map((symbol) => (
              <div className="watch-row" key={symbol}>
                <strong>{symbol}</strong>
                <span>Waiting for market data</span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">AI SIGNALS</p><h2>Model decisions</h2></div></div>
          <div className="empty-state">No model is connected yet. We will add signals only after the backtesting pipeline is trustworthy.</div>
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">BOT ACTIVITY</p><h2>Recent events</h2></div></div>
          <div className="activity-item"><span className="activity-time">Now</span><p>Dashboard initialized in paper-trading mode.</p></div>
          <div className="activity-item"><span className="activity-time">System</span><p>Risk manager remains fail-closed.</p></div>
        </article>
      </section>
    </main>
  );
}
