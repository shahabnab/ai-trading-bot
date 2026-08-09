import { getCoinExBalances, getHealth } from "../lib/api";

const watchlist = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "CETUSDT"];

function formatAmount(value: string) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value;
  return number.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

export default async function DashboardPage() {
  const health = await getHealth();
  const backendOnline = health?.status === "ok";
  const tradingMode = health?.trading_mode?.toUpperCase() ?? "PAPER";
  const coinex = health?.coinex_configured ? await getCoinExBalances() : null;
  const coinexConnected = Boolean(coinex);

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
        <strong>Paper mode only.</strong> CoinEx integration is read-only. No live-order execution is enabled.
      </section>

      <section className="metrics-grid">
        <article className="metric-card"><span>Exchange</span><strong>CoinEx</strong><small>{coinexConnected ? "Connected · read-only" : "Not connected"}</small></article>
        <article className="metric-card"><span>Spot assets</span><strong>{coinex?.balances.length ?? 0}</strong><small>Non-zero balances</small></article>
        <article className="metric-card"><span>Today&apos;s P/L</span><strong>—</strong><small>Coming after price history</small></article>
        <article className="metric-card"><span>Risk state</span><strong>LOCKED</strong><small>Execution disabled</small></article>
      </section>

      <section className="dashboard-grid">
        <article className="panel large-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">COINEX SPOT ACCOUNT</p><h2>Current balances</h2></div>
            <span className="placeholder-chip">READ ONLY</span>
          </div>
          {coinexConnected && coinex && coinex.balances.length > 0 ? (
            <div className="watchlist">
              {coinex.balances.map((balance) => (
                <div className="watch-row" key={balance.ccy}>
                  <strong>{balance.ccy}</strong>
                  <span>{formatAmount(balance.total)} total · {formatAmount(balance.available)} available</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              {health?.coinex_configured
                ? "CoinEx credentials are configured, but balances could not be loaded. Check the backend terminal."
                : "Add your NEW CoinEx API credentials to the local .env file, then restart FastAPI."}
            </div>
          )}
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">WATCHLIST</p><h2>Crypto universe</h2></div></div>
          <div className="watchlist">
            {watchlist.map((symbol) => (
              <div className="watch-row" key={symbol}>
                <strong>{symbol}</strong>
                <span>Market prices coming next</span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">AI SIGNALS</p><h2>Model decisions</h2></div></div>
          <div className="empty-state">No model is connected yet. Signals will be added after a leakage-safe backtesting pipeline is established.</div>
        </article>

        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">BOT ACTIVITY</p><h2>Recent events</h2></div></div>
          <div className="activity-item"><span className="activity-time">System</span><p>CoinEx adapter is read-only.</p></div>
          <div className="activity-item"><span className="activity-time">System</span><p>Order execution remains disabled and risk manager stays fail-closed.</p></div>
        </article>
      </section>
    </main>
  );
}
