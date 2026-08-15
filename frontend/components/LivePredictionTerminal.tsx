"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { MarketCandle, MarketKlinesResponse, MarketQuote, PaperModelAccount, PaperTrade } from "../lib/api";

type ForwardRecord = {
  recorded_at: string;
  model_id: string;
  display_name: string;
  feature_timestamp: number;
  feature_time_utc: string;
  model_sha256?: string | null;
  raw_probability: number;
  calibrated_probability: number;
  expected_gross_ev: number;
  decision_ev: number | null;
  entry_margin_bps: number;
  one_way_cost_rate: number;
  horizon_commitment_hours: number;
  position_before: "LONG" | "CASH" | string;
  policy_due: boolean;
  signal: "BUY" | "SELL" | "HOLD" | "COMMITMENT" | string;
  reason: string;
  paper_market_price: string;
  dry_run: boolean;
  trade: Record<string, unknown> | null;
};

type ForwardModel = PaperModelAccount & {
  artifact_ready: boolean;
  history: ForwardRecord[];
  trades: PaperTrade[];
};

type ForwardPayload = {
  mode: string;
  paper_only: boolean;
  real_orders_enabled: boolean;
  models: ForwardModel[];
};

type DashboardPayload = {
  generated_at: string;
  quote: MarketQuote | null;
  market: MarketKlinesResponse | null;
  forward: ForwardPayload | null;
};

type ModelKey = "primary" | "control";
type ChartRange = "1D" | "3D" | "1W" | "1M" | "ALL";

type ChartRefs = {
  chart: any;
  candle: any;
  primaryProbability: any;
  controlProbability: any;
  primaryEquity: any;
  controlEquity: any;
  primaryBuy: any;
  primarySell: any;
  controlBuy: any;
  controlSell: any;
};

const PRIMARY_ID = "v3-25bps-fullcontext-12h";
const CONTROL_ID = "v3-50bps-technical-3h";
const LWC_SRC = "https://unpkg.com/lightweight-charts@5.2.0/dist/lightweight-charts.standalone.production.js";
const REFRESH_MS = 10_000;

function loadLightweightCharts(): Promise<any> {
  const globalWindow = window as typeof window & { LightweightCharts?: any };
  if (globalWindow.LightweightCharts) return Promise.resolve(globalWindow.LightweightCharts);
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${LWC_SRC}"]`);
    if (existing) {
      existing.addEventListener("load", () => resolve(globalWindow.LightweightCharts), { once: true });
      existing.addEventListener("error", () => reject(new Error("Chart library failed to load")), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = LWC_SRC;
    script.async = true;
    script.crossOrigin = "anonymous";
    script.onload = () => resolve(globalWindow.LightweightCharts);
    script.onerror = () => reject(new Error("Chart library failed to load"));
    document.head.appendChild(script);
  });
}

function finiteNumber(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatPrice(value: unknown): string {
  const number = finiteNumber(value);
  if (number === null) return "—";
  return number.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: number >= 1000 ? 2 : 6 });
}

function formatPercent(value: unknown, digits = 2): string {
  const number = finiteNumber(value);
  if (number === null) return "—";
  return `${number > 0 ? "+" : ""}${(number * 100).toFixed(digits)}%`;
}

function formatProbability(value: unknown): string {
  const number = finiteNumber(value);
  return number === null ? "—" : `${(number * 100).toFixed(1)}%`;
}

function formatDateTime(timestamp: number): string {
  return new Date(timestamp).toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function modelById(payload: ForwardPayload | null | undefined, modelId: string): ForwardModel | null {
  return payload?.models.find((model) => model.model_id === modelId) ?? null;
}

function latest(model: ForwardModel | null): ForwardRecord | null {
  const rows = model?.history ?? [];
  return rows.length ? rows[rows.length - 1] : null;
}

function modelState(model: ForwardModel | null): string {
  if (!model?.artifact_ready) return "WAITING FOR ARTIFACT";
  if (!model.history.length) return "READY · WAITING FOR FIRST HOUR";
  return "FORWARD LIVE";
}

function signalClass(signal: string | undefined | null): string {
  if (signal === "BUY") return "pos";
  if (signal === "SELL") return "neg";
  return "neutral";
}

function nextDecision(model: ForwardModel | null): string {
  if (!model?.history.length) return "—";
  const policyRows = [...model.history].reverse().filter((row) => row.policy_due);
  const lastPolicy = policyRows[0];
  if (!lastPolicy) return "—";
  const next = lastPolicy.feature_timestamp + lastPolicy.horizon_commitment_hours * 3_600_000;
  return formatDateTime(next);
}

function dedupeLine(rows: { time: number; value: number }[]): { time: number; value: number }[] {
  const byTime = new Map<number, number>();
  for (const row of rows) {
    if (Number.isFinite(row.time) && Number.isFinite(row.value)) byTime.set(row.time, row.value);
  }
  return [...byTime.entries()].sort((a, b) => a[0] - b[0]).map(([time, value]) => ({ time, value }));
}

function equityCurve(model: ForwardModel | null): { time: number; value: number }[] {
  if (!model) return [];
  const records = [...model.history].sort((a, b) => a.feature_timestamp - b.feature_timestamp);
  const trades = [...model.trades].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
  let cash = Number(model.portfolio.initial_cash_usdt ?? 1000);
  let quantity = 0;
  let tradeIndex = 0;
  const points: { time: number; value: number }[] = [];

  for (const record of records) {
    const recordTime = new Date(record.recorded_at).getTime();
    while (tradeIndex < trades.length && new Date(trades[tradeIndex].created_at).getTime() <= recordTime) {
      const trade = trades[tradeIndex];
      const gross = Number(trade.gross_value_usdt);
      const fee = Number(trade.fee_usdt);
      const qty = Number(trade.quantity);
      if ([gross, fee, qty].every(Number.isFinite)) {
        if (trade.side === "BUY") {
          cash -= gross + fee;
          quantity += qty;
        } else if (trade.side === "SELL") {
          cash += gross - fee;
          quantity = Math.max(0, quantity - qty);
        }
      }
      tradeIndex += 1;
    }
    const price = Number(record.paper_market_price);
    if (Number.isFinite(price)) points.push({ time: Math.floor(record.feature_timestamp / 1000), value: cash + quantity * price });
  }
  return dedupeLine(points);
}

function tradePoints(model: ForwardModel | null, side: "BUY" | "SELL"): { time: number; value: number }[] {
  if (!model) return [];
  return model.trades
    .filter((trade) => trade.side === side)
    .map((trade) => ({ time: Math.floor(new Date(trade.created_at).getTime() / 1000), value: Number(trade.market_price) }))
    .filter((row) => Number.isFinite(row.time) && Number.isFinite(row.value))
    .sort((a, b) => a.time - b.time);
}

export default function LivePredictionTerminal() {
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRefs = useRef<ChartRefs | null>(null);
  const latestPayloadRef = useRef<DashboardPayload | null>(null);
  const followLiveRef = useRef(true);

  const [payload, setPayload] = useState<DashboardPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [followLive, setFollowLive] = useState(true);
  const [range, setRange] = useState<ChartRange>("1D");
  const [visibleModels, setVisibleModels] = useState<Record<ModelKey, boolean>>({ primary: true, control: true });

  const refresh = useCallback(async () => {
    try {
      const response = await fetch("/api/dashboard-data", { cache: "no-store" });
      if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
      const next = (await response.json()) as DashboardPayload;
      latestPayloadRef.current = next;
      setPayload(next);
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Dashboard data unavailable");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const applyRange = useCallback((nextRange: ChartRange) => {
    setRange(nextRange);
    const refs = chartRefs.current;
    const candles = latestPayloadRef.current?.market?.candles ?? [];
    if (!refs || !candles.length) return;
    const last = Number(candles[candles.length - 1]?.created_at);
    if (!Number.isFinite(last)) return;
    if (nextRange === "ALL") {
      refs.chart.timeScale().fitContent();
      followLiveRef.current = false;
      setFollowLive(false);
      return;
    }
    const days = nextRange === "1D" ? 1 : nextRange === "3D" ? 3 : nextRange === "1W" ? 7 : 30;
    refs.chart.timeScale().setVisibleRange({ from: Math.floor((last - days * 86_400_000) / 1000), to: Math.floor(last / 1000) + 6 * 3600 });
    followLiveRef.current = nextRange === "1D";
    setFollowLive(nextRange === "1D");
  }, []);

  const goLive = useCallback(() => {
    followLiveRef.current = true;
    setFollowLive(true);
    setRange("1D");
    chartRefs.current?.chart.timeScale().scrollToRealTime();
  }, []);

  useEffect(() => {
    let disposed = false;
    let localChart: any = null;
    void loadLightweightCharts().then((LWC) => {
      if (disposed || !chartContainerRef.current || !LWC) return;
      const chart = LWC.createChart(chartContainerRef.current, {
        autoSize: true,
        height: 720,
        layout: { background: { type: LWC.ColorType.Solid, color: "#08131d" }, textColor: "#8da2b4", panes: { separatorColor: "#1c2b37", separatorHoverColor: "#334b5e", enableResize: true } },
        grid: { vertLines: { color: "#12222e" }, horzLines: { color: "#12222e" } },
        rightPriceScale: { borderColor: "#20313f", scaleMargins: { top: 0.08, bottom: 0.08 } },
        timeScale: { borderColor: "#20313f", timeVisible: true, secondsVisible: false, rightOffset: 6, barSpacing: 8, minBarSpacing: 2 },
        handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
        handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
      });
      localChart = chart;
      const candle = chart.addSeries(LWC.CandlestickSeries, { upColor: "#20c997", downColor: "#f05d7a", wickUpColor: "#20c997", wickDownColor: "#f05d7a", borderVisible: false, priceLineVisible: true }, 0);
      const probabilityFormat = { type: "custom", minMove: 0.1, formatter: (value: number) => `${value.toFixed(1)}%` };
      const equityFormat = { type: "custom", minMove: 0.01, formatter: (value: number) => `€${value.toFixed(2)}` };
      const primaryProbability = chart.addSeries(LWC.LineSeries, { color: "#57a6ff", lineWidth: 2, title: "12h probability", priceFormat: probabilityFormat, lastValueVisible: true }, 1);
      const controlProbability = chart.addSeries(LWC.LineSeries, { color: "#d28cff", lineWidth: 2, title: "3h probability", priceFormat: probabilityFormat, lastValueVisible: true }, 1);
      const primaryEquity = chart.addSeries(LWC.LineSeries, { color: "#57a6ff", lineWidth: 2, title: "12h equity", priceFormat: equityFormat, lastValueVisible: true }, 2);
      const controlEquity = chart.addSeries(LWC.LineSeries, { color: "#d28cff", lineWidth: 2, title: "3h equity", priceFormat: equityFormat, lastValueVisible: true }, 2);
      const pointOptions = { lineVisible: false, pointMarkersVisible: true, pointMarkersRadius: 6, lastValueVisible: false, priceLineVisible: false };
      const primaryBuy = chart.addSeries(LWC.LineSeries, { ...pointOptions, color: "#20c997", title: "12h BUY" }, 0);
      const primarySell = chart.addSeries(LWC.LineSeries, { ...pointOptions, color: "#f05d7a", title: "12h SELL" }, 0);
      const controlBuy = chart.addSeries(LWC.LineSeries, { ...pointOptions, color: "#22d3ee", title: "3h BUY" }, 0);
      const controlSell = chart.addSeries(LWC.LineSeries, { ...pointOptions, color: "#f59e0b", title: "3h SELL" }, 0);
      const panes = chart.panes();
      if (panes[0]) panes[0].setHeight(430);
      if (panes[1]) panes[1].setHeight(145);
      if (panes[2]) panes[2].setHeight(145);
      chartRefs.current = { chart, candle, primaryProbability, controlProbability, primaryEquity, controlEquity, primaryBuy, primarySell, controlBuy, controlSell };
      if (latestPayloadRef.current) setPayload({ ...latestPayloadRef.current });
    }).catch((chartError) => {
      if (!disposed) setError(chartError instanceof Error ? chartError.message : "Chart initialization failed");
    });
    return () => {
      disposed = true;
      chartRefs.current = null;
      if (localChart) localChart.remove();
    };
  }, []);

  const primary = modelById(payload?.forward, PRIMARY_ID);
  const control = modelById(payload?.forward, CONTROL_ID);
  const primaryLatest = latest(primary);
  const controlLatest = latest(control);

  useEffect(() => {
    const refs = chartRefs.current;
    if (!refs || !payload) return;
    try {
      const visibleRange = refs.chart.timeScale().getVisibleRange();
      const candles = (payload.market?.candles ?? []).map((row: MarketCandle) => {
        const ms = Number(row.created_at);
        const open = Number(row.open); const high = Number(row.high); const low = Number(row.low); const close = Number(row.close);
        if (![ms, open, high, low, close].every(Number.isFinite)) return null;
        return { time: Math.floor(ms / 1000), open, high, low, close };
      }).filter((row): row is { time: number; open: number; high: number; low: number; close: number } => row !== null).sort((a, b) => a.time - b.time);

      const probabilityLine = (model: ForwardModel | null) => dedupeLine((model?.history ?? []).map((row) => ({ time: Math.floor(row.feature_timestamp / 1000), value: row.calibrated_probability * 100 })));
      refs.candle.setData(candles);
      refs.primaryProbability.setData(probabilityLine(primary));
      refs.controlProbability.setData(probabilityLine(control));
      refs.primaryEquity.setData(equityCurve(primary));
      refs.controlEquity.setData(equityCurve(control));
      refs.primaryBuy.setData(tradePoints(primary, "BUY"));
      refs.primarySell.setData(tradePoints(primary, "SELL"));
      refs.controlBuy.setData(tradePoints(control, "BUY"));
      refs.controlSell.setData(tradePoints(control, "SELL"));

      for (const series of [refs.primaryProbability, refs.primaryEquity, refs.primaryBuy, refs.primarySell]) series.applyOptions({ visible: visibleModels.primary });
      for (const series of [refs.controlProbability, refs.controlEquity, refs.controlBuy, refs.controlSell]) series.applyOptions({ visible: visibleModels.control });

      if (candles.length) {
        if (followLiveRef.current) refs.chart.timeScale().scrollToRealTime();
        else if (visibleRange) refs.chart.timeScale().setVisibleRange(visibleRange);
        else refs.chart.timeScale().fitContent();
      }
    } catch (chartError) {
      setError(chartError instanceof Error ? `Chart update failed: ${chartError.message}` : "Chart update failed");
    }
  }, [payload, primary, control, visibleModels]);

  const quoteLast = finiteNumber(payload?.quote?.last);
  const quoteOpen = finiteNumber(payload?.quote?.open);
  const quoteChange = quoteLast !== null && quoteOpen !== null && quoteOpen !== 0 ? quoteLast / quoteOpen - 1 : null;
  const refreshed = payload?.generated_at ? new Date(payload.generated_at).toLocaleTimeString() : "—";
  const auditRows = useMemo(() => [...(primary?.history ?? []), ...(control?.history ?? [])].sort((a, b) => b.feature_timestamp - a.feature_timestamp).slice(0, 20), [primary?.history, control?.history]);

  const renderModel = (model: ForwardModel | null, latestRow: ForwardRecord | null, modelClass: string) => (
    <article className={`model-card ${modelClass}`}>
      <div className="model-card-top"><span>{model?.display_name ?? "V3 model"}</span><span>{modelState(model)}</span></div>
      <strong className={signalClass(latestRow?.signal)}>{latestRow?.signal ?? "WAIT"}</strong>
      <small>{latestRow ? `p=${formatProbability(latestRow.calibrated_probability)} · net EV ${latestRow.decision_ev == null ? "commitment" : formatPercent(latestRow.decision_ev, 3)}` : "No forward inference has been recorded yet"}</small>
      <div className="model-mini-metrics">
        <span>Equity <b>€{formatPrice(model?.portfolio.portfolio_value_usdt)}</b></span>
        <span>P/L <b>{formatPercent(model?.portfolio.total_return)}</b></span>
        <span>Trades <b>{model?.performance.trade_count ?? 0}</b></span>
        <span>Next decision <b>{nextDecision(model)}</b></span>
      </div>
    </article>
  );

  return (
    <section className="terminal-shell">
      <div className="terminal-heading">
        <div>
          <p className="eyebrow">LIVE MARKET + FROZEN V3 FORWARD TEST</p>
          <div className="terminal-title-row"><h2>BTCUSDT strategy performance terminal</h2><span className="live-chip"><span className="status-dot online" /> PAPER · 1H</span></div>
          <p className="terminal-subtitle">Candles are live CoinEx market data. The two overlays are the frozen V3 paper strategies. BUY/SELL dots are actual simulated fills generated by the corresponding model policy.</p>
        </div>
        <div className="terminal-market-price"><span>BTC / USDT</span><strong>{quoteLast !== null ? `$${formatPrice(quoteLast)}` : "—"}</strong><small className={quoteChange !== null ? (quoteChange >= 0 ? "pos" : "neg") : "neutral"}>{quoteChange !== null ? formatPercent(quoteChange) : "Market unavailable"}</small></div>
      </div>

      <div className="model-strip">
        {renderModel(primary, primaryLatest, "xgb-card")}
        {renderModel(control, controlLatest, "lstm-card")}
        <article className="model-card consensus-card">
          <div className="model-card-top"><span>Execution audit</span><span>NO REAL ORDERS</span></div>
          <strong>{(primary?.performance.trade_count ?? 0) + (control?.performance.trade_count ?? 0)} PAPER FILLS</strong>
          <small>Every fill stays attached to the model that generated it. A BUY/SELL appears only when the frozen EV policy produced that signal.</small>
          <div className="model-mini-metrics"><span>12h artifact <b>{primary?.artifact_ready ? "READY" : "MISSING"}</b></span><span>3h artifact <b>{control?.artifact_ready ? "READY" : "MISSING"}</b></span></div>
        </article>
      </div>

      <article className="terminal-chart-panel">
        <div className="chart-toolbar">
          <div className="chart-toolbar-group">
            <button className={`series-toggle ${visibleModels.primary ? "active xgb" : ""}`} onClick={() => setVisibleModels((current) => ({ ...current, primary: !current.primary }))}>12h Economic</button>
            <button className={`series-toggle ${visibleModels.control ? "active lstm" : ""}`} onClick={() => setVisibleModels((current) => ({ ...current, control: !current.control }))}>3h Control</button>
          </div>
          <div className="chart-toolbar-group range-buttons">{(["1D", "3D", "1W", "1M", "ALL"] as ChartRange[]).map((item) => <button key={item} className={range === item ? "active" : ""} onClick={() => applyRange(item)}>{item}</button>)}</div>
          <button className={`go-live-button ${followLive ? "active" : ""}`} onClick={goLive}><span className="status-dot online" /> {followLive ? "FOLLOWING LIVE" : "GO LIVE"}</button>
        </div>
        <div className="chart-pane-labels" aria-hidden="true"><span>BTC PRICE + PAPER BUY/SELL FILLS</span><span>CALIBRATED EVENT PROBABILITY</span><span>PAPER EQUITY (€ EQUIVALENT)</span></div>
        <div className="terminal-chart-wrap" onWheel={() => { followLiveRef.current = false; setFollowLive(false); }} onPointerDown={() => { followLiveRef.current = false; setFollowLive(false); }}>
          <div ref={chartContainerRef} className="terminal-chart" />
          {loading && <div className="chart-overlay-message">Loading live market and V3 forward data…</div>}
          {!loading && !payload?.market?.candles?.length && <div className="chart-overlay-message">CoinEx market data is unavailable.</div>}
        </div>
        <div className="chart-footer-row"><span>{error ? `Data warning: ${error}` : `Updated ${refreshed}`}</span><span>Dots on price pane = simulated paper fills · probability and equity update hourly</span><span className="neutral">Model features: Binance · paper fill reference: CoinEx</span></div>
      </article>

      <div className="prediction-lower-grid">
        <article className="panel prediction-history-panel">
          <div className="panel-heading"><div><p className="eyebrow">DECISION AUDIT</p><h2>Why each model did or did not trade</h2></div><span className="placeholder-chip">LATEST 20</span></div>
          {auditRows.length ? (
            <div className="table-wrap"><table><thead><tr><th>Hour</th><th>Model</th><th>Probability</th><th>Net EV</th><th>Signal</th><th>Deal?</th><th>BTC price</th></tr></thead><tbody>{auditRows.map((row) => <tr key={`${row.model_id}-${row.feature_timestamp}`} title={row.reason}><td className="muted">{formatDateTime(row.feature_timestamp)}</td><td>{row.model_id === PRIMARY_ID ? "12h Economic" : "3h Control"}</td><td>{formatProbability(row.calibrated_probability)}</td><td>{row.decision_ev == null ? "commitment" : formatPercent(row.decision_ev, 3)}</td><td className={signalClass(row.signal)}><strong>{row.signal}</strong></td><td>{row.trade ? "YES · PAPER FILL" : "NO"}</td><td>${formatPrice(row.paper_market_price)}</td></tr>)}</tbody></table></div>
          ) : <div className="empty-state">No forward model decisions yet. This table will populate automatically after the frozen artifacts are installed and the hourly worker begins inference.</div>}
        </article>

        <article className="panel model-health-panel">
          <div className="panel-heading"><div><p className="eyebrow">FORWARD READINESS</p><h2>Frozen strategy status</h2></div></div>
          <div className="health-check-list">
            <div><span>Live market stream</span><strong className={payload?.market?.candles?.length ? "pos" : "neg"}>{payload?.market?.candles?.length ? "ONLINE" : "OFFLINE"}</strong></div>
            <div><span>V3 12h frozen artifact</span><strong>{primary?.artifact_ready ? "READY" : "MISSING"}</strong></div>
            <div><span>V3 3h frozen artifact</span><strong>{control?.artifact_ready ? "READY" : "MISSING"}</strong></div>
            <div><span>Forward inference</span><strong>{primary?.history.length || control?.history.length ? "ACTIVE" : "WAITING"}</strong></div>
            <div><span>Trading execution</span><strong className="neutral">PAPER ONLY</strong></div>
          </div>
          <p className="model-health-note">Historical V3 research statistics remain separate from this prospective forward curve. The equity chart above is built only from paper fills and hourly forward records generated after deployment.</p>
        </article>
      </div>
    </section>
  );
}
