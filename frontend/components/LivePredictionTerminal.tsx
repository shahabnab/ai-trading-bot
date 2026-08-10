"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type {
  MarketCandle,
  MarketKlinesResponse,
  MarketQuote,
  ModelPredictionHistory,
  ModelPredictionPoint,
  ModelPredictionSeries,
} from "../lib/api";

type DashboardPayload = {
  generated_at: string;
  quote: MarketQuote | null;
  market: MarketKlinesResponse | null;
  predictions: ModelPredictionHistory | null;
};

type ModelKey = "xgboost" | "lstm";
type ChartRange = "1D" | "3D" | "1W" | "1M" | "ALL";

type ChartRefs = {
  chart: any;
  candle: any;
  xgbPrice: any;
  lstmPrice: any;
  actualReturn: any;
  xgbReturn: any;
  lstmReturn: any;
  xgbError: any;
  lstmError: any;
};

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
  return number.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: number >= 1_000 ? 2 : 6,
  });
}

function formatPercent(value: unknown, digits = 2): string {
  const number = finiteNumber(value);
  if (number === null) return "—";
  const sign = number > 0 ? "+" : "";
  return `${sign}${(number * 100).toFixed(digits)}%`;
}

function formatMetricPercent(value: unknown): string {
  const number = finiteNumber(value);
  return number === null ? "—" : `${(number * 100).toFixed(2)}%`;
}

function formatDateTime(timestamp: number): string {
  return new Date(timestamp).toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function latestPoint(series?: ModelPredictionSeries): ModelPredictionPoint | null {
  const rows = series?.predictions ?? [];
  return rows.length > 0 ? rows[rows.length - 1] : null;
}

function modelState(series?: ModelPredictionSeries): string {
  if (!series) return "NOT TRAINED";
  return series.mode === "forward_live" ? "FORWARD LIVE" : "OOS HISTORY";
}

function modelSignal(point: ModelPredictionPoint | null): string {
  if (!point) return "—";
  if (point.predicted_log_return > 0) return "BULLISH";
  if (point.predicted_log_return < 0) return "BEARISH";
  return "FLAT";
}

function consensusLabel(xgb: ModelPredictionPoint | null, lstm: ModelPredictionPoint | null): string {
  if (!xgb || !lstm) return "INSUFFICIENT DATA";
  if (xgb.target_timestamp !== lstm.target_timestamp) return "UNALIGNED RUNS";
  const x = Math.sign(xgb.predicted_log_return);
  const l = Math.sign(lstm.predicted_log_return);
  if (x === 0 && l === 0) return "FLAT";
  if (x === l) return x > 0 ? "BULLISH AGREEMENT" : "BEARISH AGREEMENT";
  return "MODEL CONFLICT";
}

function seriesRows(series?: ModelPredictionSeries): ModelPredictionPoint[] {
  return series?.predictions ?? [];
}

function buildHistoryRows(predictions: ModelPredictionHistory | null) {
  const xgb = seriesRows(predictions?.models.xgboost);
  const lstm = seriesRows(predictions?.models.lstm);
  const merged = new Map<number, {
    timestamp: number;
    actual: number | null;
    xgb: ModelPredictionPoint | null;
    lstm: ModelPredictionPoint | null;
  }>();

  for (const point of xgb) {
    merged.set(point.target_timestamp, {
      timestamp: point.target_timestamp,
      actual: point.actual_simple_return,
      xgb: point,
      lstm: merged.get(point.target_timestamp)?.lstm ?? null,
    });
  }
  for (const point of lstm) {
    const existing = merged.get(point.target_timestamp);
    merged.set(point.target_timestamp, {
      timestamp: point.target_timestamp,
      actual: existing?.actual ?? point.actual_simple_return,
      xgb: existing?.xgb ?? null,
      lstm: point,
    });
  }

  return [...merged.values()].sort((a, b) => b.timestamp - a.timestamp);
}

export default function LivePredictionTerminal() {
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const chartRefs = useRef<ChartRefs | null>(null);
  const latestPayloadRef = useRef<DashboardPayload | null>(null);
  const followLiveRef = useRef(true);

  const [payload, setPayload] = useState<DashboardPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [followLive, setFollowLive] = useState(true);
  const [range, setRange] = useState<ChartRange>("1D");
  const [visibleModels, setVisibleModels] = useState<Record<ModelKey, boolean>>({
    xgboost: true,
    lstm: true,
  });

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
    if (!refs || candles.length === 0) return;
    const last = Number(candles[candles.length - 1]?.created_at);
    if (!Number.isFinite(last)) return;

    if (nextRange === "ALL") {
      refs.chart.timeScale().fitContent();
      followLiveRef.current = false;
      setFollowLive(false);
      return;
    }

    const days = nextRange === "1D" ? 1 : nextRange === "3D" ? 3 : nextRange === "1W" ? 7 : 30;
    refs.chart.timeScale().setVisibleRange({
      from: Math.floor((last - days * 86_400_000) / 1000),
      to: Math.floor(last / 1000) + 6 * 3600,
    });
    followLiveRef.current = nextRange === "1D";
    setFollowLive(nextRange === "1D");
  }, []);

  const goLive = useCallback(() => {
    followLiveRef.current = true;
    setFollowLive(true);
    setRange("1D");
    chartRefs.current?.chart.timeScale().scrollToRealtime();
  }, []);

  useEffect(() => {
    followLiveRef.current = followLive;
  }, [followLive]);

  useEffect(() => {
    let disposed = false;
    let localChart: any = null;

    void loadLightweightCharts()
      .then((LWC) => {
        if (disposed || !chartContainerRef.current || !LWC) return;

        const chart = LWC.createChart(chartContainerRef.current, {
          autoSize: true,
          height: 720,
          layout: {
            background: { type: LWC.ColorType.Solid, color: "#08131d" },
            textColor: "#8da2b4",
            attributionLogo: true,
            panes: {
              separatorColor: "#1c2b37",
              separatorHoverColor: "#334b5e",
              enableResize: true,
            },
          },
          grid: {
            vertLines: { color: "#12222e" },
            horzLines: { color: "#12222e" },
          },
          rightPriceScale: {
            borderColor: "#20313f",
            scaleMargins: { top: 0.08, bottom: 0.08 },
          },
          timeScale: {
            borderColor: "#20313f",
            timeVisible: true,
            secondsVisible: false,
            rightOffset: 6,
            barSpacing: 8,
            minBarSpacing: 2,
          },
          crosshair: {
            mode: LWC.CrosshairMode.Normal,
            vertLine: { color: "#64748b", width: 1, style: LWC.LineStyle.Dashed, labelBackgroundColor: "#334155" },
            horzLine: { color: "#64748b", width: 1, style: LWC.LineStyle.Dashed, labelBackgroundColor: "#334155" },
          },
          handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
          handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
        });
        localChart = chart;

        const candle = chart.addSeries(LWC.CandlestickSeries, {
          upColor: "#20c997",
          downColor: "#f05d7a",
          wickUpColor: "#20c997",
          wickDownColor: "#f05d7a",
          borderVisible: false,
          priceLineVisible: true,
        }, 0);

        const xgbPrice = chart.addSeries(LWC.LineSeries, {
          color: "#57a6ff",
          lineWidth: 2,
          lineStyle: LWC.LineStyle.Dashed,
          title: "XGBoost forecast",
          lastValueVisible: true,
          crosshairMarkerVisible: true,
        }, 0);
        const lstmPrice = chart.addSeries(LWC.LineSeries, {
          color: "#d28cff",
          lineWidth: 2,
          lineStyle: LWC.LineStyle.Dashed,
          title: "LSTM forecast",
          lastValueVisible: true,
          crosshairMarkerVisible: true,
        }, 0);

        const percentFormat = {
          type: "custom",
          minMove: 0.01,
          formatter: (value: number) => `${value.toFixed(2)}%`,
        };
        const actualReturn = chart.addSeries(LWC.LineSeries, {
          color: "#e6edf3",
          lineWidth: 2,
          title: "Actual return",
          priceFormat: percentFormat,
          lastValueVisible: false,
        }, 1);
        const xgbReturn = chart.addSeries(LWC.LineSeries, {
          color: "#57a6ff",
          lineWidth: 2,
          lineStyle: LWC.LineStyle.Dashed,
          title: "XGBoost return",
          priceFormat: percentFormat,
          lastValueVisible: false,
        }, 1);
        const lstmReturn = chart.addSeries(LWC.LineSeries, {
          color: "#d28cff",
          lineWidth: 2,
          lineStyle: LWC.LineStyle.Dashed,
          title: "LSTM return",
          priceFormat: percentFormat,
          lastValueVisible: false,
        }, 1);
        actualReturn.createPriceLine({
          price: 0,
          color: "#405366",
          lineWidth: 1,
          lineStyle: LWC.LineStyle.Dashed,
          axisLabelVisible: false,
          title: "",
        });

        const xgbError = chart.addSeries(LWC.LineSeries, {
          color: "#57a6ff",
          lineWidth: 2,
          title: "XGBoost error",
          priceFormat: percentFormat,
          lastValueVisible: false,
        }, 2);
        const lstmError = chart.addSeries(LWC.LineSeries, {
          color: "#d28cff",
          lineWidth: 2,
          title: "LSTM error",
          priceFormat: percentFormat,
          lastValueVisible: false,
        }, 2);
        xgbError.createPriceLine({
          price: 0,
          color: "#405366",
          lineWidth: 1,
          lineStyle: LWC.LineStyle.Dashed,
          axisLabelVisible: false,
          title: "",
        });

        const panes = chart.panes();
        if (panes[0]) panes[0].setHeight(430);
        if (panes[1]) panes[1].setHeight(160);
        if (panes[2]) panes[2].setHeight(130);

        chartRefs.current = {
          chart,
          candle,
          xgbPrice,
          lstmPrice,
          actualReturn,
          xgbReturn,
          lstmReturn,
          xgbError,
          lstmError,
        };
        if (latestPayloadRef.current) {
          setPayload({ ...latestPayloadRef.current });
        }

        chart.subscribeCrosshairMove((param: any) => {
          const tooltip = tooltipRef.current;
          if (!tooltip) return;
          if (!param.point || !param.time || param.point.x < 0 || param.point.y < 0) {
            tooltip.style.opacity = "0";
            return;
          }

          const candleData = param.seriesData.get(candle) as { open?: number; high?: number; low?: number; close?: number } | undefined;
          const xgbData = param.seriesData.get(xgbPrice) as { value?: number } | undefined;
          const lstmData = param.seriesData.get(lstmPrice) as { value?: number } | undefined;
          const actualReturnData = param.seriesData.get(actualReturn) as { value?: number } | undefined;
          const xgbReturnData = param.seriesData.get(xgbReturn) as { value?: number } | undefined;
          const lstmReturnData = param.seriesData.get(lstmReturn) as { value?: number } | undefined;

          const timestampSeconds = typeof param.time === "number" ? param.time : null;
          const timeText = timestampSeconds ? formatDateTime(timestampSeconds * 1000) : "Selected bar";
          tooltip.innerHTML = `
            <strong>${timeText}</strong>
            ${candleData?.close != null ? `<span>CoinEx close <b>${formatPrice(candleData.close)}</b></span>` : ""}
            ${xgbData?.value != null ? `<span>XGBoost price <b>${formatPrice(xgbData.value)}</b></span>` : ""}
            ${lstmData?.value != null ? `<span>LSTM price <b>${formatPrice(lstmData.value)}</b></span>` : ""}
            ${actualReturnData?.value != null ? `<span>Actual return <b>${actualReturnData.value.toFixed(3)}%</b></span>` : ""}
            ${xgbReturnData?.value != null ? `<span>XGBoost return <b>${xgbReturnData.value.toFixed(3)}%</b></span>` : ""}
            ${lstmReturnData?.value != null ? `<span>LSTM return <b>${lstmReturnData.value.toFixed(3)}%</b></span>` : ""}
          `;
          tooltip.style.opacity = "1";
          const left = Math.min(param.point.x + 18, Math.max(12, chartContainerRef.current!.clientWidth - 245));
          tooltip.style.left = `${left}px`;
          tooltip.style.top = `${Math.max(12, param.point.y - 40)}px`;
        });
      })
      .catch((chartError) => {
        if (!disposed) setError(chartError instanceof Error ? chartError.message : "Chart initialization failed");
      });

    return () => {
      disposed = true;
      chartRefs.current = null;
      if (localChart) localChart.remove();
    };
  }, []);

  useEffect(() => {
    const refs = chartRefs.current;
    if (!refs || !payload) return;

    const visibleRange = refs.chart.timeScale().getVisibleRange();
    const candles = (payload.market?.candles ?? [])
      .map((row: MarketCandle) => {
        const ms = Number(row.created_at);
        const open = Number(row.open);
        const high = Number(row.high);
        const low = Number(row.low);
        const close = Number(row.close);
        if (![ms, open, high, low, close].every(Number.isFinite)) return null;
        return { time: Math.floor(ms / 1000), open, high, low, close };
      })
      .filter((row): row is { time: number; open: number; high: number; low: number; close: number } => row !== null)
      .sort((a, b) => a.time - b.time);

    const xgb = seriesRows(payload.predictions?.models.xgboost);
    const lstm = seriesRows(payload.predictions?.models.lstm);
    const toLine = (rows: ModelPredictionPoint[], field: "predicted_price" | "predicted_log_return" | "error_log_return") =>
      rows.map((row) => ({
        time: Math.floor(row.target_timestamp / 1000),
        value: field === "predicted_price" ? row[field] : row[field] * 100,
      }));

    const actualByTime = new Map<number, number>();
    for (const point of xgb.length ? xgb : lstm) {
      actualByTime.set(point.target_timestamp, point.actual_log_return * 100);
    }

    refs.candle.setData(candles);
    refs.xgbPrice.setData(toLine(xgb, "predicted_price"));
    refs.lstmPrice.setData(toLine(lstm, "predicted_price"));
    refs.actualReturn.setData([...actualByTime.entries()].sort((a, b) => a[0] - b[0]).map(([timestamp, value]) => ({
      time: Math.floor(timestamp / 1000),
      value,
    })));
    refs.xgbReturn.setData(toLine(xgb, "predicted_log_return"));
    refs.lstmReturn.setData(toLine(lstm, "predicted_log_return"));
    refs.xgbError.setData(toLine(xgb, "error_log_return"));
    refs.lstmError.setData(toLine(lstm, "error_log_return"));

    refs.xgbPrice.applyOptions({ visible: visibleModels.xgboost });
    refs.xgbReturn.applyOptions({ visible: visibleModels.xgboost });
    refs.xgbError.applyOptions({ visible: visibleModels.xgboost });
    refs.lstmPrice.applyOptions({ visible: visibleModels.lstm });
    refs.lstmReturn.applyOptions({ visible: visibleModels.lstm });
    refs.lstmError.applyOptions({ visible: visibleModels.lstm });

    if (candles.length > 0) {
      if (followLiveRef.current) {
        refs.chart.timeScale().scrollToRealtime();
      } else if (visibleRange) {
        refs.chart.timeScale().setVisibleRange(visibleRange);
      } else {
        refs.chart.timeScale().fitContent();
      }
    }
  }, [payload, visibleModels]);

  const xgbSeries = payload?.predictions?.models.xgboost;
  const lstmSeries = payload?.predictions?.models.lstm;
  const xgbLatest = latestPoint(xgbSeries);
  const lstmLatest = latestPoint(lstmSeries);
  const historyRows = useMemo(() => buildHistoryRows(payload?.predictions ?? null).slice(0, 20), [payload?.predictions]);

  const quoteLast = finiteNumber(payload?.quote?.last);
  const quoteOpen = finiteNumber(payload?.quote?.open);
  const quoteChange = quoteLast !== null && quoteOpen !== null && quoteOpen !== 0 ? quoteLast / quoteOpen - 1 : null;
  const refreshed = payload?.generated_at ? new Date(payload.generated_at).toLocaleTimeString() : "—";
  const consensus = consensusLabel(xgbLatest, lstmLatest);

  const toggleModel = (model: ModelKey) => {
    setVisibleModels((current: Record<ModelKey, boolean>) => ({ ...current, [model]: !current[model] }));
  };

  return (
    <section className="terminal-shell">
      <div className="terminal-heading">
        <div>
          <p className="eyebrow">LIVE MARKET + AI FORECASTS</p>
          <div className="terminal-title-row">
            <h2>BTCUSDT intelligence terminal</h2>
            <span className="live-chip"><span className="status-dot online" /> COINEX · 1H</span>
          </div>
          <p className="terminal-subtitle">
            CoinEx candles refresh every 10 seconds. Model overlays are immutable walk-forward OOS predictions from the latest completed XGBoost/LSTM runs.
          </p>
        </div>
        <div className="terminal-market-price">
          <span>BTC / USDT</span>
          <strong>{quoteLast !== null ? `$${formatPrice(quoteLast)}` : "—"}</strong>
          <small className={quoteChange !== null ? (quoteChange >= 0 ? "pos" : "neg") : "neutral"}>
            {quoteChange !== null ? formatPercent(quoteChange) : "Market unavailable"}
          </small>
        </div>
      </div>

      <div className="model-strip">
        <article className="model-card xgb-card">
          <div className="model-card-top"><span>XGBoost</span><span>{modelState(xgbSeries)}</span></div>
          <strong className={xgbLatest ? (xgbLatest.predicted_log_return >= 0 ? "pos" : "neg") : "neutral"}>
            {xgbLatest ? formatPercent(xgbLatest.predicted_simple_return) : "No run"}
          </strong>
          <small>{xgbLatest ? `${modelSignal(xgbLatest)} · implied $${formatPrice(xgbLatest.predicted_price)}` : "Train XGBoost to populate history"}</small>
          <div className="model-mini-metrics">
            <span>Dir acc <b>{formatMetricPercent(xgbSeries?.metrics.forecast?.direction_accuracy)}</b></span>
            <span>MAE <b>{formatMetricPercent(xgbSeries?.metrics.forecast?.mae_log_return)}</b></span>
            <span>Sharpe <b>{finiteNumber(xgbSeries?.metrics.strategy?.sharpe)?.toFixed(2) ?? "—"}</b></span>
          </div>
        </article>

        <article className="model-card lstm-card">
          <div className="model-card-top"><span>LSTM</span><span>{modelState(lstmSeries)}</span></div>
          <strong className={lstmLatest ? (lstmLatest.predicted_log_return >= 0 ? "pos" : "neg") : "neutral"}>
            {lstmLatest ? formatPercent(lstmLatest.predicted_simple_return) : "No run"}
          </strong>
          <small>{lstmLatest ? `${modelSignal(lstmLatest)} · implied $${formatPrice(lstmLatest.predicted_price)}` : "Train LSTM to populate history"}</small>
          <div className="model-mini-metrics">
            <span>Dir acc <b>{formatMetricPercent(lstmSeries?.metrics.forecast?.direction_accuracy)}</b></span>
            <span>MAE <b>{formatMetricPercent(lstmSeries?.metrics.forecast?.mae_log_return)}</b></span>
            <span>Sharpe <b>{finiteNumber(lstmSeries?.metrics.strategy?.sharpe)?.toFixed(2) ?? "—"}</b></span>
          </div>
        </article>

        <article className="model-card consensus-card">
          <div className="model-card-top"><span>Model consensus</span><span>NO FAKE CONFIDENCE</span></div>
          <strong>{consensus}</strong>
          <small>
            {xgbLatest && lstmLatest && xgbLatest.target_timestamp === lstmLatest.target_timestamp
              ? `Target ${formatDateTime(xgbLatest.target_timestamp)}`
              : "Latest runs must share the same target timestamp for consensus."}
          </small>
          <div className="model-mini-metrics">
            <span>XGB run <b>{xgbSeries?.run_id?.slice(-10) ?? "—"}</b></span>
            <span>LSTM run <b>{lstmSeries?.run_id?.slice(-10) ?? "—"}</b></span>
          </div>
        </article>
      </div>

      <article className="terminal-chart-panel">
        <div className="chart-toolbar">
          <div className="chart-toolbar-group">
            <button className={`series-toggle ${visibleModels.xgboost ? "active xgb" : ""}`} onClick={() => toggleModel("xgboost")}>XGBoost</button>
            <button className={`series-toggle ${visibleModels.lstm ? "active lstm" : ""}`} onClick={() => toggleModel("lstm")}>LSTM</button>
          </div>
          <div className="chart-toolbar-group range-buttons">
            {(["1D", "3D", "1W", "1M", "ALL"] as ChartRange[]).map((item) => (
              <button key={item} className={range === item ? "active" : ""} onClick={() => applyRange(item)}>{item}</button>
            ))}
          </div>
          <button className={`go-live-button ${followLive ? "active" : ""}`} onClick={goLive}>
            <span className="status-dot online" /> {followLive ? "FOLLOWING LIVE" : "GO LIVE"}
          </button>
        </div>

        <div className="chart-pane-labels" aria-hidden="true">
          <span>PRICE + IMPLIED FORECAST</span>
          <span>ACTUAL / PREDICTED NEXT-HOUR RETURN</span>
          <span>FORECAST ERROR</span>
        </div>

        <div
          className="terminal-chart-wrap"
          onWheel={() => { followLiveRef.current = false; setFollowLive(false); }}
          onPointerDown={() => { followLiveRef.current = false; setFollowLive(false); }}
        >
          <div ref={chartContainerRef} className="terminal-chart" />
          <div ref={tooltipRef} className="chart-crosshair-tooltip" />
          {loading && <div className="chart-overlay-message">Loading live market data…</div>}
          {!loading && !payload?.market?.candles?.length && <div className="chart-overlay-message">CoinEx market data is unavailable.</div>}
        </div>

        <div className="chart-footer-row">
          <span>{error ? `Data warning: ${error}` : `Updated ${refreshed}`}</span>
          <span>Wheel to zoom · drag to pan · drag price axis to rescale · pane dividers are resizable</span>
          <span className="neutral">Model data: Binance-derived OOS artifacts · Live candles: CoinEx</span>
        </div>
      </article>

      <div className="prediction-lower-grid">
        <article className="panel prediction-history-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">PREDICTION HISTORY</p>
              <h2>Resolved model forecasts</h2>
            </div>
            <span className="placeholder-chip">LATEST 20</span>
          </div>
          {historyRows.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Target</th>
                    <th>Actual</th>
                    <th>XGBoost</th>
                    <th>LSTM</th>
                    <th>XGB error</th>
                    <th>LSTM error</th>
                    <th>Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {historyRows.map((row) => {
                    const direction = row.xgb?.direction_correct === true && row.lstm?.direction_correct === true
                      ? "both correct"
                      : row.xgb?.direction_correct === false && row.lstm?.direction_correct === false
                        ? "both wrong"
                        : "mixed";
                    return (
                      <tr key={row.timestamp}>
                        <td className="muted">{formatDateTime(row.timestamp)}</td>
                        <td className={row.actual != null ? (row.actual >= 0 ? "pos" : "neg") : "neutral"}>{row.actual != null ? formatPercent(row.actual) : "—"}</td>
                        <td>{row.xgb ? formatPercent(row.xgb.predicted_simple_return) : "—"}</td>
                        <td>{row.lstm ? formatPercent(row.lstm.predicted_simple_return) : "—"}</td>
                        <td>{row.xgb ? formatMetricPercent(Math.abs(row.xgb.error_log_return)) : "—"}</td>
                        <td>{row.lstm ? formatMetricPercent(Math.abs(row.lstm.error_log_return)) : "—"}</td>
                        <td><span className={`history-direction ${direction.replace(" ", "-")}`}>{direction}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">Run XGBoost and LSTM walk-forward evaluation to populate model-vs-actual prediction history.</div>
          )}
        </article>

        <article className="panel model-health-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">MODEL HEALTH</p>
              <h2>Forward-readiness</h2>
            </div>
          </div>
          <div className="health-check-list">
            <div><span>Live market stream</span><strong className={payload?.market?.candles?.length ? "pos" : "neg"}>{payload?.market?.candles?.length ? "ONLINE" : "OFFLINE"}</strong></div>
            <div><span>XGBoost OOS artifact</span><strong>{xgbSeries ? "AVAILABLE" : "MISSING"}</strong></div>
            <div><span>LSTM OOS artifact</span><strong>{lstmSeries ? "AVAILABLE" : "MISSING"}</strong></div>
            <div><span>Deployment inference</span><strong className="neutral">NOT ENABLED</strong></div>
            <div><span>Trading execution</span><strong className="neutral">PAPER ONLY</strong></div>
          </div>
          <p className="model-health-note">
            Fold models are evaluation artifacts. The dashboard intentionally does not label them as live deployment forecasts until an active-model training/inference service is added.
          </p>
        </article>
      </div>
    </section>
  );
}
