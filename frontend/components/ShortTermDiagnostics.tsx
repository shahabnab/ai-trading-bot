"use client";

import { useEffect, useMemo, useState } from "react";

type JsonRecord = Record<string, unknown>;
type Payload = {
  generated_at?: string;
  decisions?: JsonRecord[];
  outcomes?: JsonRecord[];
  notes?: string[];
};

type PolicySummary = {
  name: string;
  threshold: number;
  signals: number;
  wins: number;
  netSum: number;
};

type BucketSummary = {
  label: string;
  samples: number;
  positive: number;
  returnSum: number;
};

function num(value: unknown): number {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}
function pct(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}
function bps(value: number | null): string {
  if (value == null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)} bps`;
}
function confidenceBucket(value: number): string {
  if (value < 0.5) return "<50%";
  if (value < 0.6) return "50–60%";
  if (value < 0.7) return "60–70%";
  if (value < 0.8) return "70–80%";
  if (value < 0.9) return "80–90%";
  return "90–100%";
}
function labelPolicy(name: string): string {
  if (name === "official") return "Official";
  if (name === "raw_setup") return "Raw setup";
  return name.replace("shadow_", "Shadow ");
}

const card: React.CSSProperties = {
  border: "1px solid rgba(148,163,184,.22)",
  borderRadius: 14,
  padding: 16,
  background: "rgba(15,23,42,.52)",
};
const cell: React.CSSProperties = {
  padding: "9px 10px",
  borderBottom: "1px solid rgba(148,163,184,.15)",
  textAlign: "left",
};

export default function ShortTermDiagnostics() {
  const [payload, setPayload] = useState<Payload | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const response = await fetch("/api/short-term-diagnostics", { cache: "no-store" });
        if (!response.ok) return;
        const value = (await response.json()) as Payload;
        if (alive) setPayload(value);
      } catch {
        // The main trading dashboard must stay usable if diagnostics are unavailable.
      }
    };
    void load();
    const timer = window.setInterval(() => void load(), 60_000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const summaries = useMemo(() => {
    const decisions = payload?.decisions ?? [];
    const outcomes = payload?.outcomes ?? [];
    const ids = [...new Set(decisions.map((row) => String(row.model_id ?? "")).filter(Boolean))].sort();

    return ids.map((modelId) => {
      const modelDecisions = decisions.filter((row) => String(row.model_id ?? "") === modelId);
      const modelOutcomes = outcomes.filter((row) => String(row.model_id ?? "") === modelId);
      const classifications: Record<string, number> = {};
      const policies = new Map<string, PolicySummary>();
      const buckets = new Map<string, BucketSummary>();
      const bucketOrder = ["<50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"];

      for (const row of modelOutcomes) {
        const classification = String(row.classification ?? "UNKNOWN");
        classifications[classification] = (classifications[classification] ?? 0) + 1;

        const returns = typeof row.returns_bps === "object" && row.returns_bps !== null ? row.returns_bps as JsonRecord : {};
        const return2h = num(returns["2h"]);
        const bucketLabel = confidenceBucket(num(row.confidence));
        const bucket = buckets.get(bucketLabel) ?? { label: bucketLabel, samples: 0, positive: 0, returnSum: 0 };
        bucket.samples += 1;
        bucket.returnSum += return2h;
        if (return2h > 0) bucket.positive += 1;
        buckets.set(bucketLabel, bucket);

        const rawResults = Array.isArray(row.shadow_results) ? row.shadow_results : [];
        for (const raw of rawResults) {
          if (typeof raw !== "object" || raw === null || Array.isArray(raw)) continue;
          const result = raw as JsonRecord;
          const name = String(result.name ?? "unknown");
          const policy = policies.get(name) ?? { name, threshold: num(result.threshold_bps), signals: 0, wins: 0, netSum: 0 };
          if (Boolean(result.would_enter)) {
            policy.signals += 1;
            policy.netSum += num(result.net_return_2h_bps);
            if (Boolean(result.win_after_cost)) policy.wins += 1;
          }
          policies.set(name, policy);
        }
      }

      const edgeGaps = modelDecisions.map((row) => num(row.edge_gap_bps));
      const policyOrder: Record<string, number> = { official: 0, shadow_55: 1, shadow_45: 2, shadow_35: 3, raw_setup: 99 };
      return {
        modelId,
        displayName: String(modelDecisions.at(-1)?.display_name ?? modelId),
        decisions: modelDecisions.length,
        resolved: modelOutcomes.length,
        pending: Math.max(0, modelDecisions.length - modelOutcomes.length),
        setupCandidates: modelDecisions.filter((row) => Boolean(row.setup_ready)).length,
        officialSignals: modelDecisions.filter((row) => String(row.decision_action ?? "") === "ENTER_LONG").length,
        missedLong: classifications.MISSED_LONG ?? 0,
        avoidedLoss: classifications.AVOIDED_LOSS ?? 0,
        goodHold: classifications.GOOD_HOLD ?? 0,
        noSetup: classifications.NO_SETUP ?? 0,
        avgEdgeGap: edgeGaps.length ? edgeGaps.reduce((a, b) => a + b, 0) / edgeGaps.length : null,
        policies: [...policies.values()].sort((a, b) => (policyOrder[a.name] ?? 50) - (policyOrder[b.name] ?? 50)),
        buckets: bucketOrder.map((label) => buckets.get(label) ?? { label, samples: 0, positive: 0, returnSum: 0 }),
      };
    });
  }, [payload]);

  return (
    <section style={{ display: "grid", gap: 14 }}>
      <article style={card}>
        <p className="eyebrow">SHORT-TERM DIAGNOSTICS · OBSERVATIONAL ONLY</p>
        <h2 style={{ marginTop: 4 }}>Is the brain weak, or are the brakes too strong?</h2>
        <p>
          Every completed 15-minute decision is now followed for 2 hours. The official rule stays unchanged while
          55/45/35-bps and raw-setup shadow policies are scored without placing orders. Stored short-term
          “confidence” is treated here as a <b>setup score</b>, not a calibrated probability of profit.
        </p>
        <p><small>Shadow P/L below is a diagnostic fixed-2h return in basis points after the configured round-trip cost; it is not the paper ledger P/L.</small></p>
      </article>

      {!payload ? (
        <article style={card}>Loading short-term diagnostics…</article>
      ) : summaries.length === 0 ? (
        <article style={card}>Collecting new decision telemetry. Outcome rows will appear after the first 2-hour window matures.</article>
      ) : summaries.map((summary) => (
        <article key={summary.modelId} style={{ ...card, display: "grid", gap: 14 }}>
          <div>
            <small>{summary.modelId}</small>
            <h3 style={{ margin: "4px 0" }}>{summary.displayName}</h3>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 10 }}>
            <div style={card}><small>DECISIONS</small><h3>{summary.decisions}</h3><span>{summary.resolved} resolved · {summary.pending} pending</span></div>
            <div style={card}><small>SETUP CANDIDATES</small><h3>{summary.setupCandidates}</h3><span>{summary.officialSignals} official entry signals</span></div>
            <div style={card}><small>MISSED LONG</small><h3>{summary.missedLong}</h3><span>2h close cleared official hurdle</span></div>
            <div style={card}><small>AVOIDED LOSS</small><h3>{summary.avoidedLoss}</h3><span>2h close moved against long by hurdle</span></div>
            <div style={card}><small>AVG EDGE GAP</small><h3>{bps(summary.avgEdgeGap)}</h3><span>edge proxy − official hurdle</span></div>
          </div>

          <div style={{ overflowX: "auto" }}>
            <h4>Shadow entry hurdle comparison</h4>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={cell}>Policy</th><th style={cell}>Threshold</th><th style={cell}>Signals</th><th style={cell}>Wins after cost</th><th style={cell}>Win rate</th><th style={cell}>Avg net / signal</th><th style={cell}>Sum net bps</th></tr></thead>
              <tbody>{summary.policies.length ? summary.policies.map((policy) => (
                <tr key={policy.name}>
                  <td style={cell}>{labelPolicy(policy.name)}</td>
                  <td style={cell}>{policy.name === "raw_setup" ? "none" : `${policy.threshold.toFixed(0)} bps`}</td>
                  <td style={cell}>{policy.signals}</td>
                  <td style={cell}>{policy.wins}</td>
                  <td style={cell}>{pct(policy.signals ? policy.wins / policy.signals : null)}</td>
                  <td style={cell}>{bps(policy.signals ? policy.netSum / policy.signals : null)}</td>
                  <td style={cell}>{bps(policy.netSum)}</td>
                </tr>
              )) : <tr><td style={cell} colSpan={7}>Waiting for mature outcomes.</td></tr>}</tbody>
            </table>
          </div>

          <div style={{ overflowX: "auto" }}>
            <h4>Setup-score calibration against the next 2h direction</h4>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={cell}>Setup score</th><th style={cell}>Samples</th><th style={cell}>Positive 2h</th><th style={cell}>Positive rate</th><th style={cell}>Avg 2h move</th></tr></thead>
              <tbody>{summary.buckets.map((bucket) => (
                <tr key={bucket.label}>
                  <td style={cell}>{bucket.label}</td>
                  <td style={cell}>{bucket.samples}</td>
                  <td style={cell}>{bucket.positive}</td>
                  <td style={cell}>{pct(bucket.samples ? bucket.positive / bucket.samples : null)}</td>
                  <td style={cell}>{bps(bucket.samples ? bucket.returnSum / bucket.samples : null)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <small>Current short-term baselines are long-only, so this analyzer intentionally does not label “missed short” opportunities.</small>
        </article>
      ))}
    </section>
  );
}
