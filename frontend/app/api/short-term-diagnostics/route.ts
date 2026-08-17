import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function repoRoot(): string {
  return path.resolve(process.cwd(), "..");
}

function readJsonl(name: string, limit = 2000): JsonRecord[] {
  const file = path.join(repoRoot(), "state", "short_term", name);
  if (!existsSync(file)) return [];
  try {
    const rows: JsonRecord[] = [];
    const lines = readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).slice(-limit);
    for (const line of lines) {
      try {
        const value = JSON.parse(line) as unknown;
        if (isRecord(value)) rows.push(value);
      } catch {
        // Ignore a partially-written telemetry line and keep diagnostics usable.
      }
    }
    return rows;
  } catch {
    return [];
  }
}

export async function GET() {
  return NextResponse.json(
    {
      generated_at: new Date().toISOString(),
      decisions: readJsonl("decision_diagnostics.jsonl"),
      outcomes: readJsonl("decision_outcomes.jsonl"),
      notes: [
        "Shadow policies are diagnostics only and never place paper or real orders.",
        "Shadow return uses a fixed 2-hour exit and subtracts the configured round-trip cost.",
        "The current short-term baselines are long-only.",
      ],
    },
    { headers: { "Cache-Control": "no-store, max-age=0" } },
  );
}
