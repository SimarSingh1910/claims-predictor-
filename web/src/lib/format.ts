// Unit-aware value formatting. Kept in one place so every MetricCard formats a
// slot's value identically. A null value NEVER reaches these functions with a
// fallback number — callers gate on availability first; these only format REAL
// values.
import type { SlotKey } from "../types";

export function formatProbability(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

// Indian digit grouping (lakh/crore): ₹1,24,500 — not ₹124,500.
const INR = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});
export function formatINR(v: number): string {
  return `₹${INR.format(Math.round(v))}`;
}

// The unit each slot renders in.
export const SLOT_UNIT: Record<SlotKey, "probability" | "INR"> = {
  p12: "probability",
  p24: "probability",
  p36: "probability",
  expected_cost: "INR",
};

export function formatSlotValue(slot: SlotKey, value: number): string {
  return SLOT_UNIT[slot] === "INR"
    ? formatINR(value)
    : formatProbability(value);
}
