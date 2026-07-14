// Small color-scale + interpolation helpers for the cohort matrix and charts.
// A sequential risk ramp on the dark ground: low risk = cool cyan, mid = amber,
// high = red. Kept dependency-free (no d3) so it works without npm.

type RGB = [number, number, number];

function hexToRgb(hex: string): RGB {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}
function rgbToHex([r, g, b]: RGB): string {
  const c = (n: number) => Math.round(n).toString(16).padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`;
}
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function lerpRgb(a: RGB, b: RGB, t: number): RGB {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

// Three-stop ramp.
const STOPS: { at: number; rgb: RGB }[] = [
  { at: 0.0, rgb: hexToRgb("#0e7490") }, // cyan-700
  { at: 0.5, rgb: hexToRgb("#f59e0b") }, // amber-500
  { at: 1.0, rgb: hexToRgb("#dc2626") }, // red-600
];

/** Map t in [0,1] to a hex color along the risk ramp. */
export function riskColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 0; i < STOPS.length - 1; i++) {
    const a = STOPS[i];
    const b = STOPS[i + 1];
    if (x <= b.at) {
      const local = (x - a.at) / (b.at - a.at);
      return rgbToHex(lerpRgb(a.rgb, b.rgb, local));
    }
  }
  return rgbToHex(STOPS[STOPS.length - 1].rgb);
}

/** Normalise a value into [0,1] given a domain; safe when min==max. */
export function normalize(v: number, min: number, max: number): number {
  if (max <= min) return 0.5;
  return (v - min) / (max - min);
}

/** Readable text color (near-black vs near-white) for a given ramp position. */
export function textOn(t: number): string {
  // The amber mid-band is light; use dark text there, light text at the ends.
  return t > 0.28 && t < 0.72 ? "#0b0f19" : "#f8fafc";
}
