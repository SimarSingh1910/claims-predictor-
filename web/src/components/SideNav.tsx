// Left nav: Upload · Cohorts · Members · About. Cohorts/Members are disabled
// until a scored result exists (wired in a later step).
import { Upload, Grid3x3, Users, Info } from "./icons";
import type { ComponentType } from "react";

export type NavKey = "upload" | "cohorts" | "members" | "about";

const ITEMS: { key: NavKey; label: string; icon: ComponentType<{ className?: string }> }[] = [
  { key: "upload", label: "Upload", icon: Upload },
  { key: "cohorts", label: "Cohorts", icon: Grid3x3 },
  { key: "members", label: "Members", icon: Users },
  { key: "about", label: "About", icon: Info },
];

interface Props {
  active: NavKey;
  onChange: (k: NavKey) => void;
  hasResult: boolean;
}

export function SideNav({ active, onChange, hasResult }: Props) {
  return (
    <nav className="flex w-44 shrink-0 flex-col gap-1 border-r border-surface-border bg-surface-panel p-3">
      {ITEMS.map(({ key, label, icon: Icon }) => {
        // Cohorts needs a scored dataset. Members is always reachable — its
        // manual-entry scorer works with no upload.
        const gated = key === "cohorts" && !hasResult;
        const isActive = active === key;
        return (
          <button
            key={key}
            disabled={gated}
            onClick={() => onChange(key)}
            className={
              "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition " +
              (isActive
                ? "bg-brand/15 text-brand ring-1 ring-brand/30"
                : gated
                ? "cursor-not-allowed text-slate-600"
                : "text-slate-300 hover:bg-surface-raised")
            }
            title={gated ? "Upload a dataset first" : undefined}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        );
      })}
    </nav>
  );
}
