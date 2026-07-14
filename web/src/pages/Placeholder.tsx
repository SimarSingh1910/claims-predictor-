// Lightweight page stubs for the scaffold step. Upload/scoring/cohort views are
// implemented in later steps; this keeps the nav wired and the shell reviewable.
import type { ReactNode } from "react";

export function PagePlaceholder({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children?: ReactNode;
}) {
  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold text-slate-100">{title}</h1>
      {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
      <div className="mt-6 rounded-lg border border-dashed border-surface-border bg-surface-panel/40 p-8 text-center text-sm text-slate-500">
        {children ?? "Coming in a later step."}
      </div>
    </div>
  );
}
