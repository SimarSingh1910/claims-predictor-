// Upload page — drop a CSV of AHC records, preview it, check mandatory fields,
// pick the p12 model, and score. On success the parent lifts the result and
// navigates to Cohorts; on a 400 we render the EXACT validation errors the API
// returned (missing / unknown columns) rather than a generic message.
import { useCallback, useRef, useState } from "react";
import { sampleUrl, postScore } from "../lib/api";
import { parseCsvPreview, type CsvPreview } from "../lib/csv";
import { MODEL_LABEL, MODEL_RATIONALE } from "../lib/models";
import type {
  MetaResponse,
  P12Model,
  ScoreResponse,
  ApiError,
} from "../types";
import {
  Upload as UploadIcon,
  Download,
  FileText,
  X,
  CheckCircle2,
  XCircle,
} from "../components/icons";

interface Props {
  meta: MetaResponse | null;
  model: P12Model;
  models: P12Model[];
  onModelChange: (m: P12Model) => void;
  onScored: (result: ScoreResponse) => void;
}

interface ValidationDetail {
  error?: string;
  missing_columns?: string[];
  unknown_columns?: string[];
}

export function Upload({ meta, model, models, onModelChange, onScored }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [dragging, setDragging] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const mandatory = meta?.mandatory_fields ?? [];

  const takeFile = useCallback((f: File) => {
    setError(null);
    setFile(f);
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      setPreview(parseCsvPreview(text, 5));
    };
    reader.readAsText(f);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) takeFile(f);
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const score = async () => {
    if (!file) return;
    setScoring(true);
    setError(null);
    try {
      const res = await postScore(file, model);
      onScored(res); // parent stores result + navigates to Cohorts
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setScoring(false);
    }
  };

  // Present/missing mandatory columns against the detected header (client-side
  // courtesy; the backend is authoritative on submit).
  const detected = new Set(preview?.columns ?? []);
  const mandatoryStatus = mandatory.map((f) => ({
    field: f,
    present: detected.has(f),
  }));
  const allMandatoryPresent =
    preview != null && mandatoryStatus.every((m) => m.present);

  const validation =
    error && error.kind === "validation"
      ? (error.detail as ValidationDetail)
      : null;

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      <div>
        <h1 className="text-lg font-semibold text-slate-100">Upload dataset</h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload a CSV of employee Annual Health Checkup records. Every member is
          scored individually, then aggregated into age × gender cohorts.
        </p>
      </div>

      {/* ---- drop zone ---- */}
      {!file && (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={
            "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-16 text-center transition " +
            (dragging
              ? "border-brand bg-brand/10"
              : "border-surface-border bg-surface-panel/40 hover:border-slate-500")
          }
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-raised ring-1 ring-surface-border">
            <UploadIcon className="h-5 w-5 text-slate-300" />
          </div>
          <div>
            <div className="text-sm font-medium text-slate-200">
              Drop employee AHC data here, or browse
            </div>
            <div className="mt-1 text-xs text-slate-500">CSV up to ~100k rows</div>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) takeFile(f);
            }}
          />
          <a
            href={sampleUrl()}
            onClick={(e) => e.stopPropagation()}
            className="mt-1 inline-flex items-center gap-1.5 rounded-md bg-surface-raised px-3 py-1.5 text-xs text-slate-300 ring-1 ring-surface-border hover:text-slate-100"
          >
            <Download className="h-3.5 w-3.5" /> Download sample dataset
          </a>
        </div>
      )}

      {/* ---- file selected: preview + validation ---- */}
      {file && (
        <>
          <div className="flex items-center justify-between rounded-lg border border-surface-border bg-surface-panel px-4 py-3">
            <div className="flex items-center gap-2.5">
              <FileText className="h-4 w-4 text-slate-400" />
              <div>
                <div className="text-sm text-slate-200">{file.name}</div>
                <div className="text-[11px] text-slate-500">
                  {preview
                    ? `${preview.rowCount.toLocaleString()} rows · ${preview.columns.length} columns`
                    : "reading…"}
                </div>
              </div>
            </div>
            <button
              onClick={clearFile}
              className="rounded-md p-1.5 text-slate-400 hover:bg-surface-raised hover:text-slate-200"
              title="Remove file"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* mandatory-field validation panel */}
          {preview && (
            <div className="rounded-lg border border-surface-border bg-surface-panel p-4">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-200">
                  Mandatory fields
                </h2>
                <span
                  className={
                    "text-xs " +
                    (allMandatoryPresent ? "text-emerald-400" : "text-amber-400")
                  }
                >
                  {allMandatoryPresent
                    ? "all present"
                    : "missing fields — members without these are refused"}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {mandatoryStatus.map(({ field, present }) => (
                  <span
                    key={field}
                    className={
                      "inline-flex items-center gap-1.5 rounded px-2 py-1 font-mono text-[11px] ring-1 " +
                      (present
                        ? "bg-emerald-500/10 text-emerald-300 ring-emerald-500/25"
                        : "bg-red-500/10 text-red-300 ring-red-500/25")
                    }
                  >
                    {present ? (
                      <CheckCircle2 className="h-3 w-3" />
                    ) : (
                      <XCircle className="h-3 w-3" />
                    )}
                    {field}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* first-5-rows preview */}
          {preview && preview.rows.length > 0 && (
            <div className="rounded-lg border border-surface-border bg-surface-panel p-4">
              <h2 className="mb-2 text-sm font-semibold text-slate-200">
                Preview{" "}
                <span className="font-normal text-slate-500">
                  · first {preview.rows.length} rows
                </span>
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px]">
                  <thead className="text-slate-400">
                    <tr>
                      {preview.columns.map((c) => (
                        <th
                          key={c}
                          className="whitespace-nowrap px-2 py-1 font-mono font-normal"
                        >
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="text-slate-300">
                    {preview.rows.map((row, i) => (
                      <tr key={i} className="border-t border-surface-border">
                        {preview.columns.map((_, j) => (
                          <td
                            key={j}
                            className="whitespace-nowrap px-2 py-1 tabular-nums"
                          >
                            {row[j] ?? ""}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ---- model selector + rationale ---- */}
      <div className="rounded-lg border border-surface-border bg-surface-panel p-4">
        <div className="flex items-center gap-3">
          <label className="text-sm font-semibold text-slate-200">
            Frequency model
          </label>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value as P12Model)}
            className="rounded-md border border-surface-border bg-surface-raised px-2 py-1 text-xs text-slate-200 outline-none focus:border-brand"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {MODEL_LABEL[m] ?? m}
                {m === "xgboost" ? " (default)" : ""}
              </option>
            ))}
          </select>
        </div>
        <p className="mt-2 text-xs text-slate-400">{MODEL_RATIONALE[model]}</p>
      </div>

      {/* ---- validation errors from a 400 ---- */}
      {validation && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm">
          <div className="font-semibold text-red-300">
            {validation.error ?? "Validation failed"}
          </div>
          {validation.missing_columns && validation.missing_columns.length > 0 && (
            <div className="mt-2 text-red-200/90">
              <span className="text-xs uppercase tracking-wide text-red-300/70">
                Missing mandatory columns
              </span>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {validation.missing_columns.map((c) => (
                  <span
                    key={c}
                    className="rounded bg-red-500/15 px-1.5 py-0.5 font-mono text-[11px]"
                  >
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}
          {validation.unknown_columns && validation.unknown_columns.length > 0 && (
            <div className="mt-2 text-red-200/90">
              <span className="text-xs uppercase tracking-wide text-red-300/70">
                Unknown columns (not in the AHC feature set)
              </span>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {validation.unknown_columns.map((c) => (
                  <span
                    key={c}
                    className="rounded bg-red-500/15 px-1.5 py-0.5 font-mono text-[11px]"
                  >
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {error && error.kind !== "validation" && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          {error.kind === "network"
            ? "Backend unreachable — is uvicorn running on :8000?"
            : `Scoring failed: ${error.message}`}
        </div>
      )}

      {/* ---- score ---- */}
      <div className="flex items-center justify-end gap-3">
        {file && (
          <span className="text-xs text-slate-500">
            Members missing a mandatory field are refused and excluded from cohort
            statistics — never imputed.
          </span>
        )}
        <button
          disabled={!file || scoring}
          onClick={score}
          className={
            "rounded-md px-4 py-2 text-sm font-medium transition " +
            (!file || scoring
              ? "cursor-not-allowed bg-surface-raised text-slate-500"
              : "bg-brand text-white hover:bg-blue-500")
          }
        >
          {scoring ? "Scoring…" : "Score dataset"}
        </button>
      </div>
    </div>
  );
}
