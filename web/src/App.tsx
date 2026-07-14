import { useEffect, useState } from "react";
import { getHealth, getMeta } from "./lib/api";
import type {
  HealthResponse,
  MetaResponse,
  ApiError,
  P12Model,
  Provenance,
  ScoreResponse,
} from "./types";
import { TopBar } from "./components/TopBar";
import { SideNav, type NavKey } from "./components/SideNav";
import { HonestyBanner } from "./components/HonestyBanner";
import { PagePlaceholder } from "./pages/Placeholder";
import { About } from "./pages/About";
import { Upload } from "./pages/Upload";
import { Cohorts } from "./pages/Cohorts";
import { Members } from "./pages/Members";

const DEFAULT_MODELS: P12Model[] = ["xgboost", "lightgbm", "logreg"];

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<ApiError | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  const [meta, setMeta] = useState<MetaResponse | null>(null);

  const [model, setModel] = useState<P12Model>("xgboost");
  const [nav, setNav] = useState<NavKey>("upload");

  // The most recent scored result. Lifted here so Cohorts/Members can read it.
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const hasResult = result != null;

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const h = await getHealth();
        if (alive) setHealth(h);
      } catch (e) {
        if (alive) setHealthError(e as ApiError);
      } finally {
        if (alive) setHealthLoading(false);
      }
      try {
        const m = await getMeta();
        if (alive) {
          setMeta(m);
          setModel(m.default_p12_model ?? "xgboost");
        }
      } catch {
        /* meta failure is non-fatal for the shell; banner just stays empty */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const models: P12Model[] = health?.p12_models ?? meta?.p12_models ?? DEFAULT_MODELS;
  const provenance: Provenance = meta?.provenance ?? "synthetic";

  const onScored = (res: ScoreResponse) => {
    setResult(res);
    setNav("cohorts");
  };

  return (
    <div className="flex h-full flex-col">
      <TopBar
        health={health}
        healthError={healthError}
        healthLoading={healthLoading}
        provenance={provenance}
        models={models}
        model={model}
        onModelChange={setModel}
      />
      <HonestyBanner caveat={meta?.caveat ?? null} />

      <div className="flex min-h-0 flex-1">
        <SideNav active={nav} onChange={setNav} hasResult={hasResult} />
        <main className="min-w-0 flex-1 overflow-auto">
          {nav === "upload" && (
            <Upload
              meta={meta}
              model={model}
              models={models}
              onModelChange={setModel}
              onScored={onScored}
            />
          )}
          {nav === "cohorts" &&
            (result ? (
              <Cohorts result={result} />
            ) : (
              <PagePlaceholder
                title="Cohorts"
                subtitle="Upload a dataset to see the age × gender risk view."
              />
            ))}
          {nav === "members" && (
            <Members result={result} meta={meta} model={model} />
          )}
          {nav === "about" && <About meta={meta} />}
        </main>
      </div>
    </div>
  );
}
