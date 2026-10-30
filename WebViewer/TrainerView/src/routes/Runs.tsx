import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { get, post, remove, useResource } from "./api";
import { Page, Panel, State } from "./shared";
import { MetricChart } from "./MetricChart";

type Metric = { [key: string]: string | number | null };
type Checkpoint = {
  id: string;
  created_at?: number;
  epoch?: number;
  step?: number;
  loss?: number;
  metrics: Record<string, unknown>;
  trainer?: string;
  models?: Record<string, string>;
  datasets?: Record<string, string>;
  hyperparameters?: Record<string, unknown>;
  committed?: boolean;
};
type Run = {
  id: string;
  name: string;
  trainer: string;
  status: string;
  created_at: number;
  started_at?: number;
  finished_at?: number;
  progress: number;
  current_metrics: Record<string, unknown>;
  performance?: Record<string, unknown>;
  device?: {
    type: string;
    id?: string;
    index?: number;
    name?: string;
    fallback?: { requested: string; selected: string; reason: string };
  };
  configuration: {
    models?: Record<string, string>;
    datasets?: Record<string, string>;
    hyperparameters?: Record<string, unknown>;
  };
  metrics?: Metric[];
  checkpoints?: Checkpoint[];
  error?: string;
};
type RunsPayload = { runs: Run[] };
type RunPayload = Run;
type LegacyRun = {
  id: string;
  name: string;
  model: string;
  checkpoint: string;
  status: string;
  metadata: Record<string, unknown>;
  metrics: Record<string, number>;
};
type LegacyPayload = { runs: LegacyRun[] };
const EMPTY_METRICS: Metric[] = [];

function RunTerminal({ runId }: { runId: string }) {
  const [output, setOutput] = useState("");
  const [connected, setConnected] = useState(true);
  const [connectionError, setConnectionError] = useState("");
  const offset = useRef(0);
  const viewport = useRef<HTMLPreElement>(null);
  const followTail = useRef(true);

  useEffect(() => {
    let alive = true;
    let timer = 0;
    let busy = false;
    offset.current = 0;
    const poll = async () => {
      if (busy) return;
      busy = true;
      try {
        let combined = "";
        let nextOffset = offset.current;
        // Drain historical output in bounded chunks on open, then poll incrementally.
        do {
          const chunk = await get<{ content: string; next_offset: number }>(
            `/runs/${encodeURIComponent(runId)}/terminal?offset=${nextOffset}`,
          );
          combined += chunk.content;
          nextOffset = chunk.next_offset;
          if (chunk.content.length < 256 * 1024) break;
        } while (alive);
        if (alive) {
          offset.current = nextOffset;
          if (combined) setOutput((current) => current + combined);
          setConnected(true);
          setConnectionError("");
        }
      } catch (cause) {
        if (alive) {
          setConnected(false);
          setConnectionError((cause as Error).message);
        }
      } finally {
        busy = false;
        if (alive) timer = window.setTimeout(poll, 900);
      }
    };
    void poll();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [runId]);

  useEffect(() => {
    const element = viewport.current;
    if (element && followTail.current) element.scrollTop = element.scrollHeight;
  }, [output]);

  return (
    <Panel
      title="Terminal output"
      meta={connected ? (output ? "LIVE / PERSISTED" : "WAITING FOR OUTPUT") : "RECONNECTING"}
    >
      <pre
        className="run-terminal"
        ref={viewport}
        onScroll={(event) => {
          const element = event.currentTarget;
          followTail.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48;
        }}
      >
        {output || "No terminal output yet."}
      </pre>
      {connectionError && <div className="terminal-error">{connectionError}</div>}
    </Panel>
  );
}

function elapsed(run: Run) {
  const seconds = Math.max(
    0,
    (run.finished_at ?? Date.now() / 1000) - (run.started_at ?? run.created_at),
  );
  return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
}

export function Runs() {
  const { modelId, runId } = useParams();
  const navigate = useNavigate();
  const [runs, setRuns] = useState<Run[]>([]);
  const [runsLoaded, setRunsLoaded] = useState(false);
  const [run, setRun] = useState<Run | undefined>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const runIsTerminal = Boolean(
    runId &&
    run?.id === runId &&
    ["COMPLETED", "FAILED", "STOPPED"].includes(run.status),
  );
  const legacy = useResource<LegacyPayload>(
    `/models/${encodeURIComponent(modelId ?? "_none")}/runs`,
    modelId ?? "none",
  );
  useEffect(() => {
    if (modelId) return;
    let alive = true;
    const load = async () => {
      try {
        const result = await get<RunsPayload>("/runs");
        if (alive) {
          setRuns(result.runs);
          setRunsLoaded(true);
          setError("");
        }
      } catch (cause) {
        if (alive) {
          setRunsLoaded(true);
          setError((cause as Error).message);
        }
      }
    };
    void load();
    const timer = window.setInterval(load, 2500);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [modelId, refresh]);
  useEffect(() => {
    if (!runId || modelId || runIsTerminal) return;
    let alive = true;
    const load = async () => {
      try {
        const result = await get<RunPayload>(
          `/runs/${encodeURIComponent(runId)}`,
        );
        if (alive) {
          setRun(result);
          setError("");
        }
      } catch (cause) {
        if (alive) setError((cause as Error).message);
      }
    };
    void load();
    const timer = window.setInterval(load, 1800);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [runId, modelId, refresh, runIsTerminal]);
  const currentRun = run?.id === runId ? run : undefined;
  const rows = currentRun?.metrics ?? EMPTY_METRICS;
  const metricNames = useMemo(
    () => [
      ...new Set(
        rows.flatMap((row) =>
          Object.keys(row).filter(
            (key) =>
              ![
                "timestamp",
                "step",
                "epoch",
                "total_steps",
                "total_epochs",
                "progress",
                "epoch_position", "epoch_step", "epoch_steps", "validation_step", "validation_steps", "validation_progress", "indexed_records",
                "elapsed_seconds",
                "estimated_remaining_seconds",
              ].includes(key) && typeof row[key] === "number",
          ),
        ),
      ),
    ],
    [rows],
  );
  async function control(action: "pause" | "resume" | "stop") {
    if (!currentRun) return;
    setBusy(true);
    try {
      await post(`/runs/${currentRun.id}/${action}`, {});
      setRefresh((value) => value + 1);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function commitCheckpoint(checkpoint: Checkpoint) {
    if (!currentRun) return;
    setBusy(true);
    setError("");
    try {
      await post(
        `/runs/${currentRun.id}/checkpoints/${encodeURIComponent(checkpoint.id)}/commit`,
        {},
      );
      setRefresh((value) => value + 1);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function removeRun(target: Run) {
    if (
      !window.confirm(
        `Delete training run “${target.name}” and its stored run data? Committed checkpoints will be preserved.`,
      )
    )
      return;
    try {
      await remove(`/runs/${target.id}`);
      setRuns((current) => current.filter((item) => item.id !== target.id));
      if (runId === target.id) navigate("/runs");
      setRun(undefined);
      setRefresh((value) => value + 1);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }
  async function removeAllRuns() {
    if (
      !window.confirm(
        `Delete all ${runs.length} training runs and their uncommitted data? Committed checkpoints will remain usable.`,
      )
    )
      return;
    setBusy(true);
    setError("");
    try {
      await remove("/runs");
      setRuns([]);
      setRun(undefined);
      setRunsLoaded(true);
      setRefresh((value) => value + 1);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (modelId)
    return (
      <Page
        eyebrow="RESULTS EXPLORER / 04"
        title={`${modelId} results`}
        description="Existing checkpoint metadata associated with this model."
      >
        <Panel
          title="Checkpoint runs"
          meta={`${legacy.data?.runs.length ?? 0} AVAILABLE`}
        >
          <State
            loading={legacy.loading}
            error={legacy.error}
            empty={
              !legacy.data?.runs.length
                ? "No run metadata has been identified for this model."
                : undefined
            }
          />
          {legacy.data?.runs.map((item) => (
            <div className="run-row" key={item.id}>
              <span className="run-dot" />
              <div className="grow">
                <strong>{item.name}</strong>
                <small>
                  {item.model} · {item.checkpoint}
                </small>
              </div>
              <span className="badge">{item.status.toUpperCase()}</span>
            </div>
          ))}
        </Panel>
      </Page>
    );

  return (
    <Page
      eyebrow="RUN MONITOR / 04"
      title={runId ? (run?.name ?? "Training run") : "Training runs"}
      description="Live run state, persisted metrics, and resumable checkpoints."
      action={
        runId && currentRun ? (
          <button
            className="button danger"
            onClick={() => void removeRun(currentRun)}
          >
            Delete run
          </button>
        ) : !runId && runs.length ? (
          <button
            className="button danger"
            disabled={busy}
            onClick={() => void removeAllRuns()}
          >
            Delete All Training Runs
          </button>
        ) : undefined
      }
    >
      {runId ? (
        <>
          <Panel title="Run overview" meta={currentRun?.status ?? "LOADING"}>
            <State
              loading={!currentRun && !error}
              error={error || currentRun?.error}
            />
            {currentRun && (
              <>
                <div className="run-overview">
                  <div>
                    <span>RUN ID</span>
                    <strong className="mono">{currentRun.id}</strong>
                  </div>
                  <div>
                    <span>TRAINER</span>
                    <strong>{currentRun.trainer}</strong>
                  </div>
                  <div>
                    <span>ELAPSED</span>
                    <strong>{elapsed(currentRun)}</strong>
                  </div>
                  <div>
                    <span>{currentRun.current_metrics?.epoch_steps != null ? "EPOCH PROGRESS" : "PROGRESS"}</span>
                    <strong>
                      {Number(currentRun.progress ?? 0).toFixed(1)}%
                    </strong>
                  </div>
                </div>
                <div className="device-banner">
                  <span>DEVICE</span>
                  <strong>
                    {currentRun.device?.type === "cpu"
                      ? "CPU"
                      : currentRun.device
                        ? currentRun.device.index == null
                          ? "GPU"
                          : `GPU ${currentRun.device.index}`
                        : ["COMPLETED", "FAILED", "STOPPED"].includes(
                              currentRun.status,
                            )
                          ? "Unavailable"
                          : "Resolving device…"}
                  </strong>
                  {currentRun.device?.name && (
                    <small>{currentRun.device.name}</small>
                  )}
                  {currentRun.device?.fallback && (
                    <small>
                      Fallback: {currentRun.device.fallback.requested} →{" "}
                      {currentRun.device.fallback.selected}
                    </small>
                  )}
                </div>
                {currentRun.current_metrics?.total_epochs != null && (
                  <p className="mono">Epoch {String(currentRun.current_metrics.epoch ?? 0)} / {String(currentRun.current_metrics.total_epochs)} · {String(currentRun.current_metrics.phase ?? "training").replaceAll("_", " ")}</p>
                )}
                {currentRun.current_metrics?.epoch_steps != null && <p className="muted">Training · {String(currentRun.current_metrics.epoch_step ?? 0)} / {String(currentRun.current_metrics.epoch_steps)} batches</p>}
                <div className="progress-track" role="progressbar" aria-label="Training progress" aria-valuenow={Number(currentRun.progress ?? 0)} aria-valuemin={0} aria-valuemax={100}>
                  <span
                    style={{
                      width: `${Math.max(0, Math.min(100, Number(currentRun.progress ?? 0)))}%`,
                    }}
                  />
                </div>
                {currentRun.current_metrics?.validation_progress != null && <>
                  <p className="muted">Validation · {Number(currentRun.current_metrics.validation_progress).toFixed(1)}% · {String(currentRun.current_metrics.validation_step ?? 0)} / {String(currentRun.current_metrics.validation_steps ?? 0)} batches</p>
                  <div className="progress-track" role="progressbar" aria-label="Validation progress" aria-valuenow={Number(currentRun.current_metrics.validation_progress)} aria-valuemin={0} aria-valuemax={100}>
                    <span style={{ width: `${Math.max(0, Math.min(100, Number(currentRun.current_metrics.validation_progress)))}%`, background: "#e99167" }} />
                  </div>
                </>}
                <div className="live-values">
                  {Object.entries(currentRun.current_metrics ?? {})
                    .filter(
                      ([key, value]) =>
                        typeof value === "number" &&
                        !["progress", "total_steps", "total_epochs"].includes(
                          key,
                        ),
                    )
                    .map(([key, value]) => (
                      <div key={key}>
                        <span>{key.replaceAll("_", " ")}</span>
                        <strong>
                          {typeof value === "number"
                            ? value.toPrecision(5)
                            : String(value)}
                        </strong>
                      </div>
                    ))}
                </div>
                <div className="control-row">
                  {currentRun.status === "RUNNING" && (
                    <button
                      className="button"
                      disabled={busy}
                      onClick={() => control("pause")}
                    >
                      Ⅱ Pause
                    </button>
                  )}
                  {currentRun.status === "PAUSED" && (
                    <button
                      className="button primary"
                      disabled={busy}
                      onClick={() => control("resume")}
                    >
                      ▶ Resume
                    </button>
                  )}
                  {["RUNNING", "PAUSED"].includes(currentRun.status) && (
                    <button
                      className="button danger"
                      disabled={busy}
                      onClick={() => control("stop")}
                    >
                      ■ Stop gracefully
                    </button>
                  )}
                  <span className="muted mono">
                    CREATED{" "}
                    {new Date(currentRun.created_at * 1000).toLocaleString()}
                  </span>
                </div>
              </>
            )}
          </Panel>
          {currentRun && <RunTerminal key={currentRun.id} runId={currentRun.id} />}
          <Panel title="Metrics" meta={`${rows.length} OBSERVATIONS`}>
            {metricNames.length ? (
              <div className="chart-grid">
                {metricNames.map((name) => (
                  <MetricChart key={name} name={name} rows={rows} />
                ))}
              </div>
            ) : (
              <State empty="Metrics will appear as the trainer reports progress." />
            )}
          </Panel>
          <Panel
            title="Checkpoints"
            meta={`${currentRun?.checkpoints?.length ?? 0} SAVED`}
          >
            {currentRun?.checkpoints?.length ? (
              currentRun.checkpoints.map((cp) => (
                <div className="checkpoint-row-link" key={cp.id}>
                  <Link to={`/runs/${currentRun.id}/checkpoints/${cp.id}`}>
                    <span className="mono">{cp.id}</span>
                    <span>
                      Step {cp.step ?? "—"} · Epoch {cp.epoch ?? "—"}
                    </span>
                    <span>
                      Loss{" "}
                      {typeof cp.loss === "number"
                        ? cp.loss.toPrecision(4)
                        : "—"}
                    </span>
                    <span>
                      {cp.created_at
                        ? new Date(cp.created_at * 1000).toLocaleString()
                        : "—"}
                    </span>
                    <b>↗</b>
                  </Link>
                  {cp.committed ? (
                    <span className="badge">COMMITTED</span>
                  ) : (
                    <button
                      className="button small"
                      disabled={busy}
                      onClick={() => void commitCheckpoint(cp)}
                    >
                      Commit
                    </button>
                  )}
                </div>
              ))
            ) : (
              <State empty="No checkpoints saved yet." />
            )}
          </Panel>
          {currentRun?.performance && (
            <Panel
              title="Effective performance settings"
              meta="RECORDED FOR THIS RUN"
            >
              <pre className="metadata">
                {JSON.stringify(currentRun.performance, null, 2)}
              </pre>
            </Panel>
          )}
          <Panel title="Configuration" meta="RUN METADATA">
            <pre className="metadata">
              {JSON.stringify(currentRun?.configuration ?? {}, null, 2)}
            </pre>
          </Panel>
        </>
      ) : (
        <Panel title="Persisted runs" meta={`${runs.length} TOTAL`}>
          <State
            loading={!runsLoaded && !error}
            error={error}
            empty={
              !runs.length && runsLoaded && !error
                ? "No training runs yet. Configure a run to get started."
                : undefined
            }
          />
          {runs.map((item) => (
            <div className="run-row run-row-managed" key={item.id}>
              <Link to={`/runs/${item.id}`}>
                <span className="run-dot" />
                <div className="grow">
                  <strong>{item.name}</strong>
                  <small>
                    {item.trainer} · {item.id.slice(0, 12)}
                  </small>
                </div>
                <span className="badge">{item.status}</span>
                <span>{Number(item.progress ?? 0).toFixed(0)}%</span>
                <span>↗</span>
              </Link>
              <button
                className="button danger small"
                onClick={() => void removeRun(item)}
              >
                Delete
              </button>
            </div>
          ))}
        </Panel>
      )}
    </Page>
  );
}
