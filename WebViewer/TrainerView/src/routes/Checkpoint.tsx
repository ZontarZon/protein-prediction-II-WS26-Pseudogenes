import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { post, remove, useResource } from "./api";
import { Page, Panel, State } from "./shared";
import { MetricChart } from "./MetricChart";

type MetricRow = { [key: string]: string | number | null };
type Checkpoint = {
  id: string;
  model?: string;
  models?: Record<string, string>;
  created_at?: number;
  source_created_at?: number;
  epoch?: number;
  step?: number;
  loss?: number;
  metrics: Record<string, unknown>;
  trainer?: string;
  run_id?: string;
  run_name?: string;
  checkpoint?: string;
  datasets?: Record<string, string>;
  hyperparameters?: Record<string, unknown>;
  configuration?: Record<string, unknown>;
  history: MetricRow[];
};

export function CheckpointDetail() {
  const { runId = "", checkpointId = "", committedId = "" } = useParams();
  const navigate = useNavigate();
  const isCommitted = Boolean(committedId);
  const resource = isCommitted
    ? `/checkpoints/${encodeURIComponent(committedId)}`
    : `/runs/${encodeURIComponent(runId)}/checkpoints/${encodeURIComponent(checkpointId)}`;
  const { data, error, loading } = useResource<Checkpoint>(
    resource,
    `${committedId || runId}/${checkpointId}`,
  );
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const modelId = data?.model ?? Object.values(data?.models ?? {})[0] ?? "";
  const chartNames = [
    ...new Set(
      data?.history?.flatMap((row) =>
        Object.keys(row).filter(
          (key) =>
            typeof row[key] === "number" &&
            ![
              "step",
              "epoch",
              "timestamp",
              "progress",
              "total_steps",
              "total_epochs",
            ].includes(key),
        ),
      ) ?? [],
    ),
  ];
  const startUrl = isCommitted
    ? `/new-run?model=${encodeURIComponent(modelId)}&initCommitted=${encodeURIComponent(committedId)}`
    : `/new-run?model=${encodeURIComponent(modelId)}&initRun=${encodeURIComponent(runId)}&initCheckpoint=${encodeURIComponent(checkpointId)}`;
  async function extract() {
    setBusy(true);
    setMessage("Extracting model-only weights…");
    try {
      const result = await post<{ id: string }>(
        isCommitted
          ? `/checkpoints/${encodeURIComponent(committedId)}/extract-weights`
          : `/runs/${encodeURIComponent(runId)}/checkpoints/${encodeURIComponent(checkpointId)}/extract-weights`,
        {},
      );
      navigate(`/weights?weight=${encodeURIComponent(result.id)}`);
    } catch (cause) {
      setMessage((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function deleteCommitted() {
    if (
      !data ||
      !window.confirm(
        `Delete committed checkpoint “${data.id}”? This permanently removes the saved checkpoint. Extracted model weights and inference results will remain.`,
      )
    )
      return;
    setBusy(true);
    setMessage("");
    try {
      await remove(`/checkpoints/${encodeURIComponent(committedId)}`);
      navigate("/models");
    } catch (cause) {
      setMessage((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page
      eyebrow="CHECKPOINT INSPECTOR / 08"
      title={data?.id ?? (isCommitted ? committedId : checkpointId)}
      description="Checkpoint state, performance history, and reusable model-weight actions."
      action={
        <div className="checkpoint-actions">
          <Link className="button primary" to={startUrl}>
            Start New Training Run ↗
          </Link>
          <button
            className="button"
            disabled={busy || loading || !data}
            onClick={() => void extract()}
          >
            {busy ? "Extracting…" : "Extract / Save Model Weights"}
          </button>
          {isCommitted && (
            <button
              className="button danger"
              disabled={busy || loading || !data}
              onClick={() => void deleteCommitted()}
            >
              Delete Checkpoint
            </button>
          )}
        </div>
      }
    >
      <Panel
        title="Checkpoint information"
        meta={isCommitted ? "COMMITTED CHECKPOINT" : "TRAINING RUN CHECKPOINT"}
      >
        <State loading={loading} error={error} />
        {data && (
          <>
            <div className="run-overview">
              <div>
                <span>MODEL</span>
                <strong>{modelId || "—"}</strong>
              </div>
              <div>
                <span>TRAINER</span>
                <strong>{data.trainer ?? "Unknown"}</strong>
              </div>
              <div>
                <span>CREATED</span>
                <strong>
                  {data.created_at
                    ? new Date(data.created_at * 1000).toLocaleString()
                    : "—"}
                </strong>
              </div>
              <div>
                <span>SOURCE RUN</span>
                <strong className="mono">
                  {data.run_name ?? data.run_id ?? "—"}
                </strong>
              </div>
              <div>
                <span>EPOCH</span>
                <strong>{data.epoch ?? "—"}</strong>
              </div>
              <div>
                <span>STEP</span>
                <strong>{data.step ?? "—"}</strong>
              </div>
              <div>
                <span>LOSS</span>
                <strong>
                  {typeof data.loss === "number"
                    ? data.loss.toPrecision(5)
                    : "—"}
                </strong>
              </div>
              <div>
                <span>SOURCE CHECKPOINT</span>
                <strong>{data.checkpoint ?? data.id}</strong>
              </div>
            </div>
            <div className="detail-grid">
              <div>
                <h3>Metrics</h3>
                <pre className="metadata">
                  {JSON.stringify(data.metrics ?? {}, null, 2)}
                </pre>
              </div>
              <div>
                <h3>Models & data</h3>
                <pre className="metadata">
                  {JSON.stringify(
                    { models: data.models, datasets: data.datasets },
                    null,
                    2,
                  )}
                </pre>
              </div>
              <div>
                <h3>Training configuration</h3>
                <pre className="metadata">
                  {JSON.stringify(
                    {
                      configuration: data.configuration,
                      hyperparameters: data.hyperparameters,
                    },
                    null,
                    2,
                  )}
                </pre>
              </div>
            </div>
          </>
        )}
        {message && <div className="state error">{message}</div>}
      </Panel>
      <Panel
        title="Training history"
        meta={`THROUGH STEP ${data?.step ?? "—"}`}
      >
        {data && chartNames.length ? (
          <div className="chart-grid">
            {chartNames.map((name) => (
              <MetricChart key={name} name={name} rows={data.history ?? []} />
            ))}
          </div>
        ) : (
          <State empty="No historical metrics were saved for this checkpoint." />
        )}
      </Panel>
    </Page>
  );
}
