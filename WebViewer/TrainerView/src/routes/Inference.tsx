import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { apiUrl, get, post, remove, useResource } from "./api";
import { Page, Panel, State } from "./shared";
import "./Inference.css";
import { SequenceAttribution, type Attribution } from "./SequenceAttribution";

export type InputField = {
  key: string;
  label: string;
  description?: string;
  type: string;
  default?: unknown;
  required?: boolean;
  options?: (string | number)[];
  minimum?: number;
  maximum?: number;
  step?: number;
  accept?: string;
};
export type ModelInfo = {
  id: string;
  name: string;
  description: string;
  category: string;
  inputs: {
    key: string;
    label: string;
    type: string;
    shape?: string;
    description?: string;
  }[];
  output: { type: string; label: string; shape?: string };
  inference_parameters: InputField[];
  checkpoint_required: boolean;
};
type Models = { models: ModelInfo[] };
type Checkpoint = {
  id: string;
  run_id: string;
  run_name: string;
  committed_id?: string;
  weight_id?: string;
  epoch?: number;
  step?: number;
  loss?: number;
  created_at?: number;
};
type Checkpoints = { checkpoints: Checkpoint[] };
type InferenceRun = {
  id: string;
  name: string;
  model: string;
  status: string;
  created_at: number;
  started_at?: number;
  finished_at?: number;
  progress: {
    step: number;
    total_steps?: number;
    percent: number;
    elapsed_seconds: number;
  };
  intermediate_available: boolean;
  output_type: string;
  checkpoint_reference_status?: string;
  configuration?: {
    model?: string;
    checkpoint?: Record<string, string>;
    inputs?: Record<string, unknown>;
  };
  outputs?: Output[];
  output?: Output;
  error?: string;
};
export type Output = {
  id?: string;
  type: string;
  url?: string;
  file?: string;
  value?: unknown;
  created_at?: number;
  configuration?: InferenceRun["configuration"];
};

export function ConfigInput({
  field,
  value,
  onChange,
}: {
  field: InputField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (field.type === "boolean")
    return (
      <input
        type="checkbox"
        checked={Boolean(value)}
        onChange={(event) => onChange(event.target.checked)}
      />
    );
  if (field.type === "selection")
    return (
      <select
        value={String(value ?? "")}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">Select {field.label.toLowerCase()}</option>
        {(field.options ?? []).map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    );
  if (field.type === "file")
    return (
      <input
        type="file"
        accept={field.accept}
        onChange={(event: ChangeEvent<HTMLInputElement>) => {
          const file = event.target.files?.[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = () => onChange(reader.result);
            reader.readAsDataURL(file);
          }
        }}
      />
    );
  if (field.type === "text")
    return (
      <textarea
        rows={7}
        value={String(value ?? "")}
        onChange={(event) => onChange(event.target.value)}
        placeholder={">accession PE=1\nMKT..."}
      />
    );
  const type =
    field.type === "integer" || field.type === "float" ? "number" : "text";
  return (
    <input
      type={type}
      value={String(value ?? "")}
      min={field.minimum}
      max={field.maximum}
      step={field.step ?? (field.type === "float" ? "any" : undefined)}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function OutputRenderer({ output }: { output: Output }) {
  if (output.type === "classification") {
    const result = output.value as {
      classes?: { label: string; probability: number }[];
      attribution?: Attribution;
      memory_fallback?: string;
      note?: string;
    };
    const classes = result?.classes ?? [];
    const real = Math.max(0, Math.min(1, classes[0]?.probability ?? 0));
    return (
      <div className="classification-result">
        <svg
          viewBox="0 0 200 200"
          width="240"
          role="img"
          aria-label={`Real ${(real * 100).toFixed(1)} percent; Fake ${((1 - real) * 100).toFixed(1)} percent`}
        >
          <circle cx="100" cy="100" r="80" fill="none" stroke="#e99167" strokeWidth="18" />
          <circle
            cx="100" cy="100" r="80" fill="none" stroke="#4cbfa5"
            strokeWidth="18" pathLength="100"
            strokeDasharray={`${real * 100} 100`}
            transform="rotate(-90 100 100)"
          />
          <text x="100" y="96" textAnchor="middle" fill="currentColor" fontSize="26">
            {(Math.max(real, 1 - real) * 100).toFixed(1)}%
          </text>
          <text x="100" y="120" textAnchor="middle" fill="currentColor" fontSize="15">
            {real >= 0.5 ? "Real" : "Fake"}
          </text>
        </svg>
        <div>
          <span className="eyebrow">CLASS PROBABILITIES</span>
          {classes.map((item, index) => (
            <p className="classification-class" key={item.label}>
              <span className={`classification-dot ${index === 0 ? "real" : "fake"}`} />
              <strong>{item.label}</strong>
              <span>{(item.probability * 100).toFixed(2)}%</span>
            </p>
          ))}
          <p className="muted">{result?.note}</p>
          {result?.memory_fallback === "cuda_out_of_memory" && <p className="muted">Processed on CPU after GPU memory ran out.</p>}
        </div>
        {result?.attribution && <SequenceAttribution data={result.attribution} />}
      </div>
    );
  }
  if (output.type === "image" && output.url)
    return (
      <img
        className="inference-image"
        src={apiUrl(output.url)}
        alt="Model inference output"
      />
    );
  if (output.type === "text")
    return <pre className="inference-text">{String(output.value ?? "")}</pre>;
  if (output.type === "file" && output.url)
    return (
      <a className="button" href={apiUrl(output.url)} download>
        Download output
      </a>
    );
  if (output.type === "file" && typeof output.value === "string")
    return (
      <a className="button" href={output.value} download>
        Download output
      </a>
    );
  if (
    output.type === "numerical" ||
    output.type === "tensor" ||
    output.type === "json"
  )
    return (
      <pre className="metadata">{JSON.stringify(output.value, null, 2)}</pre>
    );
  return (
    <pre className="metadata">
      {JSON.stringify(output.value ?? output, null, 2)}
    </pre>
  );
}

function elapsed(run: InferenceRun) {
  const seconds = Math.max(
    0,
    (run.finished_at ?? Date.now() / 1000) - (run.started_at ?? run.created_at),
  );
  return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
}
function checkpointLabel(
  reference?: Record<string, string>,
  referenceStatus?: string,
) {
  if (!reference) return "Not required";
  if (reference.committed_id)
    return `Committed checkpoint · ${reference.committed_id}${referenceStatus === "deleted" ? " (source checkpoint deleted)" : ""}`;
  return reference.weight_id
    ? `Model weights · ${reference.weight_id}`
    : `${reference.run_id ?? "Training run"} · ${reference.checkpoint ?? "Checkpoint"}`;
}
function safeInputs(inputs?: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(inputs ?? {}).map(([key, value]) => [
      key,
      typeof value === "string" && value.startsWith("data:")
        ? "[uploaded file]"
        : value,
    ]),
  );
}
function inputSummary(inputs?: Record<string, unknown>) {
  return JSON.stringify(safeInputs(inputs));
}
function visibleConfiguration(run?: InferenceRun) {
  return {
    model: run?.configuration?.model ?? run?.model,
    checkpoint: run?.configuration?.checkpoint ?? null,
    checkpoint_status: run?.checkpoint_reference_status,
    inputs: safeInputs(run?.configuration?.inputs),
  };
}

function checkpointReferenceValue(reference?: Record<string, string>) {
  if (reference?.weight_id) return `weight::${reference.weight_id}`;
  if (reference?.committed_id) return `committed::${reference.committed_id}`;
  if (reference?.run_id && reference?.checkpoint)
    return `${reference.run_id}::${reference.checkpoint}`;
  return "";
}

function parseCheckpointReference(value: string) {
  if (!value) return undefined;
  if (value.startsWith("weight::"))
    return { weight_id: value.slice("weight::".length) };
  if (value.startsWith("committed::"))
    return { committed_id: value.slice("committed::".length) };
  const [run_id, ...parts] = value.split("::");
  return { run_id, checkpoint: parts.join("::") };
}

function randomSeed(field: InputField) {
  const minimum = field.minimum ?? 0;
  const maximum = field.maximum ?? 2147483647;
  const word = window.crypto?.getRandomValues
    ? window.crypto.getRandomValues(new Uint32Array(1))[0] ?? 0
    : Math.floor(Math.random() * (maximum - minimum + 1));
  return minimum + (word % (maximum - minimum + 1));
}

function InferenceRepeat({
  run,
  onRunUpdated,
}: {
  run: InferenceRun;
  onRunUpdated: (run: InferenceRun) => void;
}) {
  const models = useResource<Models>("/inference/models", "inference-models");
  const [modelId, setModelId] = useState(run.configuration?.model ?? run.model);
  const [checkpointRef, setCheckpointRef] = useState(
    checkpointReferenceValue(run.configuration?.checkpoint),
  );
  const [inputs, setInputs] = useState<Record<string, unknown>>(
    run.configuration?.inputs ?? {},
  );
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const selected = models.data?.models.find((item) => item.id === modelId);
  const checkpoints = useResource<Checkpoints>(
    modelId
      ? `/inference/models/${encodeURIComponent(modelId)}/checkpoints`
      : "/inference/models/_none/checkpoints",
    modelId || "none",
  );

  function changeModel(id: string) {
    setModelId(id);
    setCheckpointRef("");
    const next = models.data?.models.find((item) => item.id === id);
    setInputs(
      Object.fromEntries(
        (next?.inference_parameters ?? []).map((field) => [
          field.key,
          field.default ?? "",
        ]),
      ),
    );
  }

  function parsedInputs() {
    return Object.fromEntries(
      Object.entries(inputs).map(([key, value]) => {
        const kind = selected?.inference_parameters.find((field) => field.key === key)?.type;
        return [
          key,
          typeof value === "string" && (kind === "integer" || kind === "float") && value.trim()
            ? Number(value)
            : value,
        ];
      }),
    );
  }

  async function rerun() {
    if (!selected) return;
    setBusy(true);
    setMessage("Validating parameters…");
    try {
      const config = {
        model: modelId,
        checkpoint: parseCheckpointReference(checkpointRef),
        inputs: parsedInputs(),
      };
      await post("/inference/runs/validate", config);
      const updated = await post<InferenceRun>(
        `/inference/runs/${encodeURIComponent(run.id)}/rerun`,
        config,
      );
      onRunUpdated(updated);
      setMessage("Inference started. Previous outputs are still saved below.");
    } catch (cause) {
      setMessage((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel title="Run again" meta="EDIT PARAMETERS">
      <div className="inference-repeat-form">
        <label className="field">
          Model
          <select value={modelId} onChange={(event) => changeModel(event.target.value)}>
            <option value="">Select an inference-capable model</option>
            {models.data?.models.map((item) => (
              <option key={item.id} value={item.id}>{item.name}</option>
            ))}
          </select>
        </label>
        {selected && (
          <label className="field">
            Checkpoint or model weights{selected.checkpoint_required && <em> REQUIRED</em>}
            <select value={checkpointRef} onChange={(event) => setCheckpointRef(event.target.value)}>
              <option value="">{selected.checkpoint_required ? "Select a checkpoint or saved weights" : "Use model defaults (no checkpoint)"}</option>
              {checkpoints.data?.checkpoints.map((item) => {
                const value = item.weight_id
                  ? `weight::${item.weight_id}`
                  : item.committed_id
                    ? `committed::${item.committed_id}`
                    : `${item.run_id}::${item.id}`;
                return (
                  <option key={value} value={value}>
                    {item.run_name} · {item.id}
                    {item.weight_id ? " · MODEL WEIGHTS" : item.committed_id ? " · COMMITTED" : ` · epoch ${item.epoch ?? "—"}`}
                  </option>
                );
              })}
            </select>
            {checkpoints.error && <small>{checkpoints.error}</small>}
          </label>
        )}
        {selected?.inference_parameters.map((field) => (
          <label className="field" key={field.key}>
            <span>{field.label}{field.required && <em> REQUIRED</em>}</span>
            {field.description && <small>{field.description}</small>}
            {field.type === "file" && typeof inputs[field.key] === "string" && String(inputs[field.key]).startsWith("data:") && (
              <small>Existing uploaded input will be reused unless replaced.</small>
            )}
            {field.key === "seed" && (field.type === "integer" || field.type === "float") ? (
              <div className="parameter-with-action">
                <ConfigInput
                  field={field}
                  value={inputs[field.key] ?? field.default ?? ""}
                  onChange={(value) => setInputs((current) => ({ ...current, [field.key]: value }))}
                />
                <button
                  className="button small"
                  type="button"
                  title="Generate a random seed"
                  onClick={() => {
                    const seed = randomSeed(field);
                    setInputs((current) => ({ ...current, [field.key]: seed }));
                  }}
                >
                  ⤨ Random
                </button>
              </div>
            ) : (
              <ConfigInput
                field={field}
                value={inputs[field.key] ?? field.default ?? ""}
                onChange={(value) => setInputs((current) => ({ ...current, [field.key]: value }))}
              />
            )}
          </label>
        ))}
        <div className="inference-repeat-actions">
          <button
            className="button primary"
            disabled={busy || !selected || Boolean(selected.checkpoint_required && !checkpointRef)}
            onClick={() => void rerun()}
          >
            {busy ? "Starting…" : "▶ Run inference again"}
          </button>
          {message && <span className="form-message">{message}</span>}
        </div>
      </div>
    </Panel>
  );
}

function InferenceHistory() {
  const [runs, setRuns] = useState<InferenceRun[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const result = await get<{ runs: InferenceRun[] }>("/inference/runs");
        if (active) {
          setRuns(result.runs);
          setLoaded(true);
          setError("");
        }
      } catch (cause) {
        if (active) {
          setLoaded(true);
          setError((cause as Error).message);
        }
      }
    };
    void load();
    const timer = window.setInterval(load, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [refresh]);
  async function deleteRun(run: InferenceRun) {
    if (
      !window.confirm(
        `Delete inference “${run.name}” and its stored inputs, outputs, and progress? Referenced model weights and checkpoints will remain.`,
      )
    )
      return;
    try {
      await remove(`/inference/runs/${encodeURIComponent(run.id)}`);
      setRuns((current) => current.filter((item) => item.id !== run.id));
      setError("");
      setRefresh((value) => value + 1);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }
  return (
    <Page
      eyebrow="INFERENCE / 09"
      title="Inference"
      description="Review saved inference runs or start a new model inference."
      action={
        <Link className="button primary" to="/inference/new">
          ＋ New inference
        </Link>
      }
    >
      <Panel title="Inference history" meta={`${runs.length} SAVED RUNS`}>
        <State
          loading={!loaded && !error}
          error={error}
          empty={
            loaded && !error && !runs.length
              ? "No inference runs yet. Start a new inference to see its results here."
              : undefined
          }
        />
        {runs.map((run) => (
          <div className="inference-history-row" key={run.id}>
            <div className="run-dot" />
            <div className="grow">
              <strong>{run.name}</strong>
              <small className="mono">
                {run.id} · {run.model}
              </small>
              <small>
                Checkpoint / weights:{" "}
                {checkpointLabel(
                  run.configuration?.checkpoint,
                  run.checkpoint_reference_status,
                )}
              </small>
              <small>Input: {inputSummary(run.configuration?.inputs)}</small>
            </div>
            <span className="inference-created">
              {new Date(run.created_at * 1000).toLocaleString()}
            </span>
            <span className={`badge status-${run.status.toLowerCase()}`}>
              {run.status}
            </span>
            <Link
              className="button small"
              to={`/inference/runs/${encodeURIComponent(run.id)}`}
            >
              View
            </Link>
            <button
              className="button danger small"
              onClick={() => void deleteRun(run)}
            >
              Delete
            </button>
          </div>
        ))}
      </Panel>
    </Page>
  );
}

function InferenceDetail({ runId }: { runId: string }) {
  const navigate = useNavigate();
  const [run, setRun] = useState<InferenceRun>();
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const value = await get<InferenceRun>(
          `/inference/runs/${encodeURIComponent(runId)}`,
        );
        if (active) {
          setRun(value);
          setError("");
        }
      } catch (cause) {
        if (active) {
          setRun(undefined);
          setError((cause as Error).message);
        }
      }
    };
    void poll();
    const timer = window.setInterval(poll, 900);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [runId]);
  async function deleteRun() {
    if (
      !run ||
      !window.confirm(
        `Delete inference “${run.name}” and its stored result data? Referenced model weights and checkpoints will remain.`,
      )
    )
      return;
    try {
      await remove(`/inference/runs/${encodeURIComponent(run.id)}`);
      navigate("/inference");
    } catch (cause) {
      setError((cause as Error).message);
    }
  }
  async function stopRun() {
    try {
      const updated = await post<InferenceRun>(
        `/inference/runs/${encodeURIComponent(runId)}/stop`,
        {},
      );
      setRun(updated);
      setError("");
    } catch (cause) {
      setError((cause as Error).message);
    }
  }
  async function deleteOutput(output: Output) {
    if (!run || !output.id) return;
    if (!window.confirm("Delete this saved inference output?")) return;
    try {
      await remove(
        `/inference/runs/${encodeURIComponent(run.id)}/outputs/${encodeURIComponent(output.id)}`,
      );
      const updated = await get<InferenceRun>(`/inference/runs/${encodeURIComponent(run.id)}`);
      setRun(updated);
      setError("");
    } catch (cause) {
      setError((cause as Error).message);
    }
  }
  const outputRun = run;
  const savedOutputs = outputRun?.outputs ?? [];
  return (
    <Page
      eyebrow="INFERENCE RESULT / 09"
      title={run?.name ?? "Inference result"}
      description="Persisted run metadata, execution state, and model output."
      action={
        <button
          className="button danger"
          disabled={
            !run || ["QUEUED", "RUNNING", "STOPPING"].includes(run.status)
          }
          onClick={() => void deleteRun()}
        >
          Delete inference
        </button>
      }
    >
      <Panel title="Inference information" meta={run?.status ?? "LOADING"}>
        <State loading={!run && !error} error={error || run?.error} />
        {run && (
          <>
            <div className="run-overview">
              <div>
                <span>INFERENCE ID</span>
                <strong className="mono">{run.id}</strong>
              </div>
              <div>
                <span>MODEL</span>
                <strong>{run.model}</strong>
              </div>
              <div>
                <span>STATUS</span>
                <strong>{run.status}</strong>
              </div>
              <div>
                <span>CREATED</span>
                <strong>
                  {new Date(run.created_at * 1000).toLocaleString()}
                </strong>
              </div>
              <div>
                <span>ELAPSED</span>
                <strong>{elapsed(run)}</strong>
              </div>
              <div>
                <span>PROGRESS</span>
                <strong>
                  {run.progress.total_steps
                    ? `${run.progress.step} / ${run.progress.total_steps}`
                    : `${(run.progress.percent ?? 0).toFixed(1)}%`}
                </strong>
              </div>
            </div>
            <div className="progress-track">
              <span
                style={{
                  width: `${Math.max(0, Math.min(100, run.progress.percent ?? 0))}%`,
                }}
              />
            </div>
            {run.intermediate_available && (
              <div className="intermediate-view">
                <div className="eyebrow">
                  LIVE INTERMEDIATE OUTPUT · STEP {run.progress.step}
                </div>
                <img
                  className="inference-image"
                  src={`${apiUrl(`/inference/runs/${encodeURIComponent(run.id)}/intermediate`)}?step=${run.progress.step}`}
                  alt="Latest intermediate model output"
                />
              </div>
            )}
            {run.status === "RUNNING" && (
              <div className="control-row">
                <button
                  className="button danger"
                  onClick={() => void stopRun()}
                >
                  ■ Stop inference
                </button>
                <span className="muted mono">
                  {run.progress.elapsed_seconds.toFixed(1)}S ELAPSED
                </span>
              </div>
            )}
          </>
        )}
      </Panel>
      {run && <InferenceRepeat run={run} onRunUpdated={setRun} />}
      <Panel title="Request details" meta="SAVED INPUTS & PARAMETERS">
        <pre className="metadata">
          {JSON.stringify(visibleConfiguration(run), null, 2)}
        </pre>
      </Panel>
      <Panel
        title="Saved outputs"
        meta={`${run?.outputs?.length ?? 0} OUTPUTS`}
      >
        <State
          loading={!run && !error}
          error={error || undefined}
          empty={
            run?.outputs?.length
              ? undefined
              : ["QUEUED", "RUNNING", "STOPPING"].includes(run?.status ?? "")
                ? "The new output will be added here when this inference completes."
                : "No output has been saved for this inference yet."
          }
        />
        {outputRun && [...savedOutputs].reverse().map((output, index) => (
          <article className="inference-output-item" key={output.id ?? index}>
            <header>
              <div>
                <strong>Output {savedOutputs.length - index}</strong>
                <span className="badge">{output.type.toUpperCase()}</span>
                <small>{output.created_at ? new Date(output.created_at * 1000).toLocaleString() : "Saved output"}</small>
              </div>
              <button className="button danger small" onClick={() => void deleteOutput(output)}>
                Delete output
              </button>
            </header>
            <OutputRenderer
              output={{
                ...output,
                url: output.file && output.id
                  ? `/api/inference/runs/${encodeURIComponent(outputRun.id)}/output/file?output_id=${encodeURIComponent(output.id)}`
                  : output.url,
              }}
            />
            {output.configuration && (
              <details className="inference-output-configuration">
                <summary>Parameters used for this output</summary>
                <pre className="metadata">{JSON.stringify(visibleConfiguration({
                  id: outputRun.id,
                  name: outputRun.name,
                  model: String(output.configuration.model ?? outputRun.model),
                  status: outputRun.status,
                  created_at: outputRun.created_at,
                  progress: outputRun.progress,
                  intermediate_available: false,
                  output_type: output.type,
                  configuration: output.configuration,
                }), null, 2)}</pre>
              </details>
            )}
          </article>
        ))}
      </Panel>
      <Link className="text-link back-link" to="/inference">
        ← Inference history
      </Link>
    </Page>
  );
}

function InferenceSetup() {
  const models = useResource<Models>("/inference/models", "inference-models");
  const navigate = useNavigate();
  const [modelId, setModelId] = useState("");
  const [checkpointRef, setCheckpointRef] = useState("");
  const [inputs, setInputs] = useState<Record<string, unknown>>({});
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const selected = models.data?.models.find((item) => item.id === modelId);
  const checkpoints = useResource<Checkpoints>(
    modelId
      ? `/inference/models/${encodeURIComponent(modelId)}/checkpoints`
      : "/inference/models/_none/checkpoints",
    modelId || "none",
  );
  function selectModel(id: string) {
    setModelId(id);
    setCheckpointRef("");
    const item = models.data?.models.find((candidate) => candidate.id === id);
    setInputs(
      Object.fromEntries(
        (item?.inference_parameters ?? []).map((field) => [
          field.key,
          field.default ?? "",
        ]),
      ),
    );
    setMessage("");
  }
  function parsedInputs() {
    return Object.fromEntries(
      Object.entries(inputs).map(([key, value]) => [
        key,
        typeof value === "string" &&
        (selected?.inference_parameters.find((field) => field.key === key)
          ?.type === "integer" ||
          selected?.inference_parameters.find((field) => field.key === key)
            ?.type === "float") &&
        value.trim() !== ""
          ? Number(value)
          : value,
      ]),
    );
  }
  async function start() {
    if (!selected) return;
    setBusy(true);
    setMessage("Validating inference configuration…");
    try {
      const checkpoint = checkpointRef
        ? checkpointRef.startsWith("weight::")
          ? { weight_id: checkpointRef.slice("weight::".length) }
          : checkpointRef.startsWith("committed::")
            ? { committed_id: checkpointRef.slice("committed::".length) }
            : (() => {
                const [run_id, ...rest] = checkpointRef.split("::");
                return { run_id, checkpoint: rest.join("::") };
              })()
        : undefined;
      const config = { model: modelId, checkpoint, inputs: parsedInputs() };
      await post("/inference/runs/validate", config);
      const created = await post<InferenceRun>("/inference/runs", config);
      navigate(`/inference/runs/${created.id}`);
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page
      eyebrow="MODEL INFERENCE / 09"
      title="New inference"
      description="Choose a model, provide metadata-driven inputs, and monitor the persisted output."
      action={
        <Link className="button" to="/inference">
          Inference history
        </Link>
      }
    >
      <div className="two-col">
        <div>
          <Panel title="Inference setup" meta="MODEL-DRIVEN">
            <State loading={models.loading} error={models.error} />
            <label className="field">
              Model
              <select
                value={modelId}
                onChange={(event) => selectModel(event.target.value)}
              >
                <option value="">Select an inference-capable model</option>
                {models.data?.models.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            {selected && (
              <>
                <div className="model-description">
                  <span className="badge">{selected.category}</span>
                  <p>{selected.description}</p>
                  <strong>
                    Model output · {selected.output.label} (
                    {selected.output.type})
                  </strong>
                  {selected.inputs.map((input) => (
                    <small key={input.key}>
                      {input.label}
                      {input.shape ? ` · ${input.shape}` : ""}
                      {input.description ? ` — ${input.description}` : ""}
                    </small>
                  ))}
                </div>
                {selected.checkpoint_required && (
                  <label className="field">
                    Checkpoint or model weights
                    <select
                      value={checkpointRef}
                      onChange={(event) => setCheckpointRef(event.target.value)}
                    >
                      <option value="">
                        Select a checkpoint or saved weights
                      </option>
                      {checkpoints.data?.checkpoints.map((item) => (
                        <option
                          key={`${item.run_id}/${item.id}/${item.committed_id ?? ""}/${item.weight_id ?? ""}`}
                          value={
                            item.weight_id
                              ? `weight::${item.weight_id}`
                              : item.committed_id
                                ? `committed::${item.committed_id}`
                                : `${item.run_id}::${item.id}`
                          }
                        >
                          {item.run_name} · {item.id}
                          {item.weight_id
                            ? " · MODEL WEIGHTS"
                            : item.committed_id
                              ? " · COMMITTED"
                              : ` · epoch ${item.epoch ?? "—"}`}
                        </option>
                      ))}
                    </select>
                    {checkpoints.error && <small>{checkpoints.error}</small>}
                  </label>
                )}
                {selected.inference_parameters.map((field) => (
                  <label className="field" key={field.key}>
                    <span>
                      {field.label}
                      {field.required && <em> REQUIRED</em>}
                    </span>
                    {field.description && <small>{field.description}</small>}
                    {field.key === "seed" && (field.type === "integer" || field.type === "float") ? (
                      <div className="parameter-with-action">
                        <ConfigInput
                          field={field}
                          value={inputs[field.key] ?? field.default ?? ""}
                          onChange={(value) => setInputs((current) => ({ ...current, [field.key]: value }))}
                        />
                        <button
                          className="button small"
                          type="button"
                          title="Generate a random seed"
                          onClick={() => setInputs((current) => ({ ...current, [field.key]: randomSeed(field) }))}
                        >
                          ⤨ Random
                        </button>
                      </div>
                    ) : (
                      <ConfigInput
                        field={field}
                        value={inputs[field.key] ?? field.default ?? ""}
                        onChange={(value) => setInputs((current) => ({ ...current, [field.key]: value }))}
                      />
                    )}
                  </label>
                ))}
                <button
                  className="button primary"
                  disabled={
                    busy || (selected.checkpoint_required && !checkpointRef)
                  }
                  onClick={() => void start()}
                >
                  {busy ? "Starting…" : "Run inference"}
                </button>
                {message && <p className="form-message">{message}</p>}
              </>
            )}
          </Panel>
        </div>
        <Panel
          title="Request preview"
          meta={selected?.output.type.toUpperCase() ?? "OUTPUT"}
        >
          <pre className="metadata">
            {JSON.stringify(
              {
                model: modelId || null,
                checkpoint: checkpointRef || null,
                inputs: Object.fromEntries(
                  Object.entries(inputs).map(([key, value]) => [
                    key,
                    typeof value === "string" && value.startsWith("data:")
                      ? "[uploaded file]"
                      : value,
                  ]),
                ),
              },
              null,
              2,
            )}
          </pre>
          {selected && (
            <div className="schema-summary">
              <span className="mono">
                INPUTS ·{" "}
                {selected.inputs.map((item) => item.type).join(" + ") ||
                  "CONFIGURATION"}
              </span>
              <span className="mono">
                OUTPUT · {selected.output.type.toUpperCase()}
              </span>
            </div>
          )}
        </Panel>
      </div>
    </Page>
  );
}

export function Inference() {
  const { runId } = useParams();
  const location = useLocation();
  if (runId) return <InferenceDetail key={runId} runId={runId} />;
  return location.pathname === "/inference/new" ? (
    <InferenceSetup />
  ) : (
    <InferenceHistory />
  );
}
