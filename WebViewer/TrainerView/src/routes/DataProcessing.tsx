import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { apiUrl, post, useResource } from "./api";
import { Page, Panel, State } from "./shared";
import { ConfigInput, OutputRenderer } from "./Inference";
import type { ModelInfo, Output } from "./Inference";
import "./DataProcessing.css";

type Configuration = {
  model: string; dataset: string; adapter: string; class_field: string;
  checkpoint?: Record<string, string>; inputs: Record<string, unknown>;
  mapping: Record<string, string>; subsets: Record<string, number | null>; seed: number;
};
type Job = {
  id: string; status: string; configuration: Configuration; indexed: number;
  total: number; completed: number; failed: number; error?: string;
};
type Checkpoint = { id: string; run_id: string; run_name: string; committed_id?: string; weight_id?: string };
type Schema = { status: string; fields?: string[]; total?: number; groups?: { key: string; count: number }[]; error?: string };
type RecordRow = { id: number; label: string; class: string; preview: string; status: string; output?: Output; error?: string };
type RecordDetail = { record_id: number; source: string; source_index: number; data: Record<string, unknown>; inputs?: Record<string, unknown>; output?: Output; error?: string };
const active = (status: string) => ["QUEUED", "INDEXING", "RUNNING", "STOPPING"].includes(status);
const empty: Configuration = { model: "", dataset: "", adapter: "fasta", class_field: "class", inputs: {}, mapping: {}, subsets: {}, seed: 7 };
const checkpointValue = (c?: Record<string, string>) => c ? JSON.stringify(c) : "";

function DatasetFields({ config, setConfig, disabled }: { config: Configuration; setConfig: (c: Configuration) => void; disabled: boolean }) {
  const schema = useResource<Schema>(`/processing/datasets/${encodeURIComponent(config.dataset)}/schema?adapter=${config.adapter}&class_field=${encodeURIComponent(config.class_field)}`, config.dataset, 2500);
  const update = (key: string, count: number | null) => setConfig({ ...config, subsets: { ...config.subsets, [key]: count } });
  return <>
    <datalist id="processing-fields">{schema.data?.fields?.map(field => <option key={field} value={field} />)}</datalist>
    <div className="processing-subsets">
      {schema.data?.groups?.map(group => {
        const count = Object.hasOwn(config.subsets, group.key) ? config.subsets[group.key] : 0;
        return <label key={group.key}>
          <span><input type="checkbox" checked={count !== null} disabled={disabled}
            onChange={event => update(group.key, event.target.checked ? 0 : null)} /> {group.key} · {group.count.toLocaleString()} records</span>
          <input aria-label={`Subset count for ${group.key}`} type="number" min={0} step={1} disabled={disabled || count === null}
            value={count ?? 0} onChange={event => update(group.key, Math.max(0, Math.floor(Number(event.target.value))))} />
        </label>;
      })}
    </div>
    {schema.data?.status === "ready" ? <p className="muted">{schema.data.total?.toLocaleString()} records. Subset counts: 0 uses all records in an included class. Positive counts sample up to that many records. No training/validation split.</p>
      : <p className={schema.error || schema.data?.error ? "error" : "muted"}>{schema.error || schema.data?.error || "Scanning dataset fields and classes…"}</p>}
  </>;
}

function Settings({ models, job }: { models: ModelInfo[]; job?: Job }) {
  const [config, setConfig] = useState<Configuration>(job?.configuration ?? empty);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();
  const datasets = useResource<{ datasets: { id: string; name: string }[] }>("/datasets", "processing");
  const model = models.find(item => item.id === config.model);
  const checkpoints = useResource<{ checkpoints: Checkpoint[] }>(`/inference/models/${encodeURIComponent(config.model || "_none")}/checkpoints`, config.model);
  function chooseModel(id: string) {
    const selected = models.find(item => item.id === id);
    const inputs = Object.fromEntries((selected?.inference_parameters ?? []).filter(p => p.default !== undefined).map(p => [p.key, p.key === "explain" ? true : p.default]));
    const mapping = Object.fromEntries((selected?.inputs ?? []).map(p => [p.key, p.type === "file" ? "file" : p.key]));
    setConfig({ ...config, model: id, checkpoint: undefined, inputs, mapping });
  }
  async function start() {
    setBusy(true); setError("");
    try {
      const result = await post<Job>("/processing/runs", config);
      navigate(`/processing/runs/${result.id}`);
    } catch (exception) { setError((exception as Error).message); }
    finally { setBusy(false); }
  }
  return <Panel title="Processing settings">
    <fieldset disabled={Boolean(job) || busy} className="processing-settings">
      <div className="processing-config-grid">
        <label>Model<select value={config.model} onChange={event => chooseModel(event.target.value)}>
          <option value="">Select model</option>{models.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select></label>
        <label>Model weights / checkpoint<select value={checkpointValue(config.checkpoint)} onChange={event => setConfig({ ...config, checkpoint: event.target.value ? JSON.parse(event.target.value) : undefined })}>
          <option value="">{model?.checkpoint_required ? "Select checkpoint" : "No checkpoint"}</option>
          {checkpoints.data?.checkpoints.map(item => {
            const ref = item.weight_id ? { weight_id: item.weight_id } : item.committed_id ? { committed_id: item.committed_id } : { run_id: item.run_id, checkpoint: item.id };
            return <option key={JSON.stringify(ref)} value={JSON.stringify(ref)}>{item.run_name} · {item.id}</option>;
          })}
        </select></label>
        <label>Dataset<select value={config.dataset} onChange={event => setConfig({ ...config, dataset: event.target.value, subsets: {} })}>
          <option value="">Select dataset</option>{datasets.data?.datasets.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select></label>
        <label>Record format<select value={config.adapter} onChange={event => setConfig({ ...config, adapter: event.target.value, class_field: event.target.value === "fasta" ? "class" : event.target.value === "images" ? "label" : "", subsets: {}, mapping: Object.fromEntries((model?.inputs ?? []).map(p => [p.key, event.target.value === "images" && p.type === "file" ? "file" : p.key])) })}>
          <option value="fasta">Protein FASTA (plain / gzip)</option><option value="jsonl">JSON Lines</option><option value="csv">CSV / TSV</option><option value="images">Images (folder = class)</option>
        </select></label>
        <label>Class field (empty = all records)<input list="processing-fields" value={config.class_field} onChange={event => setConfig({ ...config, class_field: event.target.value, subsets: {} })} /></label>
        <label>Subset seed<input type="number" step={1} value={config.seed} onChange={event => setConfig({ ...config, seed: Number(event.target.value) })} /></label>
      </div>
      {config.dataset && <DatasetFields config={config} setConfig={setConfig} disabled={Boolean(job)} />}
      {model && <div className="processing-config-grid">
        {model.inference_parameters.map(field => <div key={field.key} className="processing-parameter">
          {model.inputs.some(input => input.key === field.key) &&
          <label>{field.label} · dataset field
            <input list="processing-fields" placeholder="Leave empty to use a shared value" value={config.mapping[field.key] ?? ""}
              onChange={event => setConfig({ ...config, mapping: { ...config.mapping, [field.key]: event.target.value } })} />
          </label>}
          {!config.mapping[field.key] && <label>{field.label} · shared value<ConfigInput field={field} value={config.inputs[field.key] ?? field.default ?? ""} onChange={value => {
            const parsed = field.type === "integer" || field.type === "float" ? (value === "" ? undefined : Number(value)) : value;
            setConfig({ ...config, inputs: { ...config.inputs, [field.key]: parsed } });
          }} /></label>}
        </div>)}
      </div>}
    </fieldset>
    <p className="muted">{config.adapter === "fasta" && "FASTA classes use PE 1–4 = real, PE 5 = fake; missing evidence labels are unknown. "}Enable residue contribution maps to save heat maps for record details.</p>
    {(error || datasets.error || (config.model && checkpoints.error)) && <p role="alert" className="error">{error || datasets.error || checkpoints.error}</p>}
    {!job && <button className="button primary" disabled={busy || !model || !config.dataset || Boolean(model.checkpoint_required && !config.checkpoint)} onClick={() => void start()}>{busy ? "Starting…" : "Process selected data"}</button>}
  </Panel>;
}

export function ProcessingSummary({ output }: { output?: Output }) {
  if (!output) return <span className="muted">Pending</span>;
  if (output.type === "classification") {
    const classes = (output.value as { classes?: { label: string; probability: number }[] })?.classes ?? [];
    const first = Math.max(0, Math.min(1, classes[0]?.probability ?? 0));
    const winner = classes.reduce((best, item) => item.probability > best.probability ? item : best, classes[0]);
    return <div className="processing-prediction"><svg viewBox="0 0 100 100" width="64" height="64" role="img" aria-label={classes.map(c => `${c.label}: ${(c.probability * 100).toFixed(1)}%`).join(", ")}>
      <circle cx="50" cy="50" r="39" fill="none" stroke="#e99167" strokeWidth="10" />
      <circle cx="50" cy="50" r="39" fill="none" stroke="#4cbfa5" strokeWidth="10" pathLength="100" strokeDasharray={`${first * 100} 100`} transform="rotate(-90 50 50)" />
      <text x="50" y="55" textAnchor="middle" fill="currentColor" fontSize="17">{((winner?.probability ?? 0) * 100).toFixed(0)}%</text>
    </svg><span>{winner?.label}</span></div>;
  }
  if (output.type === "image" && output.url) return <img className="processing-thumbnail" src={apiUrl(output.url)} alt="Generated output" />;
  return <span className="processing-number">{typeof output.value === "object" ? JSON.stringify(output.value) : String(output.value ?? "")}</span>;
}

function RecordView({ runId, recordId }: { runId: string; recordId: string }) {
  const result = useResource<RecordDetail>(`/processing/runs/${runId}/records/${recordId}`, `${runId}/${recordId}`, 2500);
  const record = result.data;
  return <Panel title={`Record ${recordId}`}>
    <Link to={`/processing/runs/${runId}`}>← All selected records</Link>
    <State loading={result.loading} error={result.error} />
    {record && <>
      <p>{record.source} · Record {record.source_index}</p>
      {typeof record.data.sequence === "string" && <><h3>Input sequence</h3><pre className="processing-sequence">{record.data.sequence}</pre></>}
      {record.error && <p role="alert" className="error">{record.error}</p>}
      {record.output ? <OutputRenderer output={record.output} /> : !record.error && <p className="muted">Waiting for this record to finish…</p>}
      <details><summary>Full input record</summary><pre className="metadata">{JSON.stringify(record.data, null, 2)}</pre></details>
    </>}
  </Panel>;
}

function Results({ job }: { job: Job }) {
  const [page, setPage] = useState(1);
  const navigate = useNavigate();
  const result = useResource<{ items: RecordRow[]; total: number; page_size: number }>(`/processing/runs/${job.id}/records?page=${page}`, `${job.id}/${page}`, active(job.status) ? 2000 : 0);
  const pages = Math.max(1, Math.ceil((result.data?.total ?? 0) / 50));
  return <Panel title="Selected records" meta={`${job.completed} / ${job.total} PROCESSED · ${job.failed} FAILED`}>
    <State loading={result.loading} error={result.error} />
    <div className="processing-table-scroll"><table className="processing-table"><thead><tr><th>Record</th><th>Class</th><th>Input</th><th>Output</th></tr></thead>
      <tbody>{result.data?.items.map(row => <tr key={row.id} tabIndex={0} onClick={() => navigate(`/processing/runs/${job.id}/records/${row.id}`)} onKeyDown={event => { if (event.key === "Enter") navigate(`/processing/runs/${job.id}/records/${row.id}`); }}>
        <td><Link to={`/processing/runs/${job.id}/records/${row.id}`}>{row.label}</Link><small>{row.status}</small></td><td>{row.class}</td><td className="processing-input-preview">{row.preview}</td>
        <td>{row.error ? <span className="error">{row.error}</span> : <ProcessingSummary output={row.output} />}</td>
      </tr>)}</tbody></table></div>
    {job.status === "INDEXING" && <p className="muted">Indexing {job.indexed.toLocaleString()} records and selecting class subsets…</p>}
    <div className="processing-pagination"><button disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Previous</button><span>Page {page} of {pages}</span><button disabled={page >= pages} onClick={() => setPage(p => p + 1)}>Next</button></div>
  </Panel>;
}

export function DataProcessing() {
  const { runId, recordId } = useParams();
  const navigate = useNavigate();
  const models = useResource<{ models: ModelInfo[] }>("/inference/models", "processing");
  const jobs = useResource<{ runs: Job[] }>("/processing/runs", runId ?? "new", 2000);
  const job = jobs.data?.runs.find(item => item.id === runId);
  const [error, setError] = useState("");
  async function stop() {
    try { await post(`/processing/runs/${runId}/stop`, {}); }
    catch (exception) { setError((exception as Error).message); }
  }
  return <Page eyebrow="DATA / PROCESSING" title="Data processing" action={<Link className="button" to="/processing">New processing job</Link>}>
    <div className="processing-job-picker"><label>Saved processing job<select value={runId ?? ""} onChange={event => navigate(event.target.value ? `/processing/runs/${event.target.value}` : "/processing")}>
      <option value="">New job</option>{jobs.data?.runs.map(item => <option key={item.id} value={item.id}>{item.configuration.dataset} · {item.configuration.model} · {item.status} · {item.id.slice(0, 8)}</option>)}
    </select></label></div>
    <State loading={models.loading || Boolean(runId && jobs.loading)} error={models.error || jobs.error || error} />
    {runId && jobs.data && !job && <State error="Processing job not found" />}
    {models.data && (!runId || job) && <Settings key={runId ?? "new"} models={models.data.models} job={job} />}
    {job && <>
      <div className="processing-job-actions"><strong>{job.status}</strong><span>{job.completed} / {job.total} records processed</span>
        {active(job.status) ? <button disabled={job.status === "STOPPING"} onClick={() => void stop()}>Stop processing</button> : <>
          <a className="button" href={apiUrl(`/processing/runs/${job.id}/export/jsonl`)} download>Download JSONL</a>
          <a className="button" href={apiUrl(`/processing/runs/${job.id}/export/zip`)} download>Download outputs + manifest</a>
        </>}
      </div>
      {job.error && <p role="alert" className="error">{job.error}</p>}
      {recordId ? <RecordView runId={job.id} recordId={recordId} /> : <Results key={job.id} job={job} />}
    </>}
  </Page>;
}
