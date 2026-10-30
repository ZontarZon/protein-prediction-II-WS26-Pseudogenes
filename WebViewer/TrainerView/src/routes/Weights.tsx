import { useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { apiUrl, useResource } from "./api";
import { Page, Panel, State } from "./shared";

type Model = { id: string; name: string; weight_capable?: boolean };
type Weight = {
  id: string;
  model: string;
  name: string;
  model_config: Record<string, unknown>;
  source_checkpoint?: string;
  source_training_run?: string;
  created_at: number;
  format: string;
  trainer?: string;
  dataset?: Record<string, string>;
  origin?: string;
};
type Payload = { weights: Weight[] };

export function Weights() {
  const [params] = useSearchParams();
  const [selectedModel, setSelectedModel] = useState(params.get("model") ?? "");
  const [file, setFile] = useState<File>();
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const models = useResource<{ models: Model[] }>("/models", "weight-models");
  const payload = useResource<Payload>("/weights", `weights/${refresh}`);
  const items = useMemo(
    () =>
      payload.data?.weights.filter(
        (item) => !selectedModel || item.model === selectedModel,
      ) ?? [],
    [payload.data, selectedModel],
  );
  async function upload() {
    if (!file || !selectedModel) return;
    setBusy(true);
    setMessage("Validating and importing tensor weights…");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("model", selectedModel);
      const response = await fetch(apiUrl("/weights/import"), {
        method: "POST",
        body: form,
      });
      const value = await response.json();
      if (!response.ok)
        throw new Error(value.error ?? `Import failed (${response.status})`);
      setMessage(`Imported ${value.name} · ${value.id}`);
      setFile(undefined);
      setRefresh((current) => current + 1);
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page
      eyebrow="WEIGHT LIBRARY / 10"
      title="Model weights"
      description="Manage reusable model-only parameters, extracted from checkpoints or safely imported."
    >
      <div className="two-col">
        <Panel title="Import external weights" meta="SAFE TENSOR LOAD">
          <State loading={models.loading} error={models.error} />
          <label className="field">
            Target model
            <select
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
            >
              <option value="">Select a registered model</option>
              {models.data?.models
                .filter((model) => model.weight_capable)
                .map((model) => (
                  <option value={model.id} key={model.id}>
                    {model.name}
                  </option>
                ))}
            </select>
          </label>
          <label className="field">
            PyTorch weight file (.pt / .pth)
            <input
              type="file"
              accept=".pt,.pth"
              onChange={(event: ChangeEvent<HTMLInputElement>) =>
                setFile(event.target.files?.[0])
              }
            />
            <small>
              Only tensor state dictionaries are loaded; imported files are
              never executed.
            </small>
          </label>
          <button
            className="button primary"
            disabled={!selectedModel || !file || busy}
            onClick={() => void upload()}
          >
            {busy ? "Importing…" : "Validate & import weights"}
          </button>
          {message && <p className="form-message">{message}</p>}
        </Panel>
        <Panel title="Weight artifacts" meta={`${items.length} AVAILABLE`}>
          <State
            loading={payload.loading}
            error={payload.error}
            empty={
              !payload.loading && !payload.error && !items.length
                ? "No saved weights match this model yet. Extract weights from a checkpoint or import a compatible state dictionary."
                : undefined
            }
          />
          <label className="field">
            Filter by model
            <select
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
            >
              <option value="">All models</option>
              {models.data?.models.map((model) => (
                <option value={model.id} key={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
          {items.map((item) => (
            <article className="weight-card" key={item.id}>
              <div className="weight-card-head">
                <div>
                  <strong>{item.name}</strong>
                  <small>
                    {item.model} · {item.id}
                  </small>
                </div>
                <span className="badge">{item.origin ?? "extracted"}</span>
              </div>
              <div className="weight-meta">
                <span>
                  Created {new Date(item.created_at * 1000).toLocaleString()}
                </span>
                <span>Format · {item.format}</span>
                <span>
                  Source checkpoint ·{" "}
                  {item.source_checkpoint ?? "external import"}
                </span>
                <span>Source run · {item.source_training_run ?? "—"}</span>
              </div>
              <pre className="metadata">
                {JSON.stringify(
                  {
                    model_config: item.model_config,
                    trainer: item.trainer,
                    dataset: item.dataset,
                  },
                  null,
                  2,
                )}
              </pre>
              <Link
                className="button small"
                to={`/new-run?model=${encodeURIComponent(item.model)}&initWeight=${encodeURIComponent(item.id)}`}
              >
                Use in new training run ↗
              </Link>
            </article>
          ))}
        </Panel>
      </div>
    </Page>
  );
}
