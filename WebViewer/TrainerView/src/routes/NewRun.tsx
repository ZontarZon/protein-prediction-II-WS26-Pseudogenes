import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { get, post, useResource } from "./api";
import { Page, Panel, State } from "./shared";
import { DatasetSelectionPreview } from "./DatasetSelectionPreview";

type Field = {
  key: string;
  label: string;
  description: string;
  required: boolean;
  type?: string;
  default?: unknown;
  options?: (string | number)[];
  choices?: string[];
  interface?: string;
  advanced?: boolean;
  dataset_control?: boolean;
  minimum?: number;
  maximum?: number;
};
type Model = {
  id: string;
  name: string;
  interfaces?: string[];
  supports_freezing?: boolean;
};
type Dataset = { id: string; name: string; itemCount: number };
type Trainer = { id: string; name: string };
type Device = {
  id: string;
  type: string;
  backend: string;
  name: string;
  label: string;
  index?: number;
};
type Schema = {
  id: string;
  models: Field[];
  datasets: Field[];
  hyperparameters: Field[];
};
type Weight = {
  id: string;
  model: string;
  name: string;
  source_checkpoint?: string;
  source_training_run?: string;
  created_at: number;
  format: string;
  origin?: string;
};
type Initialization = { mode: string; weight_id?: string; freeze: boolean };

function ParameterControl({
  field,
  value,
  onChange,
}: {
  field: Field;
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
  const type =
    field.type === "integer" || field.type === "float" ? "number" : "text";
  return (
    <input
      type={type}
      step={field.type === "float" ? "any" : undefined}
      min={field.minimum}
      max={field.maximum}
      value={String(value ?? "")}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function NewRun() {
  const trainers = useResource<{ trainers: Trainer[] }>(
    "/trainers",
    "trainers",
  );
  const models = useResource<{ models: Model[] }>("/models", "models");
  const datasets = useResource<{ datasets: Dataset[] }>(
    "/datasets",
    "datasets",
  );
  const weights = useResource<{ weights: Weight[] }>("/weights", "weights");
  const devices = useResource<{ devices: Device[]; default_device: string }>(
    "/devices",
    "devices",
  );
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const requestedModel = params.get("model") ?? "";
  const initWeight = params.get("initWeight");
  const initCommitted = params.get("initCommitted");
  const initRun = params.get("initRun");
  const initCheckpoint = params.get("initCheckpoint");
  const resumeRun = params.get("resumeRun");
  const resumeCheckpoint = params.get("resumeCheckpoint");
  const [trainer, setTrainer] = useState(params.get("trainer") ?? "");
  const [selectedDevice, setSelectedDevice] = useState("");
  const device = selectedDevice || devices.data?.default_device || "";
  const [schemaState, setSchemaState] = useState<{
    trainer: string;
    schema: Schema;
  }>();
  const schema =
    schemaState?.trainer === trainer ? schemaState.schema : undefined;
  const [modelValues, setModelValues] = useState<Record<string, string>>({});
  const [datasetValues, setDatasetValues] = useState<Record<string, string>>(
    {},
  );
  const [hyperparameters, setHyperparameters] = useState<
    Record<string, unknown>
  >({});
  const [initialization, setInitialization] = useState<
    Record<string, Initialization>
  >({});
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!trainer) return;
    let active = true;
    get<Schema>(`/trainers/${encodeURIComponent(trainer)}/schema`)
      .then((value) => {
        if (!active) return;
        const selectedModels = Object.fromEntries(
          value.models.map((field) => {
            const candidate = models.data?.models.find(
              (item) => item.id === requestedModel,
            );
            const compatible =
              candidate &&
              (!field.choices || field.choices.includes(candidate.id)) &&
              (!field.interface ||
                candidate.interfaces?.includes(field.interface));
            return [field.key, compatible ? candidate.id : ""];
          }),
        );
        const role = value.models.find(
          (field) => selectedModels[field.key],
        )?.key;
        const hasInit = Boolean(
          initWeight || initCommitted || (initRun && initCheckpoint),
        );
        setSchemaState({ trainer, schema: value });
        setModelValues(selectedModels);
        setDatasetValues(
          Object.fromEntries(value.datasets.map((field) => [field.key, ""])),
        );
        setHyperparameters(
          Object.fromEntries(
            value.hyperparameters.map((field) => [
              field.key,
              field.default ?? "",
            ]),
          ),
        );
        setInitialization(
          role && hasInit
            ? {
                [role]: {
                  mode: initWeight ? "weight" : "checkpoint",
                  weight_id: initWeight ?? undefined,
                  freeze: false,
                },
              }
            : {},
        );
        setMessage("");
      })
      .catch((error) => active && setMessage((error as Error).message));
    return () => {
      active = false;
    };
  }, [trainer, requestedModel, models.data, initCheckpoint, initCommitted, initRun, initWeight]);

  function parameterValue(field: Field) {
    const value = hyperparameters[field.key];
    if (
      (field.type === "integer" || field.type === "float") &&
      value !== "" &&
      value != null
    )
      return Number(value);
    return value;
  }
  function initializationConfig(): Record<string, Record<string, unknown>> {
    const result: Record<string, Record<string, unknown>> = {};
    for (const [role, value] of Object.entries(initialization)) {
      if (value.mode === "weight" && value.weight_id)
        result[role] = { weight_id: value.weight_id, freeze: value.freeze };
      else if (
        value.mode === "checkpoint" &&
        initCommitted &&
        modelValues[role] === requestedModel
      )
        result[role] = { committed_id: initCommitted, freeze: value.freeze };
      else if (
        value.mode === "checkpoint" &&
        initRun &&
        initCheckpoint &&
        modelValues[role] === requestedModel
      )
        result[role] = {
          checkpoint: { run_id: initRun, checkpoint: initCheckpoint },
          freeze: value.freeze,
        };
    }
    return result;
  }
  async function start() {
    if (!schema) return;
    const init = initializationConfig();
    const config = {
      trainer,
      device,
      models: modelValues,
      datasets: datasetValues,
      hyperparameters: Object.fromEntries(
        schema.hyperparameters.map((field) => [
          field.key,
          parameterValue(field),
        ]),
      ),
      ...(Object.keys(init).length ? { initialization: init } : {}),
      ...(resumeRun && resumeCheckpoint
        ? {
            resume_checkpoint: {
              run_id: resumeRun,
              checkpoint: resumeCheckpoint,
            },
          }
        : {}),
    };
    setBusy(true);
    setMessage("Validating trainer configuration…");
    try {
      await post("/training/validate", config);
      setMessage("Starting training run…");
      const run = await post<{ id: string }>("/runs", config);
      navigate(`/runs/${run.id}`);
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const canStart = Boolean(
    schema &&
    Boolean(devices.data?.devices.some((item) => item.id === device)) &&
    schema.models.every((field) => !field.required || modelValues[field.key]) &&
    schema.datasets.every(
      (field) => !field.required || datasetValues[field.key],
    ) &&
    schema.hyperparameters.every((field) => {
      const value = parameterValue(field);
      return (
        (!(field.required || field.dataset_control) || (value !== "" && value != null)) &&
        (typeof value !== "number" || (value >= (field.minimum ?? -Infinity) && value <= (field.maximum ?? Infinity))) &&
        (!(field.type === "integer") ||
          value == null ||
          value === "" ||
          Number.isInteger(value)) &&
        (!(field.type === "float" || field.type === "integer") ||
          value == null ||
          value === "" ||
          Number.isFinite(value))
      );
    }) &&
    trainer,
  );
  const updateInit = (role: string, value: Initialization) =>
    setInitialization((current) => ({ ...current, [role]: value }));
  const preview = {
    trainer: trainer || null,
    device,
    models: modelValues,
    datasets: datasetValues,
    hyperparameters: schema
      ? Object.fromEntries(
          schema.hyperparameters.map((field) => [
            field.key,
            parameterValue(field),
          ]),
        )
      : {},
    ...(Object.keys(initializationConfig()).length
      ? { initialization: initializationConfig() }
      : {}),
    ...(resumeRun && resumeCheckpoint
      ? {
          resume_checkpoint: {
            run_id: resumeRun,
            checkpoint: resumeCheckpoint,
          },
        }
      : {}),
  };

  return (
    <Page
      eyebrow="RUN CONFIGURATION / 07"
      title="New training run"
      description="Configure trainer requirements, model initialization, and hyperparameters."
    >
      <Panel title="Computing device" meta="BACKEND HOST">
        <label className="field">Training device
          <select value={device} onChange={(event) => setSelectedDevice(event.target.value)}>
            {devices.data?.devices.map((item) => <option key={item.id} value={item.id}>{item.label || item.name} ({item.id})</option>)}
          </select>
        </label>
        <State loading={devices.loading} error={devices.error} />
      </Panel>
      <div className="two-col">
        <div>
          <Panel title="Trainer configuration" meta="SCHEMA DRIVEN">
            <State
              loading={
                trainers.loading ||
                models.loading ||
                datasets.loading ||
                weights.loading
              }
              error={
                trainers.error ||
                models.error ||
                datasets.error ||
                weights.error
              }
            />
            <label className="field">
              Trainer
              <select
                value={trainer}
                onChange={(event) => setTrainer(event.target.value)}
              >
                <option value="">Select trainer</option>
                {trainers.data?.trainers.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            {resumeRun && resumeCheckpoint && (
              <div className="resume-notice">
                Resume full training state from{" "}
                <span className="mono">
                  {resumeRun.slice(0, 10)} / {resumeCheckpoint}
                </span>
              </div>
            )}
            {schema && (
              <>
                {schema.models.length > 0 && (
                  <div className="config-section">
                    <h3>Models and initialization</h3>
                    {schema.models.map((field) => {
                      const selected = models.data?.models.find(
                        (item) => item.id === modelValues[field.key],
                      );
                      const init = initialization[field.key] ?? {
                        mode: "random",
                        freeze: false,
                      };
                      const compatibleWeights = (
                        weights.data?.weights ?? []
                      ).filter((item) => item.model === modelValues[field.key]);
                      return (
                        <div className="config-section" key={field.key}>
                          <label className="field">
                            <span>
                              {field.label}
                              {field.required && <em> REQUIRED</em>}
                            </span>
                            {field.description && (
                              <small>{field.description}</small>
                            )}
                            <select
                              value={modelValues[field.key] ?? ""}
                              onChange={(event) => {
                                setModelValues((current) => ({
                                  ...current,
                                  [field.key]: event.target.value,
                                }));
                                updateInit(field.key, {
                                  mode: "random",
                                  freeze: false,
                                });
                              }}
                            >
                              <option value="">
                                Select {field.label.toLowerCase()}
                              </option>
                              {(models.data?.models ?? [])
                                .filter(
                                  (item) =>
                                    (!field.choices ||
                                      field.choices.includes(item.id)) &&
                                    (!field.interface ||
                                      item.interfaces?.includes(
                                        field.interface,
                                      )),
                                )
                                .map((item) => (
                                  <option key={item.id} value={item.id}>
                                    {item.name}
                                  </option>
                                ))}
                            </select>
                          </label>
                          {selected && (
                            <>
                              <label className="field">
                                Initialize model
                                <select
                                  value={init.mode}
                                  onChange={(event) =>
                                    updateInit(field.key, {
                                      ...init,
                                      mode: event.target.value,
                                      weight_id: undefined,
                                    })
                                  }
                                >
                                  <option value="random">
                                    Random initialization
                                  </option>
                                  <option value="weight">
                                    Saved model weights
                                  </option>
                                  {(initCommitted ||
                                    (initRun && initCheckpoint)) &&
                                    selected.id === requestedModel && (
                                      <option value="checkpoint">
                                        Selected checkpoint
                                      </option>
                                    )}
                                </select>
                              </label>
                              {init.mode === "weight" && (
                                <label className="field">
                                  Compatible weight artifact
                                  <select
                                    value={init.weight_id ?? ""}
                                    onChange={(event) =>
                                      updateInit(field.key, {
                                        ...init,
                                        weight_id: event.target.value,
                                      })
                                    }
                                  >
                                    <option value="">
                                      Select saved weights
                                    </option>
                                    {compatibleWeights.map((item) => (
                                      <option key={item.id} value={item.id}>
                                        {item.name} · {item.id.slice(0, 8)} ·{" "}
                                        {new Date(
                                          item.created_at * 1000,
                                        ).toLocaleDateString()}
                                      </option>
                                    ))}
                                  </select>
                                  {init.weight_id && (
                                    <small>
                                      {(() => {
                                        const item = compatibleWeights.find(
                                          (weight) =>
                                            weight.id === init.weight_id,
                                        );
                                        return item
                                          ? `Source ${item.source_checkpoint ?? item.origin ?? "import"} · ${item.format}`
                                          : "";
                                      })()}
                                    </small>
                                  )}
                                </label>
                              )}
                              {init.mode === "checkpoint" && (
                                <div className="field">
                                  <small>
                                    Initialize from checkpoint weights only.
                                    Optimizer and scheduler state will not be
                                    resumed.
                                  </small>
                                </div>
                              )}
                              {init.mode !== "random" && (
                                <label className="field">
                                  <span>Freeze pretrained weights</span>
                                  <input
                                    type="checkbox"
                                    disabled={!selected.supports_freezing}
                                    checked={init.freeze}
                                    onChange={(event) =>
                                      updateInit(field.key, {
                                        ...init,
                                        freeze: event.target.checked,
                                      })
                                    }
                                  />
                                  {!selected.supports_freezing && (
                                    <small>
                                      This model does not expose a freezing
                                      interface.
                                    </small>
                                  )}
                                </label>
                              )}
                            </>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
                {schema.datasets.length > 0 && (
                  <div className="config-section">
                    <h3>Datasets</h3>
                    {schema.datasets.map((field) => (
                      <div key={field.key}>
                        <label className="field">
                          <span>
                            {field.label}
                            {field.required && <em> REQUIRED</em>}
                          </span>
                          {field.description && (
                            <small>{field.description}</small>
                          )}
                          <select
                            value={datasetValues[field.key] ?? ""}
                            onChange={(event) =>
                              setDatasetValues((current) => ({
                                ...current,
                                [field.key]: event.target.value,
                              }))
                            }
                          >
                            <option value="">
                              Select {field.label.toLowerCase()}
                            </option>
                            {datasets.data?.datasets.map((item) => (
                              <option key={item.id} value={item.id}>
                                {item.name} · {item.itemCount.toLocaleString()}{" "}
                                items
                              </option>
                            ))}
                          </select>
                        </label>
                        <DatasetSelectionPreview trainer={trainer} role={field.key} dataset={datasetValues[field.key] ?? ""} parameters={hyperparameters} onChange={(key, value) => setHyperparameters((current) => ({ ...current, [key]: value }))} />
                      </div>
                    ))}
                  </div>
                )}
                {schema.hyperparameters.filter((field) => !field.advanced && !field.dataset_control)
                  .length > 0 && (
                  <div className="config-section">
                    <h3>Hyperparameters</h3>
                    <div className="form-grid">
                      {schema.hyperparameters
                        .filter((field) => !field.advanced && !field.dataset_control)
                        .map((field) => (
                          <label className="field" key={field.key}>
                            <span>
                              {field.label}
                              {field.required && <em> REQUIRED</em>}
                            </span>
                            {field.description && (
                              <small>{field.description}</small>
                            )}
                            <ParameterControl
                              field={field}
                              value={
                                hyperparameters[field.key] ??
                                field.default ??
                                ""
                              }
                              onChange={(value) =>
                                setHyperparameters((current) => ({
                                  ...current,
                                  [field.key]: value,
                                }))
                              }
                            />
                          </label>
                        ))}
                    </div>
                  </div>
                )}
                {schema.hyperparameters.some((field) => field.advanced) && (
                  <details className="config-section advanced-settings">
                    <summary>Advanced performance settings</summary>
                    <div className="form-grid">
                      {schema.hyperparameters
                        .filter((field) => field.advanced)
                        .map((field) => (
                          <label className="field" key={field.key}>
                            <span>
                              {field.label}
                              {field.required && <em> REQUIRED</em>}
                            </span>
                            {field.description && (
                              <small>{field.description}</small>
                            )}
                            <ParameterControl
                              field={field}
                              value={
                                hyperparameters[field.key] ??
                                field.default ??
                                ""
                              }
                              onChange={(value) =>
                                setHyperparameters((current) => ({
                                  ...current,
                                  [field.key]: value,
                                }))
                              }
                            />
                          </label>
                        ))}
                    </div>
                  </details>
                )}
              </>
            )}
            <button
              className="button primary"
              disabled={!canStart || busy}
              onClick={() => void start()}
            >
              {busy ? "Starting…" : "Validate & start run"}
            </button>
            {message && <p className="form-message">{message}</p>}
          </Panel>
        </div>
        <Panel title="Configuration preview" meta="JSON">
          <pre className="metadata">{JSON.stringify(preview, null, 2)}</pre>
          {schema && (
            <div className="schema-summary">
              <span className="mono">{schema.models.length} MODEL ROLES</span>
              <span className="mono">
                {schema.datasets.length} DATASET ROLES
              </span>
              <span className="mono">
                {schema.hyperparameters.length} PARAMETERS
              </span>
            </div>
          )}
        </Panel>
      </div>
    </Page>
  );
}
