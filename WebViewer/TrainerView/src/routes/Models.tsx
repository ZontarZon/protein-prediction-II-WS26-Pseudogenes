import { useState } from "react";
import { Link } from "react-router-dom";
import { remove, useResource } from "./api";
import { Page, Panel, State } from "./shared";
type Model = {
  id: string;
  name: string;
  path: string;
  type: string;
  checkpoints: {
    id: string;
    name: string;
    metadata: Record<string, unknown>;
  }[];
};
type Payload = { models: Model[] };
export function Models() {
  const [refresh, setRefresh] = useState(0),
    [open, setOpen] = useState(""),
    [error, setError] = useState("");
  const {
    data,
    error: loadError,
    loading,
  } = useResource<Payload>("/models", `models-${refresh}`);
  async function deleteCheckpoint(id: string, name: string) {
    if (
      !window.confirm(
        `Delete committed checkpoint “${name}” (${id})? This permanently removes the saved checkpoint. Extracted model weights and inference results will remain.`,
      )
    )
      return;
    try {
      await remove(`/checkpoints/${encodeURIComponent(id)}`);
      setError("");
      setRefresh((value) => value + 1);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }
  return (
    <Page
      eyebrow="ASSET REGISTRY / 02"
      title="Models"
      description="Models, committed checkpoints, and reusable weights in your workspace."
    >
      <Panel
        title="Model registry"
        meta={`${data?.models.length ?? 0} MODEL FILES`}
      >
        <State
          loading={loading}
          error={error || loadError}
          empty={
            data?.models.length === 0
              ? "No model files found in ModelTraining/Models."
              : undefined
          }
        />
        {data?.models.map((m) => (
          <div className="model-entry" key={m.id}>
            <div
              className="model-row"
              onClick={() => setOpen(open === m.id ? "" : m.id)}
            >
              <span className="model-glyph">◈</span>
              <div className="grow">
                <strong>{m.name}</strong>
                <small>
                  {m.path} · {m.type.toUpperCase()}
                </small>
              </div>
              <span className="badge">{m.checkpoints.length} COMMITTED</span>
              <Link
                className="icon-button"
                title="Model structure"
                to={`/models/${encodeURIComponent(m.id)}/structure`}
                onClick={(e) => e.stopPropagation()}
              >
                ⓘ
              </Link>
              <Link
                className="button small"
                to={`/new-run?model=${encodeURIComponent(m.id)}`}
                onClick={(e) => e.stopPropagation()}
              >
                Train ↗
              </Link>
              <span className="chevron">{open === m.id ? "−" : "+"}</span>
            </div>
            {open === m.id && (
              <div className="checkpoint-list">
                <div className="table-head">
                  <span>CHECKPOINT</span>
                  <span>PERFORMANCE</span>
                  <span>ACTIONS</span>
                  <span></span>
                </div>
                {m.checkpoints.length ? (
                  m.checkpoints.map((cp) => (
                    <div className="checkpoint-row" key={cp.id}>
                      <span className="mono">{cp.name}</span>
                      <span>
                        {Object.entries(cp.metadata)
                          .filter(([key]) =>
                            [
                              "step",
                              "epoch",
                              "loss",
                              "trainer",
                              "created_at",
                            ].includes(key),
                          )
                          .map(
                            ([key, value]) =>
                              `${key}: ${key === "created_at" && typeof value === "number" ? new Date(value * 1000).toLocaleString() : value}`,
                          )
                          .join(" · ") || "Performance metadata unavailable"}
                      </span>
                      <Link
                        className="text-link"
                        to={`/checkpoints/${encodeURIComponent(cp.id)}`}
                      >
                        View ↗
                      </Link>
                      <button
                        className="button danger small"
                        onClick={() => void deleteCheckpoint(cp.id, cp.name)}
                      >
                        Delete
                      </button>
                    </div>
                  ))
                ) : (
                  <div className="state compact">
                    No saved checkpoints for this model.
                  </div>
                )}
                <div className="state compact">
                  <Link
                    className="text-link"
                    to={`/weights?model=${encodeURIComponent(m.id)}`}
                  >
                    Manage model weights ↗
                  </Link>
                </div>
              </div>
            )}
          </div>
        ))}
      </Panel>
      <div className="control-row">
        <Link className="button" to="/network-builder">Build / edit a network</Link>
        <Link className="button" to="/weights">
          Manage / import model weights
        </Link>
      </div>
    </Page>
  );
}
