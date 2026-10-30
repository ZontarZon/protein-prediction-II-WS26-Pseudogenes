import { useParams } from "react-router-dom";
import { useResource } from "./api";
import { Page, Panel, State } from "./shared";
type StructureData = {
  model: string;
  source: string;
  components: { name: string; kind: string; bases: string[]; line: number }[];
  available: boolean;
};
export function Structure() {
  const { modelId = "" } = useParams(),
    { data, error, loading } = useResource<StructureData>(
      `/models/${encodeURIComponent(modelId)}/structure`,
      modelId,
    );
  return (
    <Page
      eyebrow="MODEL INSPECTOR / 03"
      title={data?.model ?? modelId}
      description="Structure parsed from the model source without executing it."
    >
      <Panel title="Architecture components" meta={data?.source}>
        <State
          loading={loading}
          error={error}
          empty="No class or function declarations could be parsed."
        />
        {data?.components.map((c, i) => (
          <div className="structure-row" key={`${c.name}-${i}`}>
            <span className="mono accent">
              {String(i + 1).padStart(2, "0")}
            </span>
            <div className="grow">
              <strong>{c.name}</strong>
              <small>
                {c.kind}
                {c.bases.length ? ` · extends ${c.bases.join(", ")}` : ""}
              </small>
            </div>
            <span className="mono muted">LINE {c.line}</span>
          </div>
        ))}
      </Panel>
    </Page>
  );
}
