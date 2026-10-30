import { useEffect, useState } from "react";
import { get } from "./api";
import { selectionEstimate } from "./datasetSelection";
import type { DatasetGroup } from "./datasetSelection";
import "./DatasetSelectionPreview.css";

type Statistics = {
  status: string; objective?: string; total?: number; groups?: DatasetGroup[];
  validation_parameter?: string; note?: string; error?: string;
};
export function DatasetSelectionPreview({ trainer, role, dataset, parameters, onChange }: {
  trainer: string; role: string; dataset: string; parameters: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) {
  const identity = `${trainer}/${role}/${dataset}`;
  const [state, setState] = useState<{ identity: string; statistics: Statistics }>();
  useEffect(() => {
    if (!dataset) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const statistics = await get<Statistics>(`/trainers/${encodeURIComponent(trainer)}/dataset-statistics?role=${encodeURIComponent(role)}&dataset=${encodeURIComponent(dataset)}`);
        if (!active) return;
        setState({ identity, statistics });
        if (["scanning", "busy"].includes(statistics.status)) timer = setTimeout(() => void load(), 1500);
      } catch (error) {
        if (active) setState({ identity, statistics: { status: "error", error: (error as Error).message } });
      }
    };
    void load();
    return () => { active = false; clearTimeout(timer); };
  }, [trainer, role, dataset, identity]);
  if (!dataset) return null;
  const statistics = state?.identity === identity ? state.statistics : undefined;
  if (!statistics || ["scanning", "busy"].includes(statistics.status))
    return <p className="muted" role="status">Counting training elements… This scans the dataset once; counts are cached until files change.</p>;
  if (statistics.status === "error") return <p role="alert">Dataset preview: {statistics.error}</p>;
  if (statistics.status === "unsupported") return <p className="muted">This trainer has no objective-aware dataset preview adapter.</p>;
  const estimate = selectionEstimate(statistics.groups ?? [], parameters, statistics.validation_parameter);
  return <div className="dataset-selection-preview">
    <p><strong>{statistics.objective} · {statistics.total?.toLocaleString()} available elements</strong></p>
    <div className="table-wrap"><table>
      <thead><tr><th>Class / group</th><th>Available</th><th>Subset count</th><th>Selected</th></tr></thead>
      <tbody>{estimate.groups.map((group) => <tr key={group.key}>
        <td>{group.label}</td><td>{group.count.toLocaleString()}</td>
        <td>{group.subset_parameter ? <input type="number" min="0" step="1" aria-label={`Subset count for ${group.label}`} value={String(parameters[group.subset_parameter] ?? 0)} onChange={(event) => onChange(group.subset_parameter!, event.target.value)} /> : "All"}</td>
        <td>{group.selected.toLocaleString()}</td>
      </tr>)}</tbody>
    </table></div>
    {estimate.groups.some((group) => group.subset_parameter) && <p className="muted">0 = all available entries. Positive counts cap each class independently. Each new run draws a fresh random subset, fixed for all its epochs and on resume. Validation is split from that subset.</p>}
    {estimate.valid ? <p role="status"><strong>Estimated batches per epoch: {estimate.trainingBatches.toLocaleString()} training{statistics.validation_parameter ? ` + ${estimate.validationBatches.toLocaleString()} validation` : ""}</strong><br />{estimate.total.toLocaleString()} selected · {estimate.training.toLocaleString()} training · {estimate.validation.toLocaleString()} validation. Final partial batches are included.</p> : <p role="alert">Enter nonnegative whole subset counts and a positive whole batch size.</p>}
    {statistics.note && <small className="muted">{statistics.note}</small>}
    {statistics.validation_parameter && estimate.groups.some((group) => group.selected < 2) && <p role="alert">Each class needs at least two distinct sequences to appear in both splits.</p>}
  </div>;
}
