import { useEffect, useState } from "react";
import { get } from "./api";
import { Panel, State } from "./shared";
import "./DatasetSelectionPreview.css";

type Count = { key: string; label: string; count: number };
type Inventory = { status: string; total?: number; unit?: string; file_count?: number; groups?: Count[]; labels?: Count[]; note?: string; error?: string };
export function DatasetCounts({ inventory }: { inventory: Inventory }) {
  return <div className="dataset-selection-preview">
    <p className="dataset-record-total"><strong>{inventory.total?.toLocaleString()} total {inventory.unit === "records" ? "entries" : "files"}</strong></p>
    {inventory.unit === "records" && <p className="muted">Stored in {inventory.file_count?.toLocaleString()} files. Entry counts are independent of the selected subset and validation split.</p>}
    {!!inventory.groups?.length && <div className="table-wrap"><table>
      <thead><tr><th>Class</th><th>Entries</th></tr></thead>
      <tbody>{inventory.groups.map((group) => <tr key={group.key}><td>{group.label}</td><td><strong>{group.count.toLocaleString()}</strong></td></tr>)}</tbody>
    </table></div>}
    {!!inventory.labels?.length && <details open><summary>Entries per PE label</summary><div className="table-wrap"><table>
      <thead><tr><th>Label</th><th>Entries</th></tr></thead>
      <tbody>{inventory.labels.map((label) => <tr key={label.key}><td>{label.label}</td><td>{label.count.toLocaleString()}</td></tr>)}</tbody>
    </table></div></details>}
    <p className="muted">{inventory.note}</p>
  </div>;
}
export function DatasetContentsSummary({ dataset }: { dataset: string }) {
  const [state, setState] = useState<{ dataset: string; inventory: Inventory }>();
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const inventory = await get<Inventory>(`/datasets/${encodeURIComponent(dataset)}/statistics`);
        if (!active) return;
        setState({ dataset, inventory });
        if (["scanning", "busy"].includes(inventory.status)) timer = setTimeout(() => void load(), 1500);
      } catch (error) {
        if (active) setState({ dataset, inventory: { status: "error", error: (error as Error).message } });
      }
    };
    void load();
    return () => { active = false; clearTimeout(timer); };
  }, [dataset]);
  const inventory = state?.dataset === dataset ? state.inventory : undefined;
  return <Panel title="Dataset contents" meta={dataset}>
    {!inventory || ["scanning", "busy"].includes(inventory.status)
      ? <p role="status">Counting dataset entries and classes… The first scan may take a few seconds.</p>
      : inventory.status === "error" ? <State error={inventory.error} /> : <DatasetCounts inventory={inventory} />}
  </Panel>;
}
