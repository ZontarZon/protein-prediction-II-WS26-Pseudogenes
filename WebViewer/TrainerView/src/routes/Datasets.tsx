import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { apiUrl, useResource } from "./api";
import { Page, Panel, State } from "./shared";
import { DatasetContentsSummary } from "./DatasetContentsSummary";
import { filterRows, filterNode } from "./dataFilters";
import type { DataNode, FieldFilter } from "./dataFilters";

type Dataset = { id: string; name: string; itemCount: number };
type Data = { datasets: Dataset[] };
type Entry = { name: string; path: string; size: number };
type Entries = {
  entries: Entry[];
  itemCount: number;
  page: number;
  page_size: number;
  total_pages: number;
  unfiltered_total: number;
};
type Layout = { layout: DataNode };
type EntryView = { view: DataNode; searchable?: boolean; loader?: string; notice?: string; has_more?: boolean; next_offset?: number };

type RecordSearch = { query: string; fieldQuery: string; filters: FieldFilter[]; sort: string; descending: boolean; caseSensitive: boolean };
const EMPTY_SEARCH: RecordSearch = { query: "", fieldQuery: "", filters: [], sort: "", descending: false, caseSensitive: false };

function StructureNode({
  node,
  depth = 0,
}: {
  node: DataNode;
  depth?: number;
}) {
  const children = node.children ?? (node.item ? [node.item] : []);
  return (
    <div
      className="dataset-structure-node"
      style={{ marginLeft: depth ? 18 : 0 }}
    >
      <div>
        <strong>{node.label ?? "Value"}</strong>
        <span className="badge">{node.type}</span>
      </div>
      {children.length > 0 && (
        <div className="dataset-structure-children">
          {children.map((child, index) => (
            <StructureNode
              key={`${child.label}-${index}`}
              node={child}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ImageNode({ node }: { node: DataNode }) {
  const [failed, setFailed] = useState(false);
  return (
    <div className="dataset-image-view">
      <strong>{node.label}</strong>
      {node.width != null && node.height != null && (
        <span className="mono muted">
          {node.width} × {node.height}
        </span>
      )}
      {node.url && !failed ? (
        <img
          src={node.url.startsWith("data:image/") ? node.url : apiUrl(node.url)}
          alt={node.label ?? "Dataset image"}
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="state compact">
          {node.error ?? "Image is missing or could not be decoded."}
        </div>
      )}
    </div>
  );
}

function PreviewTable({ node, search, onSearch, loading = false }: { node: DataNode; search?: RecordSearch; onSearch?: (search: RecordSearch) => void; loading?: boolean }) {
  const [localSearch, setLocalSearch] = useState<RecordSearch>(EMPTY_SEARCH);
  const state = search ?? localSearch;
  const update = (next: RecordSearch) => (onSearch ?? setLocalSearch)(next);
  const { query, fieldQuery, filters, sort, descending, caseSensitive } = state;
  const setQuery = (query: string) => update({ ...state, query });
  const setFieldQuery = (fieldQuery: string) => update({ ...state, fieldQuery });
  const setFilters = (filters: FieldFilter[]) => update({ ...state, filters });
  const setSort = (sort: string) => update({ ...state, sort });
  const setDescending = (descending: boolean) => update({ ...state, descending });
  const setCaseSensitive = (caseSensitive: boolean) => update({ ...state, caseSensitive });
  const columns = node.columns ?? [];
  const selectable = [...new Set([...columns, ...filters.map((filter) => filter.column), ...(sort ? [sort] : [])])];
  const visible = columns.map((name, index) => ({ name, index })).filter(({ name }) => name.toLocaleLowerCase().includes(fieldQuery.toLocaleLowerCase()));
  const rows = onSearch ? (node.rows ?? []) : filterRows(node.rows ?? [], filters.map((filter) => ({ ...filter, column: String(columns.indexOf(filter.column)) })), query, caseSensitive, sort ? String(columns.indexOf(sort)) : "", descending);
  return <>
    <div className="preview-filters">
      <label className="field">Search all values<input type="search" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
      <label className="field">Show fields by name<input type="search" placeholder="e.g. sequence" value={fieldQuery} onChange={(event) => setFieldQuery(event.target.value)} /></label>
      <label className="field">Sort by<select value={sort} onChange={(event) => setSort(event.target.value)}><option value="">Original order</option>{selectable.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
      <label><input type="checkbox" checked={descending} onChange={(event) => setDescending(event.target.checked)} /> Descending</label>
      <label><input type="checkbox" checked={caseSensitive} onChange={(event) => setCaseSensitive(event.target.checked)} /> Match case</label>
    </div>
    {filters.map((filter, index) => <div className="preview-filter-row" key={index}>
      <label className="field">Field<select value={filter.column} onChange={(event) => setFilters(filters.map((item, i) => i === index ? { ...item, column: event.target.value } : item))}>{selectable.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
      <label className="field">Condition<select value={filter.operation} onChange={(event) => setFilters(filters.map((item, i) => i === index ? { ...item, operation: event.target.value } : item))}>
        <option value="contains">Contains / partial match</option><option value="not_contains">Does not contain</option><option value="equals">Exact match</option><option value="starts">Starts with</option><option value="ends">Ends with</option><option value="gte">Number ≥</option><option value="lte">Number ≤</option><option value="empty">Is empty</option><option value="not_empty">Is not empty</option>
      </select></label>
      <label className="field">Value<input disabled={["empty", "not_empty"].includes(filter.operation)} value={filter.value} onChange={(event) => setFilters(filters.map((item, i) => i === index ? { ...item, value: event.target.value } : item))} /></label>
      <button onClick={() => setFilters(filters.filter((_, i) => i !== index))}>Remove filter</button>
    </div>)}
    <div className="page-actions"><button disabled={!columns.length} onClick={() => setFilters([...filters, { column: columns[0], operation: "contains", value: "" }])}>Add field filter</button><button onClick={() => update(EMPTY_SEARCH)}>Clear filters</button></div>
    <p className="muted" role="status">{loading ? "Searching…" : `${rows.length} matching records shown.`} {onSearch ? "Filters combine with AND and search full records across this file, including later pages." : "Filters apply to the displayed table."}</p>
    <div className="dataset-preview-table"><table>
      <thead><tr>{visible.map(({ name, index }) => <th key={index} scope="col">{name}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>{visible.map(({ index: column }) => <td key={column}><DataNodeView node={row[column] ?? { type: "text", value: null }} /></td>)}</tr>)}</tbody>
    </table>{!loading && !rows.length && <p>No records match these filters.</p>}{!loading && !visible.length && <p>No fields match this name.</p>}</div>
  </>;
}

function SearchableEntry({ node }: { node: DataNode }) {
  const [query, setQuery] = useState("");
  const [field, setField] = useState("");
  if (node.type === "table") return <DataNodeView node={node} />;
  const filtered = query || field ? filterNode(node, query, field) : node;
  return <>
    <div className="preview-filters"><label className="field">Search values<input type="search" value={query} onChange={(event) => setQuery(event.target.value)} /></label><label className="field">Field name<input type="search" value={field} onChange={(event) => setField(event.target.value)} /></label><button onClick={() => { setQuery(""); setField(""); }}>Clear filters</button></div>
    <p className="muted">Search applies to the currently displayed entry.</p>
    {filtered ? <DataNodeView node={filtered} /> : <p role="status">No fields match these filters.</p>}
  </>;
}

function DataNodeView({ node }: { node: DataNode }) {
  if (node.type === "image") return <ImageNode key={node.url} node={node} />;
  if (node.type === "table") return <PreviewTable node={node} />;
  if (node.type === "group" || node.type === "object")
    return (
      <section className="dataset-data-group">
        <h3>{node.label ?? "Object"}</h3>
        <div>
          {(node.children ?? []).map((child, index) => (
            <DataNodeView key={`${child.label}-${index}`} node={child} />
          ))}
        </div>
      </section>
    );
  if (node.type === "list" || node.type === "array")
    return (
      <section className="dataset-data-field">
        <strong>{node.label ?? "List"}</strong>
        {node.items?.length ? (
          <ol>
            {node.items.map((item, index) => (
              <li key={index}>
                <DataNodeView node={item} />
              </li>
            ))}
          </ol>
        ) : (
          <span className="muted">Empty list</span>
        )}
      </section>
    );
  if (node.type === "file" || node.type === "blob")
    return (
      <div className="dataset-data-field">
        <strong>{node.label}</strong>
        <span>
          {String(node.value ?? node.error ?? "File data unavailable")}
        </span>
      </div>
    );
  if (node.type === "json")
    return (
      <div className="dataset-data-field">
        <strong>{node.label}</strong>
        <pre className="metadata">{JSON.stringify(node.value, null, 2)}</pre>
      </div>
    );
  return (
    <div className="dataset-data-field">
      <strong>{node.label ?? node.type}</strong>
      <span>
        {node.value === null
          ? "null"
          : typeof node.value === "object"
            ? JSON.stringify(node.value)
            : String(node.value ?? "")}
      </span>
    </div>
  );
}

export function Datasets() {
  const { datasetId, entryId } = useParams();
  const navigate = useNavigate();
  const { data, error, loading } = useResource<Data>("/datasets", "datasets");
  const [page, setPage] = useState(1);
  const [datasetSearch, setDatasetSearch] = useState("");
  const defaultFileFilters = { q: "", extension: "", min_bytes: "", max_bytes: "", sort: "name", order: "asc" };
  const [fileFilters, setFileFilters] = useState(defaultFileFilters);
  function updateFileFilter(key: keyof typeof fileFilters, value: string) {
    setFileFilters((current) => ({ ...current, [key]: value }));
    setPage(1);
  }
  const fileQuery = new URLSearchParams({ ...fileFilters, min_bytes: fileFilters.min_bytes || "0", page: String(page), page_size: "100" }).toString();
  const visibleDatasets = data?.datasets.filter((dataset) => dataset.name.toLowerCase().includes(datasetSearch.toLowerCase())) ?? [];
  const selected = datasetId ?? "";
  const identity = `${selected}/${entryId ?? ""}`;
  const [preview, setPreview] = useState({ identity: "", loader: "", offset: 0, search: EMPTY_SEARCH });
  const currentPreview = preview.identity === identity ? preview : { identity, loader: "", offset: 0, search: EMPTY_SEARCH };
  const previewLoaders = useResource<{ loaders: string[] }>("/datasets/preview-loaders", "preview-loaders");
  const previewQuery = new URLSearchParams({ loader: currentPreview.loader, offset: String(currentPreview.offset), search: JSON.stringify({
    query: currentPreview.search.query, filters: currentPreview.search.filters,
    case_sensitive: currentPreview.search.caseSensitive, sort: currentPreview.search.sort, descending: currentPreview.search.descending,
  }) }).toString();
  const items = useResource<Entries>(
    selected
      ? `/datasets/${encodeURIComponent(selected)}/entries?${fileQuery}`
      : "/datasets/_none/entries",
    `${selected}/${fileQuery}`,
  );
  const layout = useResource<Layout>(
    selected
      ? `/datasets/${encodeURIComponent(selected)}/layout`
      : "/datasets/_none/layout",
    selected,
  );
  const entry = useResource<EntryView>(
    selected && entryId
      ? `/datasets/${encodeURIComponent(selected)}/entries/${encodeURIComponent(entryId)}/view?${previewQuery}`
      : "/datasets/_none/entries/_none/view",
    `${identity}/${previewQuery}`,
  );
  const totalPages = items.data?.total_pages ?? 0;
  const pageNumbers = [
    ...new Set([1, totalPages, page - 2, page - 1, page, page + 1, page + 2]),
  ]
    .filter((value) => value >= 1 && value <= totalPages)
    .sort((a, b) => a - b);
  function chooseDataset(id: string) {
    setPage(1);
    setFileFilters(defaultFileFilters);
    navigate(`/datasets/${encodeURIComponent(id)}`);
  }
  if (selected && entryId)
    return (
      <Page
        eyebrow="DATA ENTRY / 05"
        title={entryId}
        description={`Structured entry from ${selected}.`}
        action={
          <Link
            className="button"
            to={`/datasets/${encodeURIComponent(selected)}`}
          >
            ← Back to {selected}
          </Link>
        }
      >
        <DatasetContentsSummary dataset={selected} />
        <Panel title="Dataset entry" meta={selected}>
          <label className="field">Preview using
            <select value={currentPreview.loader} onChange={(event) => setPreview({ identity, loader: event.target.value, offset: 0, search: EMPTY_SEARCH })}>
              <option value="">Automatic</option>
              {previewLoaders.data?.loaders.map((loader) => <option key={loader} value={loader}>{loader}</option>)}
            </select>
          </label>
          <State loading={entry.loading} error={entry.error || previewLoaders.error} />
          {entry.data?.loader && <p>Loader: {entry.data.loader} · {entry.data.view.rows?.length ?? 0} records shown · Offset {currentPreview.offset}</p>}
          {entry.data?.notice && <p className="muted">{entry.data.notice}</p>}
          {(!entry.data || entry.data.searchable) && <PreviewTable
            node={entry.data?.view ?? { type: "table", columns: [], rows: [] }}
            search={currentPreview.search} loading={entry.loading}
            onSearch={(search) => setPreview({ ...currentPreview, search, offset: search.fieldQuery !== currentPreview.search.fieldQuery ? currentPreview.offset : 0 })}
          />}
          {entry.data && !entry.data.searchable && <SearchableEntry key={`${identity}/${currentPreview.loader}`} node={entry.data.view} />}
          {entry.data?.searchable && <div className="pagination">
            <button disabled={entry.loading || currentPreview.offset === 0} onClick={() => setPreview({ ...currentPreview, offset: Math.max(0, currentPreview.offset - 20) })}>← Previous records</button>
            <button disabled={entry.loading || !entry.data.has_more} onClick={() => setPreview({ ...currentPreview, offset: entry.data?.next_offset ?? 0 })}>Next records →</button>
          </div>}
        </Panel>
      </Page>
    );
  return (
    <Page
      eyebrow="DATA CATALOG / 05"
      title="Datasets"
      description="Inspect dataset structure and browse individual records."
    >
      <Panel
        title="Available datasets"
        meta={`${data?.datasets.length ?? 0} DATASETS`}
      >
        <State
          loading={loading}
          error={error}
          empty={
            data?.datasets.length === 0
              ? "No dataset folders found. Add a dataset subfolder under one of the configured dataset roots."
              : undefined
          }
        />
        <label className="field">Search datasets<input type="search" value={datasetSearch} onChange={(event) => setDatasetSearch(event.target.value)} /></label>
        {data && !visibleDatasets.length && datasetSearch && <p role="status">No datasets match this name.</p>}
        {visibleDatasets.map((dataset) => (
          <button
            className={`dataset-row ${selected === dataset.id ? "selected" : ""}`}
            key={dataset.id}
            onClick={() => chooseDataset(dataset.id)}
          >
            <span className="model-glyph">▤</span>
            <strong className="grow">{dataset.name}</strong>
            <span className="badge">
              {dataset.itemCount.toLocaleString()} FILES
            </span>
            <span>↗</span>
          </button>
        ))}
      </Panel>
      {selected && (
        <>
          <DatasetContentsSummary dataset={selected} />
          <Panel title="Dataset Structure" meta={selected}>
            <State loading={layout.loading} error={layout.error} />
            {layout.data && (
              <div className="dataset-structure">
                <StructureNode node={layout.data.layout} />
              </div>
            )}
          </Panel>
          <Panel
            title="Dataset entries"
            meta={`${items.data?.itemCount ?? 0} / ${items.data?.unfiltered_total ?? 0} FILES`}
          >
            <div className="preview-filters">
              <label className="field">File name or path<input type="search" value={fileFilters.q} onChange={(event) => updateFileFilter("q", event.target.value)} /></label>
              <label className="field">File extension<input placeholder="e.g. fasta.gz" value={fileFilters.extension} onChange={(event) => updateFileFilter("extension", event.target.value)} /></label>
              <label className="field">Minimum bytes<input type="number" min="0" value={fileFilters.min_bytes} onChange={(event) => updateFileFilter("min_bytes", event.target.value)} /></label>
              <label className="field">Maximum bytes<input type="number" min="0" value={fileFilters.max_bytes} onChange={(event) => updateFileFilter("max_bytes", event.target.value)} /></label>
              <label className="field">Sort files by<select value={fileFilters.sort} onChange={(event) => updateFileFilter("sort", event.target.value)}><option value="name">Name</option><option value="size">Size</option></select></label>
              <label className="field">Order<select value={fileFilters.order} onChange={(event) => updateFileFilter("order", event.target.value)}><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
            </div>
            <button onClick={() => { setFileFilters(defaultFileFilters); setPage(1); }}>Clear file filters</button>
            <p className="muted">File filters search the entire selected dataset before pagination.</p>
            <State
              loading={items.loading}
              error={items.error}
              empty={
                items.data?.itemCount === 0
                  ? "No files match these filters."
                  : undefined
              }
            />
            {items.data?.entries.map((item) => (
              <div className="dataset-entry-row" key={item.name}>
                <span className="mono accent">
                  {item.name.toLowerCase().endsWith(".json")
                    ? "JSON"
                    : (item.name.split(".").at(-1)?.toUpperCase() ?? "DATA")}
                </span>
                <span className="grow">{item.name}</span>
                <span className="mono muted">
                  {(item.size / 1024).toFixed(1)} KB
                </span>
                <Link
                  className="button small"
                  to={`/datasets/${encodeURIComponent(selected)}/entries/${encodeURIComponent(item.name)}`}
                >
                  View
                </Link>
              </div>
            ))}
            {(items.data?.total_pages ?? 0) > 1 && (
              <div className="pagination">
                <button
                  className="button small"
                  disabled={page <= 1}
                  onClick={() => setPage((value) => Math.max(1, value - 1))}
                >
                  ← Previous
                </button>
                <span className="mono">
                  Page {items.data?.page ?? page} / {items.data?.total_pages}
                </span>
                <div className="pagination-pages">
                  {pageNumbers.map((value, index) => (
                    <span className="pagination-page" key={value}>
                      {index > 0 && value - pageNumbers[index - 1] > 1 && (
                        <span className="muted">…</span>
                      )}
                      <button
                        className={`button small ${value === page ? "primary" : ""}`}
                        onClick={() => setPage(value)}
                      >
                        {value}
                      </button>
                    </span>
                  ))}
                </div>
                <button
                  className="button small"
                  disabled={page >= (items.data?.total_pages ?? 1)}
                  onClick={() =>
                    setPage((value) =>
                      Math.min(items.data?.total_pages ?? 1, value + 1),
                    )
                  }
                >
                  Next →
                </button>
              </div>
            )}
          </Panel>
        </>
      )}
    </Page>
  );
}
