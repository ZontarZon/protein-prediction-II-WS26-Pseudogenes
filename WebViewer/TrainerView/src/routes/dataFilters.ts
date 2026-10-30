export type DataNode = {
  type: string;
  label?: string;
  value?: unknown;
  url?: string;
  error?: string;
  width?: number;
  height?: number;
  children?: DataNode[];
  items?: DataNode[];
  item?: DataNode;
  columns?: string[];
  rows?: DataNode[][];
};
export type FieldFilter = { column: string; operation: string; value: string };

export function nodeText(node: DataNode): string {
  return [node.value == null ? "" : typeof node.value === "object" ? JSON.stringify(node.value) : String(node.value),
    ...(node.children ?? node.items ?? []).map((child) => `${child.label ?? ""} ${nodeText(child)}`)].join(" ");
}

export function filterRows(source: DataNode[][], filters: FieldFilter[], query: string, caseSensitive: boolean, sort: string, descending: boolean): DataNode[][] {
  const normalize = (text: string) => caseSensitive ? text : text.toLocaleLowerCase();
  const rows = source.filter((row) => {
    if (!row.some((cell) => normalize(nodeText(cell)).includes(normalize(query)))) return false;
    return filters.every((filter) => {
      const index = Number(filter.column);
      const cell = row[index];
      const text = normalize(cell ? nodeText(cell) : "");
      const value = normalize(filter.value);
      switch (filter.operation) {
        case "equals": return text === value;
        case "not_contains": return !text.includes(value);
        case "starts": return text.startsWith(value);
        case "ends": return text.endsWith(value);
        case "empty": return !text.trim();
        case "not_empty": return Boolean(text.trim());
        case "gte": return Boolean(text.trim() && value.trim()) && Number.isFinite(Number(text)) && Number.isFinite(Number(value)) && Number(text) >= Number(value);
        case "lte": return Boolean(text.trim() && value.trim()) && Number.isFinite(Number(text)) && Number.isFinite(Number(value)) && Number(text) <= Number(value);
        default: return text.includes(value);
      }
    });
  });
  if (sort !== "") rows.sort((a, b) => {
    const first = nodeText(a[Number(sort)] ?? { type: "text" });
    const second = nodeText(b[Number(sort)] ?? { type: "text" });
    return first.localeCompare(second, undefined, { numeric: true, sensitivity: caseSensitive ? "variant" : "base" }) * (descending ? -1 : 1);
  });
  return rows;
}

export function filterNode(node: DataNode, query: string, field: string, inherited = false): DataNode | null {
  const labelMatches = inherited || !field || (node.label ?? "").toLowerCase().includes(field.toLowerCase());
  const valueMatches = !query || nodeText(node).toLowerCase().includes(query.toLowerCase());
  const children = node.children?.map((child) => filterNode(child, query, field, labelMatches && Boolean(field))).filter((child): child is DataNode => child !== null);
  const items = node.items?.map((child) => filterNode(child, query, field, labelMatches && Boolean(field))).filter((child): child is DataNode => child !== null);
  if (children?.length || items?.length) return { ...node, children, items };
  return labelMatches && valueMatches && !node.children && !node.items ? node : null;
}
