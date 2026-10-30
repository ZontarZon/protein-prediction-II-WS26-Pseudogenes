import { useEffect, useMemo, useRef, useState } from "react";

type MetricRow = { [key: string]: string | number | null };

function niceStep(span: number, desired: number) {
  if (!Number.isFinite(span) || span <= 0) return 1;
  const raw = span / Math.max(1, desired);
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  return (
    (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * power
  );
}

function ticks(min: number, max: number, desired: number) {
  if (min === max) {
    const step = niceStep(Math.abs(min) || 1, desired);
    min -= step / 2;
    max += step / 2;
  }
  const step = niceStep(max - min, desired);
  const first = Math.ceil(min / step) * step;
  const result: number[] = [];
  for (
    let value = first, count = 0;
    value <= max + step * 1e-9 && count < 50;
    value += step, count++
  )
    result.push(Number(value.toPrecision(12)));
  return result.length ? result : [min, max];
}

function format(value: number) {
  if (!Number.isFinite(value)) return "—";
  if (
    Math.abs(value) >= 1e6 ||
    (Math.abs(value) > 0 && Math.abs(value) < 0.001)
  )
    return value.toExponential(1);
  return Number(value.toPrecision(4)).toString();
}

export function MetricChart({
  name,
  rows,
  comparison,
}: {
  name: string;
  rows: MetricRow[];
  comparison?: string;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ width: 700, height: 340 });
  useEffect(() => {
    if (!svg.current) return;
    const observer = new ResizeObserver((entries) => {
      const bounds = entries[0]?.contentRect;
      if (bounds?.width && bounds.height)
        setSize({ width: bounds.width, height: bounds.height });
    });
    observer.observe(svg.current);
    return () => observer.disconnect();
  }, []);

  const points = useMemo(
    () =>
      rows
        .filter((row) => !row.phase || (name.startsWith("validation_") ? row.phase === "validation" : row.phase === "training"))
        .map((row) => ({
          x: Number(row.epoch_position ?? row.step ?? row.epoch ?? row.timestamp ?? 0),
          y: row[name],
        }))
        .filter(
          (point): point is { x: number; y: number } =>
            Number.isFinite(point.x) &&
            typeof point.y === "number" &&
            Number.isFinite(point.y),
        ),
    [name, rows],
  );
  const secondary = rows
    .filter((row) => !row.phase || row.phase === "validation")
    .map((row) => ({ x: Number(row.epoch_position ?? row.step ?? row.epoch ?? 0), y: comparison ? row[comparison] : null }))
    .filter((point): point is { x: number; y: number } => Number.isFinite(point.x) && typeof point.y === "number" && Number.isFinite(point.y));
  const color = name.startsWith("validation_") ? "#e99167" : undefined;
  const plot = {
    left: Math.min(72, size.width * 0.2),
    right: size.width - 20,
    top: 22,
    bottom: size.height - 48,
  };
  const bounds = [...points, ...secondary].reduce(
    (range, point) => ({
      xMin: Math.min(range.xMin, point.x),
      xMax: Math.max(range.xMax, point.x),
      yMin: Math.min(range.yMin, point.y),
      yMax: Math.max(range.yMax, point.y),
    }),
    { xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity },
  );
  const xMin = points.length + secondary.length ? bounds.xMin : 0;
  const xMax = points.length + secondary.length ? bounds.xMax : 1;
  const yMin = points.length + secondary.length ? bounds.yMin : 0;
  const yMax = points.length + secondary.length ? bounds.yMax : 1;
  const xPadding = xMin === xMax ? niceStep(Math.abs(xMin) || 1, 5) / 2 : 0;
  const yPadding = yMin === yMax ? niceStep(Math.abs(yMin) || 1, 5) / 2 : 0;
  const xDomainMin = xMin - xPadding,
    xDomainMax = xMax + xPadding;
  const yDomainMin = yMin - yPadding,
    yDomainMax = yMax + yPadding;
  const xRange = xDomainMax - xDomainMin || 1;
  const yRange = yDomainMax - yDomainMin || 1;
  const xTicks = ticks(
    xDomainMin,
    xDomainMax,
    Math.max(2, Math.floor((size.width - 95) / 100)),
  );
  const yTicks = ticks(
    yDomainMin,
    yDomainMax,
    Math.max(2, Math.floor((size.height - 75) / 55)),
  );
  const x = (value: number) =>
    plot.left + ((value - xDomainMin) * (plot.right - plot.left)) / xRange;
  const y = (value: number) =>
    plot.bottom - ((value - yDomainMin) * (plot.bottom - plot.top)) / yRange;
  const line = points
    .map((point, index) => `${index ? "L" : "M"} ${x(point.x)} ${y(point.y)}`)
    .join(" ");

  return (
    <div className="chart-card">
      <div className="panel-head">
        <h3>{name === "loss" ? "Training loss" : name.replaceAll("_", " ")}</h3>
        <span className="mono muted">{points.length} POINTS</span>
      </div>
      {comparison && <p className="muted">Training · teal / Validation · orange</p>}
      {points.length + secondary.length ? (
        <svg
          ref={svg}
          className="metric-chart"
          viewBox={`0 0 ${size.width} ${size.height}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`${name}${comparison ? ` and ${comparison}` : ""} history`}
        >
          {yTicks.map((value) => (
            <g key={`y-${value}`}>
              <path
                className="chart-grid-line"
                d={`M${plot.left} ${y(value)}H${plot.right}`}
              />
              <text
                className="chart-tick"
                x={plot.left - 10}
                y={y(value) + 4}
                textAnchor="end"
              >
                {format(value)}
              </text>
            </g>
          ))}
          {xTicks.map((value) => (
            <g key={`x-${value}`}>
              <path
                className="chart-grid-line vertical"
                d={`M${x(value)} ${plot.top}V${plot.bottom}`}
              />
              <text
                className="chart-tick"
                x={x(value)}
                y={plot.bottom + 20}
                textAnchor="middle"
              >
                {format(value)}
              </text>
            </g>
          ))}
          <path
            className="chart-axis-line"
            d={`M${plot.left} ${plot.top}V${plot.bottom}H${plot.right}`}
          />
          {points.length > 1 && <path className="chart-line" style={{ stroke: color }} d={line} />}
          {points.length === 1 && (
            <circle
              className="chart-point"
              style={{ fill: color }}
              cx={x(points[0].x)}
              cy={y(points[0].y)}
              r="4"
            />
          )}
          {secondary.length > 1 && <path className="chart-line" style={{ stroke: "#e99167" }} d={secondary.map((point, index) => `${index ? "L" : "M"} ${x(point.x)} ${y(point.y)}`).join(" ")} />}
          {secondary.length === 1 && <circle cx={x(secondary[0].x)} cy={y(secondary[0].y)} r="4" fill="#e99167" />}
          <text
            className="chart-axis-label"
            x={(plot.left + plot.right) / 2}
            y={size.height - 5}
            textAnchor="middle"
          >
            {rows.some((row) => row.epoch_position != null) ? "Epoch" : "Step"}
          </text>
          <text
            className="chart-axis-label"
            transform={`translate(17 ${(plot.top + plot.bottom) / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            {name.replaceAll("_", " ")}
          </text>
        </svg>
      ) : (
        <div className="chart-empty">
          No numeric metric history is available.
        </div>
      )}
    </div>
  );
}
