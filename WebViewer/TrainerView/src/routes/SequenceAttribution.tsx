import { useEffect, useRef, useState } from "react";

export type Attribution = {
  sequence: string; scores: number[]; method: string; baseline: string;
  steps: number; chunk_size: number; completeness_error: number;
};

export function SequenceAttribution({ data }: { data: Attribution }) {
  const [start, setStart] = useState(0);
  const [view, setView] = useState<"fit" | "all">("fit");
  const [capacity, setCapacity] = useState(40);
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => {
      setCapacity(Math.max(1, Math.floor(entry.contentRect.width / 24)));
    });
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const length = Math.min(data.sequence.length, view === "fit" ? capacity : data.sequence.length);
  const maxStart = Math.max(0, data.sequence.length - length);
  const windowStart = Math.min(start, maxStart);
  const maximum = data.scores.reduce((max, value) => Math.max(max, Math.abs(value)), 0) || 1;
  const color = (score: number) => `rgba(${score >= 0 ? "76,191,165" : "233,145,103"},${Math.abs(score) / maximum})`;
  const end = windowStart + length;
  const bins = Math.min(200, data.sequence.length);
  return <section className="sequence-attribution" aria-label="Residue contribution heat map">
    <h3>Residue contributions</h3>
    <p>Teal: toward Real · Orange: toward Fake · Pale: near zero</p>
    <div className="attribution-overview" aria-label="Sequence overview">
      {Array.from({ length: bins }, (_, index) => {
        const from = Math.floor(index * data.sequence.length / bins);
        const to = Math.floor((index + 1) * data.sequence.length / bins);
        const mean = data.scores.slice(from, to).reduce((sum, value) => sum + value, 0) / (to - from);
        return <button key={index} style={{ backgroundColor: color(mean) }}
          title={`Residues ${from + 1}–${to}; mean contribution ${mean.toPrecision(3)}`}
          aria-label={`Show residues from ${from + 1}`} onClick={() => setStart(from)} />;
      })}
    </div>
    <div className="attribution-location" aria-label={`Current window: residues ${windowStart + 1}–${end}`}>
      <span className="attribution-window-marker" style={{ left: `${100 * windowStart / data.sequence.length}%`, width: `${100 * length / data.sequence.length}%` }} />
      <span className="attribution-window-arrow" style={{ left: `${100 * (windowStart + length / 2) / data.sequence.length}%` }} aria-hidden="true">▲</span>
    </div>
    <div className="attribution-controls">
      <button type="button" onClick={() => setView("fit")} aria-pressed={view === "fit"}>Fit one line</button>
      <button type="button" onClick={() => { setView("all"); setStart(0); }} aria-pressed={view === "all"}>Everything</button>
    </div>
    <label>Sequence window: {windowStart + 1}–{end} of {data.sequence.length}
      <input type="range" min={0} max={maxStart} value={windowStart} disabled={maxStart === 0}
        onChange={event => setStart(Number(event.target.value))} />
    </label>
    <div className={`attribution-residues${view === "all" ? " attribution-residues-all" : ""}`} ref={container}>
      {data.sequence.slice(windowStart, end).split("").map((residue, index) => {
        const position = windowStart + index;
        const score = data.scores[position];
        return <span key={position} tabIndex={0} style={{ backgroundColor: color(score) }}
          className={position % data.chunk_size === 0 ? "attribution-boundary" : ""}
          title={`Position ${position + 1}: ${residue}; contribution ${score.toPrecision(5)}${position % data.chunk_size === 0 ? "; chunk starts here" : ""}`}>
          {residue}<small>{position + 1}</small>
        </span>;
      })}
    </div>
  </section>;
}
