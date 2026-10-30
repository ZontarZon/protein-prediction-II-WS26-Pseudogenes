import { Link } from "react-router-dom";
import { useResource } from "./api";
import { Page, State } from "./shared";
type Summary = { models: { name: string }[] };
type Datasets = { datasets: { itemCount: number }[] };
type Runs = { checkpoints: { id: string }[] };
export function Home() {
  const m = useResource<Summary>("/models", "models"),
    d = useResource<Datasets>("/datasets", "datasets"),
    c = useResource<Runs>("/checkpoints", "checkpoints");
  const stats: [string, string | number, string][] = [
    ["MODEL FILES", m.data?.models.length ?? "—", "/models"],
    ["CHECKPOINTS", c.data?.checkpoints.length ?? "—", "/models"],
    ["DATASETS", d.data?.datasets.length ?? "—", "/datasets"],
  ];
  const modules: [string, string, string, string][] = [
    ["Models", "Inspect architectures and checkpoints", "/models", "01"],
    ["Training runs", "Review checkpoint metrics and results", "/runs", "02"],
    ["Datasets", "Browse available training data", "/datasets", "03"],
    [
      "Inference",
      "Start runs and inspect inference history",
      "/inference",
      "04",
    ],
  ];
  return (
    <Page
      eyebrow="CONTROL CENTER / 01"
      title="Training overview"
      description="Your AI training workspace, at a glance."
      action={
        <Link className="button primary" to="/new-run">
          ＋ Configure training run
        </Link>
      }
    >
      <div className="hero">
        <div>
          <div className="live-label">
            <i /> WORKSPACE READY
          </div>
          <h2>
            Build, inspect, <span>iterate.</span>
          </h2>
          <p>
            Manage model assets and prepare training configurations from one
            control plane.
          </p>
          <Link to="/models" className="text-link">
            Explore models <b>↗</b>
          </Link>
        </div>
        <div className="hero-orbit">
          <div className="orbit o1" />
          <div className="orbit o2" />
          <div className="orbit-core">AI</div>
          <span className="orbit-tag t1">MODEL</span>
          <span className="orbit-tag t2">DATA</span>
          <span className="orbit-tag t3">TRAIN</span>
        </div>
      </div>
      <div className="stat-grid">
        {stats.map(([label, value, to]) => (
          <Link className="stat-card" to={to} key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
            <small>
              DISCOVERED IN WORKSPACE <b>↗</b>
            </small>
          </Link>
        ))}
      </div>
      <div className="home-bottom">
        <div>
          <div className="eyebrow">QUICK ACCESS</div>
          <h2>Workspace modules</h2>
        </div>
        <div className="module-grid">
          {modules.map(([name, copy, path, num]) => (
            <Link className="module-card" to={path} key={name}>
              <span className="mono module-number">/{num}</span>
              <h3>{name}</h3>
              <p>{copy}</p>
              <b>↗</b>
            </Link>
          ))}
        </div>
      </div>
      <State
        loading={m.loading || d.loading || c.loading}
        error={m.error || d.error || c.error}
      />
    </Page>
  );
}
