import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { Home } from "./routes/Home";
import { Models } from "./routes/Models";
import { Structure } from "./routes/Structure";
import { Runs } from "./routes/Runs";
import { CheckpointDetail } from "./routes/Checkpoint";
import { NewRun } from "./routes/NewRun";
import { Datasets } from "./routes/Datasets";
import { Inference } from "./routes/Inference";
import { DataProcessing } from "./routes/DataProcessing";
import { Weights } from "./routes/Weights";
import { NetworkBuilder } from "./routes/NetworkBuilder";
import "./App.css";

const links = [
  ["/", "Overview"],
  ["/models", "Models"],
  ["/network-builder", "Builder"],
  ["/runs", "Training runs"],
  ["/new-run", "New run"],
  ["/weights", "Model weights"],
  ["/datasets", "Datasets"],
  ["/inference", "Inference"],
  ["/processing", "Data processing"],
] as const;
export default function App() {
  return (
    <BrowserRouter>
      <div className="shell">
        <header className="topbar">
          <NavLink to="/" className="brand">
            <span className="brand-mark">N</span>
            <span>
              NEURAL<span className="brand-light"> / OPS</span>
            </span>
          </NavLink>
          <nav>
            {links.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"}>
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="system-status">
            <i /> SYSTEM ONLINE
          </div>
        </header>
        <main className="content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/models" element={<Models />} />
            <Route path="/network-builder" element={<NetworkBuilder />} />
            <Route path="/models/:modelId/structure" element={<Structure />} />
            <Route path="/models/:modelId/runs" element={<Runs />} />
            <Route path="/runs" element={<Runs />} />
            <Route
              path="/runs/:runId/checkpoints/:checkpointId"
              element={<CheckpointDetail />}
            />
            <Route
              path="/checkpoints/:committedId"
              element={<CheckpointDetail />}
            />
            <Route path="/runs/:runId" element={<Runs />} />
            <Route path="/new-run" element={<NewRun />} />
            <Route path="/weights" element={<Weights />} />
            <Route path="/inference" element={<Inference />} />
            <Route path="/inference/new" element={<Inference />} />
            <Route path="/inference/runs/:runId" element={<Inference />} />
            <Route path="/processing" element={<DataProcessing />} />
            <Route path="/processing/runs/:runId" element={<DataProcessing />} />
            <Route path="/processing/runs/:runId/records/:recordId" element={<DataProcessing />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/datasets/:datasetId" element={<Datasets />} />
            <Route
              path="/datasets/:datasetId/entries/:entryId"
              element={<Datasets />}
            />
          </Routes>
        </main>
        <footer>
          NEURAL OPS <span>·</span> TRAINING CONTROL PLANE{" "}
          <span className="footer-right">LOCAL WORKSPACE</span>
        </footer>
      </div>
    </BrowserRouter>
  );
}
