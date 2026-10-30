# Running DNA Trainer

Run either script from any working directory:

```bash
./build_run.sh
# Reuse a saved container and image (default):
./docker_run.sh
# Remove the container when you finish (still reuses the image):
./docker_run.sh --temporary
# Rebuild dependencies and replace the saved container:
./docker_run.sh --rebuild
# Apply settings or mount changes without rebuilding the image:
./docker_run.sh --recreate
# Rebuild the image and run a temporary container:
./docker_run.sh --temporary --rebuild
# For NVIDIA GPU access with the NVIDIA Container Toolkit installed:
./docker_run.sh --gpus all
```

Both scripts use `backend/settings.json` without a folder preflight check.
The frontend is built into `frontend_static_dir` on startup; the build can create
that folder. Set it to an empty string to disable frontend building and hosting.

`frontend_source_dir` identifies the frontend source folder; `frontend_static_dir`
identifies the build output served by Flask. Change either setting to relocate
that folder. Relative paths resolve from the project root; absolute paths and
`~/` paths also work. The Vite configuration and backend already share these
settings.

Local startup needs Python 3 with venv support and Node.js/npm (Node 22 is used
in Docker). It reuses `backend/.venv`, then `.venv`, or creates `backend/.venv`.
It installs `backend/requirements.txt` into that venv before launching. npm
installs frontend dependencies with `npm ci` when a lockfile exists, otherwise
`npm install`, then builds the frontend and starts Flask.

Docker startup needs host Python 3 (standard library only) and a running Docker
daemon. It builds `dnatrainer:local` only if the image is missing or `--rebuild`
is supplied. Python requirements are installed inside the image; no host venv
is used. Frontend dependencies and the frontend are built when the container
process starts.

By default (`--keep`), the script creates a named container, waits for the
backend to respond, then exits with the container running in the background.
Running the script again reuses the running container or starts it if stopped.
It prints commands to view logs and stop it. Use `--name NAME` for a custom
container name. Use `--recreate` after changing settings, mounts, Docker options,
or when you want to restart with changed frontend code. Use `--rebuild` after
changing Python requirements or the Dockerfile; this also replaces the saved
container. Files in bind mounts survive replacement.

With `--temporary`, the script follows logs after readiness; Ctrl+C stops and
removes the temporary container. The image and build cache are still retained.
`--temporary --rebuild` combines an image rebuild with a disposable container;
it leaves any saved reusable container intact. Stop an existing container using
the same backend port before launching a temporary one.

The project and existing configured external folders are bind mounted at their
original absolute paths. Missing external folders are skipped; create and mount
them before launching if you need their contents to persist. Frontend
dependencies, build output, datasets, and model outputs in mounted folders
persist on the host. The process runs as your host UID/GID to avoid root-owned
files. Use a local Linux Docker daemon with access to these paths. Docker starts
the backend on `0.0.0.0` inside the container and publishes `backend_port` on the
host; open `http://localhost:25565` with the default settings. Additional script
arguments are passed as Docker run options (for example `--gpus all`).

Wait for the script to print `Ready` before opening the site. Use **HTTP**:
`http://localhost:25565`, or your configured `backend_port`. Port 5173 is for
Vite development and is not used by this Docker launcher. When accessing from
another computer, replace `localhost` with the Docker host's address.
The launcher tests `/api/health` on the published port and shows recent logs if
the backend exits or fails to respond. The first frontend installation/build
can take time; increase `--wait-timeout 600` if needed. A reusable container
stays available for inspection after a failed startup. If Docker reports socket
permission errors, run from an account with access to the Docker daemon.

The requirements include PyTorch and CUDA packages, so the first installation
can be large. GPU use additionally requires compatible host drivers and NVIDIA
Container Toolkit; Docker does not install host drivers.
