# DNA Trainer

## Running

Run these commands from the project root. Both launchers build the frontend and
serve it through the backend using `backend/settings.json`.

```bash
# Local: reuse or create a venv and install requirements.
./build_run.sh

# Docker: keep and reuse the container and image (default).
./docker_run.sh

# Docker: temporary container; Ctrl+C stops and removes it.
./docker_run.sh --temporary

# Rebuild the image and replace the saved container after dependency changes.
./docker_run.sh --rebuild

# Apply settings/mount changes without rebuilding the image.
./docker_run.sh --recreate

# Combine a temporary container with an image rebuild.
./docker_run.sh --temporary --rebuild
```

Docker builds the image only when it is missing or `--rebuild` is specified.
Temporary mode removes the container, but keeps the image and build cache for
next time. Reusable mode leaves the container running in the background and
prints commands to view logs and stop it. Bind-mounted files survive container
removal or replacement.

The Docker backend includes MMseqs2 on `PATH`.
Prepare training data once with the standalone
[MMseqs2 preprocessing tool](../MMseqs2Preprocessing/README.md), then select its
output dataset directory in PLM_MLPTrainer. Training loads the saved sequence
split, runs frozen ProtT5 embeddings, and trains a classification head. Neither training nor inference invokes MMseqs2.
Run `./docker_run.sh --rebuild` to add it to an existing installation. Standalone
preprocessing requires MMseqs2 installed on the host.

Wait for Docker startup to print `Ready`, then open
**http://localhost:25565** (or the `backend_port` in settings).

See [RUNNING.md](RUNNING.md) for prerequisites, GPU options, folder settings,
and connection troubleshooting.
