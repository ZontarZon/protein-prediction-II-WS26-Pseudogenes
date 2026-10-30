"""Dependency-free settings loader and Docker lifecycle management."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from backend.config import PROJECT_ROOT, load_settings


def docker(*args, **kwargs):
    return subprocess.run(['docker', *args], check=True, **kwargs)


def inspect(kind, name):
    result = subprocess.run(['docker', kind, 'inspect', name],
                            capture_output=True, text=True)
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def wait_for_backend(name, port, timeout):
    url = f'http://127.0.0.1:{port}/api/health'
    deadline = time.monotonic() + timeout
    # Local readiness requests must not go through a configured HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        state = inspect('container', name)
        if not state or not state['State']['Running']:
            raise RuntimeError(f'Container {name} stopped before the backend became ready.')
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200 and json.load(response).get('status') == 'ok':
                    return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise RuntimeError(f'Backend did not respond at {url} within {timeout}s. '
                       'Frontend installation/build may need more time; '
                       'increase --wait-timeout or inspect the logs.')


def run_container(settings):
    parser = argparse.ArgumentParser(description='Run DNA Trainer in Docker; reuse by default.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--keep', action='store_true', help='keep and reuse the container (default)')
    mode.add_argument('--temporary', action='store_true', help='remove the container on exit')
    parser.add_argument('--rebuild', action='store_true', help='rebuild the image and replace the saved container')
    parser.add_argument('--recreate', action='store_true', help='replace the saved container without rebuilding the image')
    default_name = 'dnatrainer-' + hashlib.sha256(str(PROJECT_ROOT).encode()).hexdigest()[:10]
    parser.add_argument('--name', default=default_name, help='name of the reusable container')
    parser.add_argument('--wait-timeout', type=int, default=300, help='seconds to wait for HTTP readiness (default: 300)')
    args, extra = parser.parse_known_args(sys.argv[2:])
    if args.wait_timeout < 1:
        parser.error('--wait-timeout must be positive')
    if not shutil.which('docker'):
        raise RuntimeError('Docker is required. Install Docker and start its daemon first.')
    docker('info', stdout=subprocess.DEVNULL)

    # Preserve absolute paths, including external storage and resolved symlinks.
    folders = {PROJECT_ROOT}
    for key, value in settings.items():
        if key.endswith('_dir') and value is not None:
            folders.add(value)
    folders.update(settings['dataset_roots'])
    mounts = []
    for folder in sorted(folders, key=lambda path: (len(path.parts), str(path))):
        if not folder.is_dir():
            continue
        if not any(parent == folder or parent in folder.parents for parent in mounts):
            mounts.append(folder)

    image = 'dnatrainer:local'
    if args.rebuild or inspect('image', image) is None:
        docker('build', '-t', image, '-f', str(PROJECT_ROOT / 'Dockerfile'), str(PROJECT_ROOT))
    else:
        print('Reusing existing image (use --rebuild after changing Python requirements or Dockerfile).', flush=True)

    port = str(settings['backend_port'])
    name = f'{default_name}-temp-{uuid.uuid4().hex[:8]}' if args.temporary else args.name
    signature = hashlib.sha256(json.dumps({
        'settings': settings, 'mounts': mounts, 'options': extra,
        'uid': os.getuid(), 'gid': os.getgid(),
        'image': inspect('image', image)['Id'],
    }, default=str, sort_keys=True).encode()).hexdigest()
    existing = inspect('container', name)
    if existing:
        labels = existing['Config'].get('Labels') or {}
        if labels.get('dnatrainer.project') != str(PROJECT_ROOT):
            raise RuntimeError(f'Container name {name} is already in use by another project.')
        if args.rebuild or args.recreate:
            docker('rm', '-f', name)
            existing = None
        elif labels.get('dnatrainer.configuration') != signature:
            raise RuntimeError('Saved container settings, mounts, image, or Docker options changed. '
                               'Run again with --recreate to apply them.')

    command = ['run', '--detach', '--init', '--name', name,
               '--user', f'{os.getuid()}:{os.getgid()}',
               '--workdir', str(PROJECT_ROOT),
               '--env', f'HOME={Path.home()}',
               '--env', 'DNATRAINER_BACKEND_HOST=0.0.0.0',
               '--label', f'dnatrainer.project={PROJECT_ROOT}',
               '--label', f'dnatrainer.configuration={signature}',
               '--publish', f'{port}:{port}']
    if args.temporary:
        command.append('--rm')
    for folder in mounts:
        if ',' in str(folder):
            raise RuntimeError(f'Docker bind paths cannot contain commas: {folder}')
        command.extend(['--mount', f'type=bind,source={folder},target={folder}'])
    command.extend(extra)
    command.append(image)
    started = False
    try:
        if existing:
            if not existing['State']['Running']:
                docker('start', name)
        else:
            docker(*command)
        started = True
        print(f'Waiting for backend in {name} (up to {args.wait_timeout}s)...', flush=True)
        wait_for_backend(name, port, args.wait_timeout)
        print(f'Ready: http://localhost:{port} (HTTP, not HTTPS)', flush=True)
        if args.temporary:
            print('Temporary mode: Ctrl+C stops and removes this container.', flush=True)
            subprocess.run(['docker', 'logs', '--follow', name], check=False)
        else:
            print(f'Container stays running. Logs: docker logs -f {name}\n'
                  f'Stop: docker stop {name}\n'
                  'Run this script again to reuse it.', flush=True)
    except RuntimeError:
        subprocess.run(['docker', 'logs', '--tail', '100', name], check=False)
        raise
    finally:
        if args.temporary and started:
            subprocess.run(['docker', 'stop', name], stdout=subprocess.DEVNULL, check=False)


def main():
    try:
        settings = load_settings()
        mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
        if mode == 'check':
            return
        if mode != 'docker':
            raise RuntimeError(f'Unknown startup mode: {mode}')
        run_container(settings)
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Startup failed: {error}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
