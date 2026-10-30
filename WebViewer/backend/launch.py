"""Build the configured frontend, then start the backend with the same interpreter."""
import os
import subprocess
import sys

from backend.config import PROJECT_ROOT, load_settings


def main():
    settings = load_settings()
    if settings['frontend_static_dir'] is not None:
        env = {**os.environ, 'DNATRAINER_SETTINGS': str(PROJECT_ROOT / 'backend/settings.json')}
        source = settings['frontend_source_dir']
        subprocess.run(['npm', 'ci' if (source / 'package-lock.json').is_file() else 'install',
                        '--prefix', str(source)], cwd=PROJECT_ROOT, env=env, check=True)
        subprocess.run(['npm', 'run', 'build', '--prefix', str(settings['frontend_source_dir'])],
                       cwd=PROJECT_ROOT, env=env, check=True)
    os.chdir(PROJECT_ROOT)
    os.execv(sys.executable, [sys.executable, '-m', 'backend.main'])


if __name__ == '__main__':
    main()
