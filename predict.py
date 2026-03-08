from pathlib import Path
import subprocess
import sys

project_root = Path(__file__).resolve().parent
command = [sys.executable, str(project_root / 'main.py'), 'predict']

raise SystemExit(subprocess.call(command, cwd=project_root))
