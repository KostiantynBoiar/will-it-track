"""Put the repo root on ``sys.path`` so ``import src`` resolves (pytest + scripts).

Nothing is pip-installed. Tests live under ``tests/`` and import ``from src...``; scripts run as
``PYTHONPATH=. python -m src.<subpackage>.<module>``.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
