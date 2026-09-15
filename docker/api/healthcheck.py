"""Healthcheck interno P19 del servicio api: solo GET /healthz en loopback.

Sin dependencias externas (stdlib unicamente); no imprime ni registra
nada mas alla del codigo de salida del proceso Docker HEALTHCHECK.
"""

from __future__ import annotations

import sys
import urllib.request


def main() -> int:
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/healthz", timeout=2
        ) as response:
            return 0 if response.status == 200 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
