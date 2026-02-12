"""Entry point for running as module: python -m apx_clsct."""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
