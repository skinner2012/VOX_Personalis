"""Entry point for python -m scripts.data_inventory"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
