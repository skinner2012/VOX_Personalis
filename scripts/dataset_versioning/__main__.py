"""Entry point for python -m scripts.dataset_versioning"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
