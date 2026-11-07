"""py -3 beatrig.py <command> ...  (same as `python -m beatrig`)"""
import sys

from beatrig.cli import main

if __name__ == "__main__":
    sys.exit(main())
