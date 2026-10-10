"""Compatibility wrapper for tree SPP experiments."""

import sys

from mecbench.cli import SOLVERS, main, make_solver


if __name__ == "__main__":
    raise SystemExit(main(["--instance", "tree", *sys.argv[1:]]))
