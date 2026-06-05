"""Compatibility wrapper for tree SPP experiments."""

import sys

from exp import SOLVERS, main, make_solver


if __name__ == "__main__":
    main(["--instance", "tree", *sys.argv[1:]])
