"""Compatibility wrapper for star SPP experiments."""

import sys

from exp import SOLVERS, main, make_solver


if __name__ == "__main__":
    main(["--instance", "star", *sys.argv[1:]])
