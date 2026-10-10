"""Command-line entry point for MEC placement benchmarks."""
from pathlib import Path
import argparse
from mecbench.benchmarks.runner import INSTANCES, SOLVERS, make_solver, run_benchmarks

def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number
def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run SPP benchmark experiments.")
    parser.add_argument("--instance", choices=INSTANCES, default="tree")
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--groups", type=positive_int, default=4)
    parser.add_argument("--neighbors", type=nonnegative_int, default=1)
    parser.add_argument("--start-users", type=positive_int)
    parser.add_argument("--stop-users", type=positive_int)
    parser.add_argument("--step-users", type=positive_int)
    parser.add_argument("--ilp-time-limit", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)



def set_instance_defaults(args):
    if args.start_users is None:
        args.start_users = 20 if args.instance == "tree" else 15
    if args.stop_users is None:
        args.stop_users = 410 if args.instance == "tree" else 125
    if args.step_users is None:
        args.step_users = 10 if args.instance == "tree" else 5
    if args.output is None:
        args.output = Path("artifacts") / "data" / f"exp_{args.instance}.csv"
    if args.figures_dir is None:
        args.figures_dir = Path("artifacts") / "figures" / args.instance
    return args



def main(argv=None):
    args = set_instance_defaults(parse_args(argv))
    return run_benchmarks(args)


if __name__ == "__main__":
    raise SystemExit(main())
