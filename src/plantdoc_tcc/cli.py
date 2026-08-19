from __future__ import annotations

import argparse
import json

from .config import iter_specs, load_config
from .data import audit_dataset, class_weights


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="plantdoc-tcc")
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--data", required=True)
    audit.add_argument("--split", default="train")
    audit.add_argument("--expected-classes", type=int)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--strategy", action="append")
    train.add_argument("--epoch", action="append", type=int)
    train.add_argument("--device")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--output", default="artifacts/metrics.json")
    staged = commands.add_parser("staged")
    staged.add_argument("--data", required=True)
    staged.add_argument("--config", default="configs/colab_protocol.json")
    staged.add_argument("--project", required=True)
    staged.add_argument(
        "--phase",
        choices=["loss", "search", "promote200", "confirm300", "all"],
        required=True,
    )
    staged.add_argument("--device", default="0")
    final_test = commands.add_parser("final-test")
    final_test.add_argument("--data", required=True)
    final_test.add_argument("--project", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "audit":
        result = audit_dataset(args.data, args.split, args.expected_classes)
        weights = class_weights(result.instance_counts) if all(result.instance_counts) else None
        print(json.dumps({**result.__dict__, "suggested_weights": weights}, ensure_ascii=False, indent=2))
    elif args.command == "plan":
        specs = list(iter_specs(load_config(args.config)))
        print(f"{len(specs)} execuções no grid completo")
        for spec in specs:
            print(spec.name)
    elif args.command == "train":
        from .runner import train_grid

        train_grid(args.data, load_config(args.config), args.strategy, args.device, args.epoch)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        print(json.dumps(evaluate(args.weights, args.data, args.split, args.output), ensure_ascii=False, indent=2))
    elif args.command == "staged":
        from .staged import run_phase

        result = run_phase(args.data, args.config, args.project, args.phase, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "final-test":
        from .staged import finalize_test

        print(json.dumps(finalize_test(args.data, args.project), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
