from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.config import MoreVQAConfig
from engine.models.registry import build_model_bundle
from framework.morevqa import MoReVQA


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoReVQA-style video question answering.")
    parser.add_argument("--config", default="configs/LLM_config.yaml", help="Path to YAML config.")
    parser.add_argument("--video", "--videos", dest="video", required=True, help="Path to a video file.")
    parser.add_argument("--question", required=True, help="Video question.")
    parser.add_argument("--options", nargs="*", default=None, help="Candidate answers for MCQA.")
    parser.add_argument("--trace", help="Optional path to save JSON execution trace.")
    parser.add_argument("--mock", action="store_true", help="Force deterministic mock models.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = MoreVQAConfig.from_file(config_path) if config_path.exists() else MoreVQAConfig.defaults()
    config = config.merged_with_defaults()
    models = build_model_bundle(config, force_mock=args.mock)
    model = MoReVQA(config=config, models=models)
    output = model.inference(args.question, args.video, args.options)

    if args.trace:
        output.save_trace(args.trace)

    print(json.dumps({"answer": output.prediction.answer}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
