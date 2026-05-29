from __future__ import annotations

import argparse
import json
from pathlib import Path

from Datasets.loaders import (
    NextQADataset,
    build_nextqa_dataloader,
    normalize_nextqa_prediction,
)
from engine.config import MoreVQAConfig
from engine.models.registry import build_model_bundle
from framework.morevqa import MoReVQA


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoReVQA-style video question answering.")
    parser.add_argument(
        "--dataset",
        choices=["single", "nextqa"],
        default="single",
        help="Run one manually supplied video question or a dataset split.",
    )
    parser.add_argument("--config", default="configs/LLM_config.yaml", help="Path to YAML config.")
    parser.add_argument("--video", "--videos", dest="video", help="Path to a video file.")
    parser.add_argument("--question", help="Video question.")
    parser.add_argument("--options", nargs="*", default=None, help="Candidate answers for MCQA.")
    parser.add_argument("--trace", help="Optional path to save JSON execution trace.")
    parser.add_argument("--data-root", default="data/nextqa", help="Dataset root for --dataset nextqa.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of dataset requests to run.")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset row index to start from.")
    parser.add_argument("--interval", type=int, default=1, help="Read every Nth dataset row.")
    parser.add_argument(
        "--keep-missing-videos",
        action="store_true",
        help="Raise an error instead of skipping rows whose video file is missing.",
    )
    parser.add_argument(
        "--output",
        default="outputs/nextqa_predictions.jsonl",
        help="JSONL output path for --dataset nextqa.",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Directory for per-sample traces when running --dataset nextqa.",
    )
    parser.add_argument("--mock", action="store_true", help="Force deterministic mock models.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = MoreVQAConfig.from_file(config_path) if config_path.exists() else MoreVQAConfig.defaults()
    config = config.merged_with_defaults()
    models = build_model_bundle(config, force_mock=args.mock)
    model = MoReVQA(config=config, models=models)

    if args.dataset == "nextqa":
        _run_nextqa(model, args)
        return

    if not args.video or not args.question:
        parser.error("--video and --question are required when --dataset single")

    output = model.inference(args.question, args.video, args.options)
    if args.trace:
        output.save_trace(args.trace)

    print(json.dumps({"answer": output.prediction.answer}, ensure_ascii=False, indent=2))


def _run_nextqa(model: MoReVQA, args: argparse.Namespace) -> None:
    data_num = args.num_samples
    print("[NExT-QA] 初始化数据集读取器", flush=True)
    dataset = NextQADataset(
        data_root=args.data_root,
        split=args.split,
        data_num=data_num,
        start_index=args.start_index,
        interval=args.interval,
        skip_missing_videos=not args.keep_missing_videos,
    )

    # Same structure as CLOVA task demos: Dataset -> DataLoader -> task framework.
    # Each yielded sample is one complete MoReVQA request.
    dataloader = build_nextqa_dataloader(dataset, batch_size=1, shuffle=False)
    print(
        "[NExT-QA] 数据集准备完成: "
        f"split={args.split}, num_samples={args.num_samples}, valid_samples={len(dataset)}, "
        f"start_index={args.start_index}, interval={args.interval}",
        flush=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for batch in dataloader:
            sample = batch[0]
            total += 1

            # Print the question and candidates before entering M1/M2/M3.
            _print_nextqa_request(sample, total, len(dataset))

            output = model.inference(
                question=sample["question"],
                video=sample["video_path"],
                options=sample["options"],
            )
            prediction = output.prediction.answer
            predicted_index = normalize_nextqa_prediction(prediction, sample["options"])
            is_correct = predicted_index == sample["answer_index"]
            if is_correct:
                correct += 1

            # One JSON trace per sample helps inspect the paper-style modules offline.
            trace_path = None
            if trace_dir is not None:
                trace_path = trace_dir / f"{sample['sample_id']}.json"
                output.save_trace(trace_path)

            record = {
                **sample,
                "prediction": prediction,
                "predicted_index": predicted_index,
                "correct": is_correct,
                "trace_path": str(trace_path) if trace_path is not None else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                "[NExT-QA] 当前请求完成: "
                f"prediction={prediction}, gold={sample['answer']}, correct={is_correct}",
                flush=True,
            )

    accuracy = correct / total if total else 0.0
    print(
        json.dumps(
            {
                "dataset": "nextqa",
                "split": args.split,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "output_path": str(output_path),
                "trace_dir": str(trace_dir) if trace_dir is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _print_nextqa_request(sample: dict, index: int, total: int) -> None:
    print("-" * 80, flush=True)
    print(f"[NExT-QA] 请求 {index}/{total}", flush=True)
    print(f"[NExT-QA] sample_id: {sample['sample_id']}", flush=True)
    print(f"[NExT-QA] video_path: {sample['video_path']}", flush=True)
    print(f"[NExT-QA] question: {sample['question']}", flush=True)
    print("[NExT-QA] options:", flush=True)
    for option_index, option in enumerate(sample["options"]):
        print(f"  {option_index}. {option}", flush=True)
    print(
        f"[NExT-QA] gold answer: {sample['answer_index']} -> {sample['answer']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
