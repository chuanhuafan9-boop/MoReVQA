from __future__ import annotations

import json
from pathlib import Path

from engine.config import MoreVQAConfig
from engine.models.registry import build_model_bundle
from engine.schemas import VideoFrame
from framework.morevqa import MoReVQA


PROJECT_ROOT = Path(__file__).resolve().parent

# Edit these values in PyCharm, then click Run on this file.
CONFIG_PATH = PROJECT_ROOT / "configs" / "LLM_config.yaml"
VIDEO_PATH = PROJECT_ROOT / "examples" / "demo_video.mp4"
QUESTION = "What is happening in the video?"
OPTIONS = [
    "a colored square is moving across the screen",
    "the video is blank",
    "someone is reading text aloud",
]
TRACE_PATH = PROJECT_ROOT / "outputs" / "trace.json"

# Use False for actual reproduction inference. True validates the pipeline locally.
FORCE_MOCK = False

# The sample file is generated when no video is supplied, for an immediate PyCharm run.
CREATE_DEMO_VIDEO_IF_MISSING = True
DEMO_FPS = 6
DEMO_FRAME_COUNT = 24


def main() -> None:
    video_input = _prepare_video_input()
    config = MoreVQAConfig.from_file(CONFIG_PATH) if CONFIG_PATH.exists() else MoreVQAConfig.defaults()
    config = config.merged_with_defaults()
    models = build_model_bundle(config, force_mock=FORCE_MOCK)
    model = MoReVQA(config=config, models=models)
    output = model.inference(QUESTION, video_input, OPTIONS)

    if TRACE_PATH is not None:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        output.save_trace(TRACE_PATH)

    print(
        json.dumps(
            {
                "pipeline": "morevqa",
                "answer": output.prediction.answer,
                "trace": str(TRACE_PATH) if TRACE_PATH is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _prepare_video_input() -> Path | list[VideoFrame]:
    video_path = Path(VIDEO_PATH)
    if video_path.exists():
        return video_path
    if not CREATE_DEMO_VIDEO_IF_MISSING:
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    try:
        _create_demo_video(video_path)
    except RuntimeError:
        return _create_demo_frames()
    return video_path


def _create_demo_video(path: Path) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 180
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(DEMO_FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {path}")

    for frame_index in range(DEMO_FRAME_COUNT):
        canvas = np.full((height, width, 3), (244, 242, 236), dtype=np.uint8)
        x = 24 + int((width - 88) * frame_index / max(DEMO_FRAME_COUNT - 1, 1))
        y = 68
        cv2.rectangle(canvas, (x, y), (x + 52, y + 52), (70, 130, 220), thickness=-1)
        cv2.putText(
            canvas,
            "MoReVQA demo",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 52, 66),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"frame {frame_index + 1:02d}",
            (18, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (88, 96, 110),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()


def _create_demo_frames() -> list[VideoFrame]:
    from PIL import Image, ImageDraw

    frames: list[VideoFrame] = []
    width, height = 320, 180
    for frame_index in range(DEMO_FRAME_COUNT):
        image = Image.new("RGB", (width, height), (236, 242, 244))
        draw = ImageDraw.Draw(image)
        x = 24 + int((width - 88) * frame_index / max(DEMO_FRAME_COUNT - 1, 1))
        y = 68
        draw.rectangle((x, y, x + 52, y + 52), fill=(220, 130, 70))
        draw.text((18, 18), "MoReVQA demo", fill=(40, 52, 66))
        draw.text((18, 150), f"frame {frame_index + 1:02d}", fill=(88, 96, 110))
        frames.append(
            VideoFrame(
                frame_id=frame_index,
                timestamp=frame_index / float(DEMO_FPS),
                image=image,
                source_path="generated demo frames",
            )
        )
    return frames


if __name__ == "__main__":
    main()
