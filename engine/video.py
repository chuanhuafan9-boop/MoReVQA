from __future__ import annotations

from pathlib import Path
from typing import Iterable

from engine.schemas import VideoFrame


class VideoLoadError(RuntimeError):
    """Raised when videos frames cannot be decoded."""


def load_video_frames(
    video_path: str | Path,
    sample_fps: float = 1.0,
    max_frames: int | None = None,
) -> list[VideoFrame]:
    """Load sampled frames from a videos file.

    Frame ids are counted in sampled-frame order, matching the paper examples
    where frame references are the sampled temporal sequence.
    """

    path = Path(video_path)
    if not path.exists():
        raise VideoLoadError(f"Video file does not exist: {path}")
    try:
        return _load_with_cv2(path, sample_fps=sample_fps, max_frames=max_frames)
    except ImportError:
        return _load_with_imageio(path, sample_fps=sample_fps, max_frames=max_frames)


def uniform_sample_frames(frames: list[VideoFrame], count: int | None) -> list[VideoFrame]:
    if not frames:
        return []
    if count is None or count <= 0 or count >= len(frames):
        return list(frames)
    if count == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / float(count - 1)
    indices = [round(i * step) for i in range(count)]
    return [frames[index] for index in indices]


def select_frames(frames: Iterable[VideoFrame], frame_ids: Iterable[int]) -> list[VideoFrame]:
    wanted = {int(frame_id) for frame_id in frame_ids}
    return [frame for frame in frames if frame.frame_id in wanted]


def frame_lookup(frames: Iterable[VideoFrame]) -> dict[int, VideoFrame]:
    return {frame.frame_id: frame for frame in frames}


def _load_with_cv2(
    path: Path,
    sample_fps: float,
    max_frames: int | None,
) -> list[VideoFrame]:
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise ImportError from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoLoadError(f"Could not open videos: {path}")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps or 1.0
    interval = max(int(round(native_fps / max(sample_fps, 1e-6))), 1)
    frames: list[VideoFrame] = []
    raw_index = 0
    sampled_index = 0
    while True:
        ok, image_bgr = capture.read()
        if not ok:
            break
        if raw_index % interval == 0:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            timestamp = raw_index / native_fps
            frames.append(
                VideoFrame(
                    frame_id=sampled_index,
                    timestamp=timestamp,
                    image=Image.fromarray(image_rgb),
                    source_path=str(path),
                )
            )
            sampled_index += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
        raw_index += 1
    capture.release()
    if not frames:
        raise VideoLoadError(f"No frames decoded from videos: {path}")
    return frames


def _load_with_imageio(
    path: Path,
    sample_fps: float,
    max_frames: int | None,
) -> list[VideoFrame]:
    try:
        import imageio.v3 as iio
        from PIL import Image
    except ImportError as exc:
        raise VideoLoadError("Install opencv-python or imageio to read videos.") from exc

    meta = iio.immeta(path)
    native_fps = float(meta.get("fps") or sample_fps or 1.0)
    interval = max(int(round(native_fps / max(sample_fps, 1e-6))), 1)
    frames: list[VideoFrame] = []
    for raw_index, array in enumerate(iio.imiter(path)):
        if raw_index % interval != 0:
            continue
        frames.append(
            VideoFrame(
                frame_id=len(frames),
                timestamp=raw_index / native_fps,
                image=Image.fromarray(array),
                source_path=str(path),
            )
        )
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        raise VideoLoadError(f"No frames decoded from videos: {path}")
    return frames
