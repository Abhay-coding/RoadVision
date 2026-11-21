"""Video processing pipeline for RoadVision dashcam semantic segmentation."""

import os
import subprocess
import time
import uuid
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Optional

import cv2
import imageio_ffmpeg
import numpy as np

from src.inference import RoadVisionSegmentor
from src.visualization import overlay_mask

def processed_video_path(input_path: str, output_dir: str) -> str:
    """Build an MP4 output path using the source video's filename stem and a UUID to prevent race conditions."""
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    unique_id = uuid.uuid4().hex[:8]
    return os.path.join(output_dir, f"processed_{input_stem}_{unique_id}.mp4")


class VideoProcessor:
    """Processes dashcam video streams with semantic segmentation overlays and HUD statistics."""

    def __init__(
        self,
        segmentor: RoadVisionSegmentor,
        overlay_alpha: float = 0.45,
        frame_skip: int = 1,
        target_resolution: Optional[tuple] = None # e.g. (640, 360) or None for native
    ):
        """Initialize VideoProcessor.

        Args:
            segmentor: Instance of RoadVisionSegmentor.
            overlay_alpha: Blending weight for segmentation mask.
            frame_skip: Process every k-th frame (1 = process all frames).
            target_resolution: (width, height) to resize frames for faster processing, or None.
        """
        self.segmentor = segmentor
        self.overlay_alpha = overlay_alpha
        self.frame_skip = max(1, frame_skip)
        self.target_resolution = target_resolution

    @staticmethod
    def _draw_text_with_outline(
        image: np.ndarray,
        text: str,
        position: tuple,
        font: int,
        font_scale: float,
        text_color: tuple,
        outline_color: tuple,
        outline_thickness: int,
        text_thickness: int
    ) -> None:
        """Draw text with an outline (background) for better visibility.

        Args:
            image: Image to draw on (modified in place).
            text: Text to draw.
            position: (x, y) position for text.
            font: OpenCV font type.
            font_scale: Font scale factor.
            text_color: Text color (BGR).
            outline_color: Outline color (BGR).
            outline_thickness: Outline thickness.
            text_thickness: Text thickness.
        """
        cv2.putText(image, text, position, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(image, text, position, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, Optional[int], float], None]] = None
    ) -> Dict:
        """Process video file, save annotated output video, and return telemetry statistics.

        Args:
            input_path: Path to source video file.
            output_path: Path to destination MP4 video file (H.264 MP4).
            max_frames: Optional maximum number of frames to process.
            progress_callback: Optional callback func(current_frame, total_frames, current_fps).

        Returns:
            Dictionary containing processing statistics:
                'total_frames_read', 'frames_inferred', 'frames_skipped',
                'model_latency_mean_ms', 'end_to_end_latency_mean_ms',
                'overall_fps', 'output_path', 'resolution'.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        if os.path.splitext(output_path)[1].lower() != ".mp4":
            raise ValueError("Output video path must end in .mp4")
        source = os.path.normcase(os.path.realpath(input_path))
        destination = os.path.normcase(os.path.realpath(output_path))
        if source == destination:
            raise ValueError("Input and output video paths must differ")

        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        with TemporaryDirectory(dir=output_dir, prefix=".roadvision-video-") as temp_dir:
            temp_avi = os.path.join(temp_dir, "frames.avi")
            temp_mp4 = os.path.join(temp_dir, "encoded.mp4")
            return self._process_video(
                input_path, output_path, temp_avi, temp_mp4, max_frames, progress_callback
            )

    def _process_video(self, input_path, output_path, temp_avi, temp_mp4, max_frames, progress_callback):
        cap = cv2.VideoCapture(input_path)
        writer = None
        global_start = time.perf_counter()
        frame_idx = 0
        frames_inferred = 0
        frames_skipped = 0
        model_latencies = []
        e2e_latencies = []

        last_colored_mask = None
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {input_path}")

            total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(native_fps) or native_fps <= 0:
                native_fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_w, out_h = self.target_resolution or (width, height)
            # MJPG and H.264 yuv420p require even frame dimensions.
            out_w -= out_w % 2
            out_h -= out_h % 2
            if out_w < 2 or out_h < 2:
                raise ValueError(f"Invalid video dimensions: {out_w}x{out_h}")

            writer = cv2.VideoWriter(
                temp_avi, cv2.VideoWriter_fourcc(*"MJPG"), native_fps, (out_w, out_h)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to initialize video writer for output: {output_path}")

            progress_total = total_input_frames or max_frames
            if max_frames and progress_total:
                progress_total = min(max_frames, progress_total)

            while cap.isOpened():
                if max_frames and frame_idx >= max_frames:
                    break

                e2e_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Resize if target resolution specified
                if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                # Convert BGR (OpenCV) to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Check frame skip
                is_inference_frame = (frame_idx - 1) % self.frame_skip == 0

                if is_inference_frame:
                    # Model inference
                    m_start = time.perf_counter()
                    pred_res = self.segmentor.predict(rgb_frame)
                    m_latency = (time.perf_counter() - m_start) * 1000.0
                    model_latencies.append(m_latency)

                    last_colored_mask = pred_res["colored_mask"]
                    frames_inferred += 1

                    # Update HUD text on inference frames
                    current_fps = 1000.0 / m_latency if m_latency > 0 else 0.0
                    avg_model_latency = np.mean(model_latencies[-10:])
                    last_hud_text = f"RoadVision | Frame: {frame_idx} | FPS: {current_fps:.1f} | Model: {avg_model_latency:.1f}ms"
                else:
                    frames_skipped += 1

                # Reuse the last prediction on the current frame to preserve motion while skipping inference.
                if last_colored_mask is not None:
                    blended = overlay_mask(rgb_frame, last_colored_mask, alpha=self.overlay_alpha)
                else:
                    blended = rgb_frame

                # Convert back to BGR for OpenCV writer
                bgr_out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

                # Compute latency
                e2e_latency = (time.perf_counter() - e2e_start) * 1000.0
                e2e_latencies.append(e2e_latency)

                self._draw_text_with_outline(
                    bgr_out, last_hud_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), (0, 0, 0), 3, 1
                )

                writer.write(bgr_out)

                if progress_callback:
                    progress_callback(frame_idx, progress_total, current_fps)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if not frame_idx:
            raise RuntimeError(f"Input video has no decodable frames: {input_path}")

        self._transcode_to_h264_mp4(temp_avi, temp_mp4)
        self._validate_video_file(temp_mp4)
        os.replace(temp_mp4, output_path)

        total_time = time.perf_counter() - global_start
        overall_fps = frame_idx / total_time if total_time > 0 else 0.0

        return {
            "total_frames_read": frame_idx,
            "frames_inferred": frames_inferred,
            "frames_skipped": frames_skipped,
            "model_latency_mean_ms": round(float(np.mean(model_latencies)), 2) if model_latencies else 0.0,
            "end_to_end_latency_mean_ms": round(float(np.mean(e2e_latencies)), 2) if e2e_latencies else 0.0,
            "overall_fps": round(float(overall_fps), 1),
            "output_path": output_path,
            "resolution": (out_w, out_h),
            "total_processing_time_s": round(float(total_time), 2)
        }

    @staticmethod
    def _transcode_to_h264_mp4(input_path: str, output_path: str) -> None:
        """Encode a browser-compatible H.264 MP4 using the bundled FFmpeg."""
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError(f"FFmpeg transcoding failed: {result.stderr.strip()}")

    @staticmethod
    def _validate_video_file(file_path: str) -> None:
        """Check that the encoded video has a decodable H.264 frame."""
        cap = cv2.VideoCapture(file_path)
        try:
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError(f"Output video has no decodable frames: {file_path}")

            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = fourcc.to_bytes(4, "little").lower()
            if codec not in (b"avc1", b"h264"):
                raise RuntimeError(f"Output video did not use H.264: {codec!r}")
        finally:
            cap.release()
