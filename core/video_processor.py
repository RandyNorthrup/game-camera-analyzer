"""
Video processing module for extracting and analyzing frames from game camera videos.

This module provides functionality for:
- Frame extraction from video files
- Motion detection using background subtraction
- Timestamp extraction from video metadata
- Integration with existing detection pipeline
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from utils.validators import validate_file_exists

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """
    Video file metadata and properties.

    Attributes:
        path: Path to video file
        fps: Frames per second
        frame_count: Total number of frames
        width: Frame width in pixels
        height: Frame height in pixels
        duration_seconds: Total video duration
        codec: Video codec name
    """

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    codec: str

    def __post_init__(self) -> None:
        """Validate video info after initialization."""
        if self.fps <= 0:
            raise ValueError(f"FPS must be > 0, got {self.fps}")
        if self.frame_count < 0:
            raise ValueError(f"Frame count must be >= 0, got {self.frame_count}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")


@dataclass
class FrameExtraction:
    """
    Extracted frame with metadata.

    Attributes:
        frame_number: Frame index in video
        timestamp: Time in seconds from video start
        image: Frame image array (RGB format)
        has_motion: Whether motion was detected in this frame
        motion_score: Motion intensity score (0.0-1.0)
    """

    frame_number: int
    timestamp: float
    image: NDArray[np.uint8]
    has_motion: bool = False
    motion_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate frame extraction after initialization."""
        if self.frame_number < 0:
            raise ValueError(f"Frame number must be >= 0, got {self.frame_number}")
        if self.timestamp < 0:
            raise ValueError(f"Timestamp must be >= 0, got {self.timestamp}")
        if not isinstance(self.image, np.ndarray):
            raise TypeError(f"Image must be ndarray, got {type(self.image)}")
        if not 0.0 <= self.motion_score <= 1.0:
            raise ValueError(f"Motion score must be 0-1, got {self.motion_score}")


@dataclass
class MotionDetectionConfig:
    """
    Configuration for motion detection.

    Attributes:
        enabled: Whether motion detection is enabled
        threshold: Motion detection sensitivity (0-255)
        min_area: Minimum contour area for motion (pixels)
        learning_rate: Background subtraction learning rate (0.0-1.0)
        history: Number of frames for background model
    """

    enabled: bool = True
    threshold: int = 25
    min_area: int = 500
    learning_rate: float = 0.01
    history: int = 500

    def __post_init__(self) -> None:
        """Validate motion detection config."""
        if not 0 <= self.threshold <= 255:
            raise ValueError(f"Threshold must be 0-255, got {self.threshold}")
        if self.min_area < 0:
            raise ValueError(f"Min area must be >= 0, got {self.min_area}")
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError(f"Learning rate must be 0-1, got {self.learning_rate}")
        if self.history < 1:
            raise ValueError(f"History must be >= 1, got {self.history}")


class VideoProcessor:
    """
    Process video files for game camera analysis.

    Handles frame extraction, motion detection, and integration with
    the existing detection pipeline.
    """

    def __init__(
        self,
        motion_config: Optional[MotionDetectionConfig] = None,
    ) -> None:
        """
        Initialize video processor.

        Args:
            motion_config: Motion detection configuration
        """
        self.motion_config = motion_config or MotionDetectionConfig()
        self.bg_subtractor: Optional[cv2.BackgroundSubtractor] = None

        if self.motion_config.enabled:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.motion_config.history,
                varThreshold=self.motion_config.threshold,
                detectShadows=False,
            )

        logger.info(
            f"VideoProcessor initialized: motion_detection={self.motion_config.enabled}"
        )

    def get_video_info(self, video_path: Path) -> VideoInfo:
        """
        Get video file metadata and properties.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo object with metadata

        Raises:
            ValueError: If video file is invalid or cannot be opened
            FileNotFoundError: If video file does not exist
        """
        validate_file_exists(video_path, "video")

        cap = cv2.VideoCapture(str(video_path))

        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Convert fourcc to string
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            duration = frame_count / fps if fps > 0 else 0.0

            info = VideoInfo(
                path=video_path,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                duration_seconds=duration,
                codec=codec,
            )

            logger.info(
                f"Video info: {video_path.name} - "
                f"{frame_count} frames @ {fps:.2f} fps, "
                f"{width}x{height}, {duration:.2f}s"
            )

            return info

        except Exception as e:
            logger.error(f"Failed to get video info: {e}", exc_info=True)
            raise ValueError(f"Invalid video file {video_path}: {e}")
        finally:
            cap.release()

    def extract_frames(
        self,
        video_path: Path,
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
    ) -> List[FrameExtraction]:
        """
        Extract frames from video with optional motion detection.

        Args:
            video_path: Path to video file
            frame_interval: Extract every Nth frame (1 = all frames)
            max_frames: Maximum number of frames to extract (None = all)

        Returns:
            List of extracted frames with metadata

        Raises:
            ValueError: If video file is invalid
            FileNotFoundError: If video file does not exist
        """
        validate_file_exists(video_path, "video")

        if frame_interval < 1:
            raise ValueError(f"Frame interval must be >= 1, got {frame_interval}")

        cap = cv2.VideoCapture(str(video_path))
        frames: List[FrameExtraction] = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = 0
            extracted_count = 0

            logger.info(
                f"Extracting frames from {video_path.name}: "
                f"interval={frame_interval}, max_frames={max_frames}"
            )

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Extract every Nth frame
                if frame_number % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Calculate timestamp
                    timestamp = frame_number / fps if fps > 0 else 0.0

                    # Detect motion if enabled
                    has_motion = False
                    motion_score = 0.0

                    if self.motion_config.enabled and self.bg_subtractor is not None:
                        has_motion, motion_score = self._detect_motion(frame)

                    extraction = FrameExtraction(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        image=frame_rgb,
                        has_motion=has_motion,
                        motion_score=motion_score,
                    )

                    frames.append(extraction)
                    extracted_count += 1

                    if max_frames is not None and extracted_count >= max_frames:
                        logger.info(f"Reached max_frames limit: {max_frames}")
                        break

                frame_number += 1

            logger.info(
                f"Extracted {len(frames)} frames from {video_path.name} "
                f"(processed {frame_number} total frames)"
            )

            return frames

        except Exception as e:
            logger.error(f"Failed to extract frames: {e}", exc_info=True)
            raise RuntimeError(f"Frame extraction failed: {e}")
        finally:
            cap.release()

    def extract_motion_frames(
        self,
        video_path: Path,
        frame_interval: int = 1,
        motion_threshold: float = 0.1,
        max_frames: Optional[int] = None,
    ) -> List[FrameExtraction]:
        """
        Extract only frames with detected motion.

        Args:
            video_path: Path to video file
            frame_interval: Check every Nth frame for motion
            motion_threshold: Minimum motion score to include frame (0.0-1.0)
            max_frames: Maximum number of frames to extract (None = all)

        Returns:
            List of frames with motion detected

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If motion detection is not enabled
        """
        if not self.motion_config.enabled:
            raise RuntimeError("Motion detection is not enabled")

        if not 0.0 <= motion_threshold <= 1.0:
            raise ValueError(f"Motion threshold must be 0-1, got {motion_threshold}")

        all_frames = self.extract_frames(video_path, frame_interval, None)

        motion_frames = [
            frame for frame in all_frames if frame.motion_score >= motion_threshold
        ]

        if max_frames is not None:
            motion_frames = motion_frames[:max_frames]

        logger.info(
            f"Filtered {len(motion_frames)} motion frames from "
            f"{len(all_frames)} total frames "
            f"(threshold={motion_threshold})"
        )

        return motion_frames

    def _detect_motion(self, frame: NDArray[np.uint8]) -> Tuple[bool, float]:
        """
        Detect motion in a frame using background subtraction.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Tuple of (has_motion, motion_score)
        """
        if self.bg_subtractor is None:
            return False, 0.0

        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(
                frame, learningRate=self.motion_config.learning_rate
            )

            # Threshold the mask
            _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Calculate total motion area
            total_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.motion_config.min_area:
                    total_area += area

            # Calculate motion score (normalized by frame size)
            frame_area = frame.shape[0] * frame.shape[1]
            motion_score = min(1.0, total_area / (frame_area * 0.1))

            has_motion = total_area >= self.motion_config.min_area

            return has_motion, float(motion_score)

        except Exception as e:
            logger.warning(f"Motion detection failed: {e}")
            return False, 0.0

    def save_frame(
        self, frame: FrameExtraction, output_path: Path, format: str = "jpg"
    ) -> None:
        """
        Save extracted frame to file.

        Args:
            frame: Frame to save
            output_path: Output file path
            format: Image format (jpg, png, etc.)

        Raises:
            ValueError: If format is invalid
            RuntimeError: If save fails
        """
        valid_formats = ["jpg", "jpeg", "png", "bmp"]
        if format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid format '{format}', must be one of {valid_formats}"
            )

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert RGB back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)

            # Save image
            success = cv2.imwrite(str(output_path), frame_bgr)

            if not success:
                raise RuntimeError(f"Failed to save frame to {output_path}")

            logger.debug(
                f"Saved frame {frame.frame_number} to {output_path} "
                f"(motion_score={frame.motion_score:.3f})"
            )

        except Exception as e:
            logger.error(f"Failed to save frame: {e}", exc_info=True)
            raise RuntimeError(f"Frame save failed: {e}")
