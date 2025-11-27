"""
Tests for video processing module.

Tests video info extraction, frame extraction, motion detection,
and integration with the detection pipeline.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.video_processor import (
    VideoInfo,
    FrameExtraction,
    MotionDetectionConfig,
    VideoProcessor,
)
from utils.validators import ValidationError

logger = logging.getLogger(__name__)


class TestVideoInfo:
    """Test VideoInfo dataclass validation."""

    def test_valid_video_info(self) -> None:
        """Test creation of valid video info."""
        info = VideoInfo(
            path=Path("test.mp4"),
            fps=30.0,
            frame_count=900,
            width=1920,
            height=1080,
            duration_seconds=30.0,
            codec="h264",
        )

        assert info.path == Path("test.mp4")
        assert info.fps == 30.0
        assert info.frame_count == 900
        assert info.width == 1920
        assert info.height == 1080
        assert info.duration_seconds == 30.0
        assert info.codec == "h264"

    def test_invalid_fps(self) -> None:
        """Test validation fails for invalid FPS."""
        with pytest.raises(ValueError, match="FPS must be > 0"):
            VideoInfo(
                path=Path("test.mp4"),
                fps=0.0,
                frame_count=900,
                width=1920,
                height=1080,
                duration_seconds=30.0,
                codec="h264",
            )

    def test_invalid_frame_count(self) -> None:
        """Test validation fails for negative frame count."""
        with pytest.raises(ValueError, match="Frame count must be >= 0"):
            VideoInfo(
                path=Path("test.mp4"),
                fps=30.0,
                frame_count=-1,
                width=1920,
                height=1080,
                duration_seconds=30.0,
                codec="h264",
            )

    def test_invalid_dimensions(self) -> None:
        """Test validation fails for invalid dimensions."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            VideoInfo(
                path=Path("test.mp4"),
                fps=30.0,
                frame_count=900,
                width=0,
                height=1080,
                duration_seconds=30.0,
                codec="h264",
            )


class TestFrameExtraction:
    """Test FrameExtraction dataclass validation."""

    def test_valid_frame_extraction(self) -> None:
        """Test creation of valid frame extraction."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = FrameExtraction(
            frame_number=10,
            timestamp=0.333,
            image=image,
            has_motion=True,
            motion_score=0.75,
        )

        assert frame.frame_number == 10
        assert frame.timestamp == 0.333
        assert frame.has_motion is True
        assert frame.motion_score == 0.75
        assert np.array_equal(frame.image, image)

    def test_invalid_frame_number(self) -> None:
        """Test validation fails for negative frame number."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Frame number must be >= 0"):
            FrameExtraction(
                frame_number=-1, timestamp=0.0, image=image, has_motion=False
            )

    def test_invalid_timestamp(self) -> None:
        """Test validation fails for negative timestamp."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Timestamp must be >= 0"):
            FrameExtraction(
                frame_number=0, timestamp=-1.0, image=image, has_motion=False
            )

    def test_invalid_motion_score(self) -> None:
        """Test validation fails for invalid motion score."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Motion score must be 0-1"):
            FrameExtraction(
                frame_number=0,
                timestamp=0.0,
                image=image,
                has_motion=False,
                motion_score=1.5,
            )


class TestMotionDetectionConfig:
    """Test MotionDetectionConfig validation."""

    def test_valid_config(self) -> None:
        """Test creation of valid motion config."""
        config = MotionDetectionConfig(
            enabled=True,
            threshold=30,
            min_area=1000,
            learning_rate=0.02,
            history=300,
        )

        assert config.enabled is True
        assert config.threshold == 30
        assert config.min_area == 1000
        assert config.learning_rate == 0.02
        assert config.history == 300

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MotionDetectionConfig()

        assert config.enabled is True
        assert config.threshold == 25
        assert config.min_area == 500
        assert config.learning_rate == 0.01
        assert config.history == 500

    def test_invalid_threshold(self) -> None:
        """Test validation fails for invalid threshold."""
        with pytest.raises(ValueError, match="Threshold must be 0-255"):
            MotionDetectionConfig(threshold=300)

    def test_invalid_min_area(self) -> None:
        """Test validation fails for negative min area."""
        with pytest.raises(ValueError, match="Min area must be >= 0"):
            MotionDetectionConfig(min_area=-1)

    def test_invalid_learning_rate(self) -> None:
        """Test validation fails for invalid learning rate."""
        with pytest.raises(ValueError, match="Learning rate must be 0-1"):
            MotionDetectionConfig(learning_rate=1.5)

    def test_invalid_history(self) -> None:
        """Test validation fails for invalid history."""
        with pytest.raises(ValueError, match="History must be >= 1"):
            MotionDetectionConfig(history=0)


class TestVideoProcessor:
    """Test VideoProcessor functionality."""

    def test_processor_initialization(self) -> None:
        """Test processor initializes correctly."""
        config = MotionDetectionConfig(enabled=True)
        processor = VideoProcessor(motion_config=config)

        assert processor.motion_config.enabled is True
        assert processor.bg_subtractor is not None

    def test_processor_without_motion_detection(self) -> None:
        """Test processor with motion detection disabled."""
        config = MotionDetectionConfig(enabled=False)
        processor = VideoProcessor(motion_config=config)

        assert processor.motion_config.enabled is False
        assert processor.bg_subtractor is None

    def test_get_video_info_synthetic(self, tmp_path: Path) -> None:
        """Test getting video info from synthetic video."""
        # Create a simple synthetic video
        video_path = tmp_path / "test_video.mp4"
        self._create_test_video(video_path, frames=30, fps=10.0)

        processor = VideoProcessor()
        info = processor.get_video_info(video_path)

        assert info.path == video_path
        assert info.fps == 10.0
        assert info.frame_count == 30
        assert info.width == 320
        assert info.height == 240
        assert info.duration_seconds == pytest.approx(3.0, abs=0.1)

    def test_get_video_info_nonexistent_file(self) -> None:
        """Test error handling for nonexistent file."""
        processor = VideoProcessor()

        with pytest.raises(ValidationError):
            processor.get_video_info(Path("/nonexistent/video.mp4"))

    def test_extract_frames_all(self, tmp_path: Path) -> None:
        """Test extracting all frames from video."""
        video_path = tmp_path / "test_video.mp4"
        self._create_test_video(video_path, frames=10, fps=10.0)

        processor = VideoProcessor()
        frames = processor.extract_frames(video_path, frame_interval=1)

        assert len(frames) == 10
        assert all(isinstance(f, FrameExtraction) for f in frames)
        assert frames[0].frame_number == 0
        assert frames[-1].frame_number == 9

    def test_extract_frames_interval(self, tmp_path: Path) -> None:
        """Test extracting frames with interval."""
        video_path = tmp_path / "test_video.mp4"
        self._create_test_video(video_path, frames=20, fps=10.0)

        processor = VideoProcessor()
        frames = processor.extract_frames(video_path, frame_interval=5)

        # Should extract frames 0, 5, 10, 15
        assert len(frames) == 4
        assert frames[0].frame_number == 0
        assert frames[1].frame_number == 5
        assert frames[2].frame_number == 10
        assert frames[3].frame_number == 15

    def test_extract_frames_max_limit(self, tmp_path: Path) -> None:
        """Test extracting frames with max limit."""
        video_path = tmp_path / "test_video.mp4"
        self._create_test_video(video_path, frames=20, fps=10.0)

        processor = VideoProcessor()
        frames = processor.extract_frames(video_path, frame_interval=1, max_frames=5)

        assert len(frames) == 5

    def test_extract_frames_invalid_interval(self, tmp_path: Path) -> None:
        """Test error handling for invalid frame interval."""
        video_path = tmp_path / "test_video.mp4"
        self._create_test_video(video_path, frames=10, fps=10.0)

        processor = VideoProcessor()

        with pytest.raises(ValueError, match="Frame interval must be >= 1"):
            processor.extract_frames(video_path, frame_interval=0)

    def test_extract_motion_frames(self, tmp_path: Path) -> None:
        """Test extracting frames with motion."""
        video_path = tmp_path / "test_video.mp4"
        self._create_test_video_with_motion(video_path)

        config = MotionDetectionConfig(enabled=True, min_area=100)
        processor = VideoProcessor(motion_config=config)

        motion_frames = processor.extract_motion_frames(
            video_path, frame_interval=1, motion_threshold=0.01
        )

        # Should detect some motion frames
        assert len(motion_frames) > 0
        assert all(frame.has_motion for frame in motion_frames)

    def test_extract_motion_frames_disabled(self, tmp_path: Path) -> None:
        """Test error when motion detection is disabled."""
        video_path = tmp_path / "test_video.mp4"
        self._create_test_video(video_path, frames=10, fps=10.0)

        config = MotionDetectionConfig(enabled=False)
        processor = VideoProcessor(motion_config=config)

        with pytest.raises(RuntimeError, match="Motion detection is not enabled"):
            processor.extract_motion_frames(video_path)

    def test_save_frame(self, tmp_path: Path) -> None:
        """Test saving extracted frame."""
        output_path = tmp_path / "frame.jpg"
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        frame = FrameExtraction(
            frame_number=0, timestamp=0.0, image=image, has_motion=False
        )

        processor = VideoProcessor()
        processor.save_frame(frame, output_path, format="jpg")

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_frame_invalid_format(self, tmp_path: Path) -> None:
        """Test error handling for invalid format."""
        output_path = tmp_path / "frame.xyz"
        image = np.zeros((240, 320, 3), dtype=np.uint8)

        frame = FrameExtraction(
            frame_number=0, timestamp=0.0, image=image, has_motion=False
        )

        processor = VideoProcessor()

        with pytest.raises(ValueError, match="Invalid format"):
            processor.save_frame(frame, output_path, format="xyz")

    def test_frame_timestamps(self, tmp_path: Path) -> None:
        """Test frame timestamps are calculated correctly."""
        video_path = tmp_path / "test_video.mp4"
        self._create_test_video(video_path, frames=30, fps=10.0)

        processor = VideoProcessor()
        frames = processor.extract_frames(video_path, frame_interval=10)

        # Should be frames 0, 10, 20 at timestamps 0.0, 1.0, 2.0
        assert len(frames) == 3
        assert frames[0].timestamp == pytest.approx(0.0, abs=0.1)
        assert frames[1].timestamp == pytest.approx(1.0, abs=0.1)
        assert frames[2].timestamp == pytest.approx(2.0, abs=0.1)

    # Helper methods for creating test videos

    @staticmethod
    def _create_test_video(
        path: Path, frames: int = 30, fps: float = 10.0, width: int = 320, height: int = 240
    ) -> None:
        """Create a simple test video with static frames."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

        try:
            for i in range(frames):
                # Create frame with gradient
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :] = (i * 255 // frames, 128, 255 - i * 255 // frames)
                out.write(frame)
        finally:
            out.release()

    @staticmethod
    def _create_test_video_with_motion(
        path: Path, frames: int = 30, fps: float = 10.0
    ) -> None:
        """Create a test video with moving object."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))

        try:
            for i in range(frames):
                # Create frame with static background
                frame = np.ones((240, 320, 3), dtype=np.uint8) * 128

                # Add moving rectangle
                x = (i * 10) % 280
                cv2.rectangle(frame, (x, 100), (x + 40, 140), (255, 255, 255), -1)

                out.write(frame)
        finally:
            out.release()
