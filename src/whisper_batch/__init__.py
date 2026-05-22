"""whisper-batch: parallel whisper.cpp transcription via silence-aware chunking."""

from .config import Config
from .pipeline import transcribe_file
from .types import Chunk, Segment, Transcript

__all__ = ["Config", "transcribe_file", "Chunk", "Segment", "Transcript"]
__version__ = "0.1.0"
