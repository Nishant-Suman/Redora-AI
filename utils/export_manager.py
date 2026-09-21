"""
Export and Packaging Manager for Audio, Subtitles, and Transcripts.
"""

from typing import Dict, Any, Optional
import io
import zipfile
import json
import time


class ExportManager:
    """Handles bundling, formatting, and packaging of audio and transcript outputs."""

    @staticmethod
    def create_zip_bundle(
        audio_bytes: bytes,
        audio_ext: str = "mp3",
        clean_text: str = "",
        srt_subtitles: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        base_name: str = "pdf_speech",
    ) -> bytes:
        """
        Creates an in-memory ZIP package containing:
        - Audio file (MP3/WAV)
        - Cleaned text transcript (.txt and .md)
        - Subtitles (.srt)
        - Metadata JSON summary (.json)
        """
        zip_buf = io.BytesIO()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{base_name}_{timestamp}"

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Audio file
            zf.writestr(f"{prefix}.{audio_ext}", audio_bytes)

            # 2. Text transcript (.txt)
            if clean_text:
                zf.writestr(f"{prefix}_transcript.txt", clean_text.encode("utf-8"))

                # 3. Markdown version
                md_content = f"# Voice Reader Transcript\n\n**Generated:** {timestamp}\n\n---\n\n{clean_text}\n"
                zf.writestr(f"{prefix}_transcript.md", md_content.encode("utf-8"))

            # 4. SRT Subtitles
            if srt_subtitles:
                zf.writestr(f"{prefix}_subtitles.srt", srt_subtitles.encode("utf-8"))

            # 5. Metadata JSON
            if metadata:
                zf.writestr(f"{prefix}_metadata.json", json.dumps(metadata, indent=2).encode("utf-8"))

        zip_buf.seek(0)
        return zip_buf.read()
