"""
Comprehensive Pipeline Test Suite for PDF Voice Reader AI.
"""

import os
import unittest
import numpy as np
from PIL import Image
import cv2

from core.pdf_processor import PDFProcessor
from core.image_preprocessor import ImagePreprocessor
from core.text_cleaner import TextCleaner
from core.tts_engine import TTSEngine
from utils.audio_visualizer import AudioVisualizer
from utils.export_manager import ExportManager


class TestPDFVoiceReaderPipeline(unittest.TestCase):

    def setUp(self):
        self.image_preprocessor = ImagePreprocessor()
        self.text_cleaner = TextCleaner()
        self.tts_engine = TTSEngine()

    def test_text_cleaner_dehyphenation(self):
        """Test line-break hyphenation repair."""
        raw = "This is a multi-\nlayered neural archi-\ntecture."
        cleaned = self.text_cleaner.clean_text(raw)
        self.assertIn("multilayered", cleaned)
        self.assertIn("architecture", cleaned)
        self.assertNotIn("multi-", cleaned)

    def test_text_cleaner_headers_footers(self):
        """Test removal of page numbers and headers."""
        raw = "Page 1 of 10\nMachine learning is transformative.\n- 12 -\n===================="
        cleaned = self.text_cleaner.clean_text(raw)
        self.assertIn("Machine learning is transformative.", cleaned)
        self.assertNotIn("Page 1 of 10", cleaned)
        self.assertNotIn("- 12 -", cleaned)

    def test_text_cleaner_statistics(self):
        """Test calculation of text statistics."""
        text = "Deep learning enables machines to process language. Text-to-speech models produce lifelike voices."
        stats = self.text_cleaner.calculate_statistics(text)
        self.assertEqual(stats["sentence_count"], 2)
        self.assertGreater(stats["word_count"], 10)
        self.assertGreater(stats["character_count"], 50)
        self.assertIn("formatted_duration", stats)

    def test_image_preprocessor_deskew_and_enhancement(self):
        """Test image deskewing and contrast enhancement."""
        # Create synthetic image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Test OCR Preprocessing", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Rotate slightly
        center = (300, 200)
        rot = cv2.getRotationMatrix2D(center, 3.0, 1.0)
        rotated = cv2.warpAffine(img, rot, (600, 400), borderValue=(255, 255, 255))
        
        pil_res, meta = self.image_preprocessor.preprocess_pipeline(
            rotated, do_deskew=True, do_denoise=True, do_enhance=True, do_binarize=True
        )
        self.assertIsInstance(pil_res, Image.Image)
        self.assertIn("skew_angle", meta)
        self.assertTrue(meta.get("binarized"))

    def test_pdf_processor(self):
        """Test PDF loading, metadata inspection, and page image rendering."""
        pdf_path = "sample_documents/sample_document.pdf"
        if os.path.exists(pdf_path):
            with PDFProcessor(pdf_path) as proc:
                self.assertEqual(proc.page_count, 2)
                info = proc.get_page_info(1)
                self.assertEqual(info["page_number"], 1)
                self.assertFalse(info["is_scanned"])
                
                # Render page to image
                img = proc.render_page_to_image(1, dpi=100)
                self.assertIsInstance(img, Image.Image)
                self.assertGreater(img.width, 0)
                self.assertGreater(img.height, 0)

                # Extract text
                txt = proc.extract_direct_text(1)
                self.assertIn("Machine learning", txt)

    def test_tts_engine_and_visualizer(self):
        """Test TTS voice synthesis and audio visualizer."""
        test_text = "Testing speech synthesis pipeline."
        audio_bytes, events, fmt = self.tts_engine.synthesize(
            text=test_text,
            engine="edge-tts",
            voice="en-US-JennyNeural",
            speed=1.0,
            pitch_hz=0
        )
        self.assertIsInstance(audio_bytes, bytes)
        self.assertGreater(len(audio_bytes), 1000)
        self.assertEqual(fmt, "mp3")

        # Test audio visualizer metadata and waveform image
        meta = AudioVisualizer.get_audio_info(audio_bytes, fmt)
        self.assertGreater(meta["duration_seconds"], 0)
        
        waveform_png = AudioVisualizer.generate_waveform_image(audio_bytes, fmt)
        self.assertIsInstance(waveform_png, bytes)
        self.assertGreater(len(waveform_png), 500)

    def test_export_manager(self):
        """Test ZIP bundle packaging."""
        dummy_audio = b"FAKE_AUDIO_DATA_BYTES"
        clean_text = "This is the transcript."
        srt_subtitles = "1\n00:00:00,000 --> 00:00:02,000\nThis is the transcript.\n"
        meta = {"test": True}

        zip_bytes = ExportManager.create_zip_bundle(
            audio_bytes=dummy_audio,
            audio_ext="mp3",
            clean_text=clean_text,
            srt_subtitles=srt_subtitles,
            metadata=meta
        )
        self.assertIsInstance(zip_bytes, bytes)
        self.assertGreater(len(zip_bytes), 100)


if __name__ == "__main__":
    unittest.main()
