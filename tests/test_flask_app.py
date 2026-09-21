"""
Flask REST API & Web Application Test Suite.
"""

import json
import unittest
from flask_app import app, DOC_STORE


class TestFlaskVoiceReaderApp(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_index_page(self):
        """Test home page rendering."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Nishant", response.data)

    def test_get_voices(self):
        """Test /api/voices endpoint."""
        response = self.client.get("/api/voices")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("en-US-JennyNeural", data["voices"])

    def test_load_sample_pdf(self):
        """Test loading sample PDF."""
        response = self.client.get("/api/sample/pdf")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["is_pdf"])
        self.assertGreater(data["total_pages"], 0)
        self.assertIn("data:image", data["original_image_url"])
        self.assertIn("data:image", data["preprocessed_image_url"])

    def test_load_sample_image(self):
        """Test loading sample scanned page."""
        response = self.client.get("/api/sample/image")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("doc_id", data)

    def test_clean_text_api(self):
        """Test /api/clean-text endpoint."""
        payload = {"text": "State-of-the-art ma-\nchine learning al-\ngorithms."}
        response = self.client.post(
            "/api/clean-text",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("machine", data["cleaned_text"])
        self.assertIn("algorithms", data["cleaned_text"])
        self.assertGreater(data["stats"]["word_count"], 0)

    def test_tts_api(self):
        """Test /api/tts speech synthesis."""
        payload = {
            "text": "Testing Flask neural speech synthesis.",
            "engine": "edge-tts",
            "voice": "en-US-JennyNeural",
            "speed": 1.0,
            "pitch": 0,
            "lang": "en"
        }
        response = self.client.post(
            "/api/tts",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("data:audio/mp3;base64,", data["audio_data_uri"])
        self.assertGreater(data["duration_seconds"], 0)
        self.assertIn("waveform_image_url", data)

    def test_export_api(self):
        """Test export endpoints."""
        # Load sample first to create doc_id
        res_sample = self.client.get("/api/sample/pdf")
        doc_id = json.loads(res_sample.data)["doc_id"]

        # Synthesize audio for it
        payload = {
            "doc_id": doc_id,
            "text": "Sample export audio.",
            "engine": "edge-tts",
            "voice": "en-US-JennyNeural"
        }
        self.client.post("/api/tts", data=json.dumps(payload), content_type="application/json")

        # Test audio export
        res_audio = self.client.get(f"/api/export/{doc_id}/audio")
        self.assertEqual(res_audio.status_code, 200)

        # Test transcript export
        res_txt = self.client.get(f"/api/export/{doc_id}/transcript")
        self.assertEqual(res_txt.status_code, 200)

        # Test zip bundle export
        res_zip = self.client.get(f"/api/export/{doc_id}/zip")
        self.assertEqual(res_zip.status_code, 200)

    def test_spectrogram_api(self):
        """Test /api/spectrogram endpoint."""
        response = self.client.get("/api/spectrogram")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("data:image/png;base64,", data["spectrogram_image_url"])


if __name__ == "__main__":
    unittest.main()
