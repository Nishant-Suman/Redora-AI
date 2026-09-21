"""
Multi-Engine Optical Character Recognition (OCR) Layer.
Supports EasyOCR (Deep Learning), Tesseract, and Direct Native Extraction with confidence scoring.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from PIL import Image
import cv2

# Global EasyOCR reader cache
_EASYOCR_READERS: Dict[str, Any] = {}


class OCREngine:
    """Robust Multi-Engine OCR system with deep learning recognition and fallback."""

    def __init__(self, default_lang: str = "en", use_gpu: bool = False):
        self.default_lang = default_lang
        self.use_gpu = use_gpu

    def _get_easyocr_reader(self, lang: str = "en"):
        """Loads and caches EasyOCR reader for specified language."""
        global _EASYOCR_READERS
        lang_key = f"{lang}_{self.use_gpu}"
        if lang_key not in _EASYOCR_READERS:
            try:
                import easyocr
                # Map language codes if necessary
                langs = [lang] if isinstance(lang, str) else lang
                _EASYOCR_READERS[lang_key] = easyocr.Reader(langs, gpu=self.use_gpu, verbose=False)
            except Exception as e:
                print(f"[OCREngine] Warning: EasyOCR reader initialization failed: {e}")
                return None
        return _EASYOCR_READERS[lang_key]

    def extract_with_easyocr(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
        lang: str = "en",
        detail: int = 1
    ) -> Dict[str, Any]:
        """
        Runs EasyOCR on image.
        Returns:
            {
                "text": str,
                "confidence": float (0-100),
                "blocks": [{"bbox": [...], "text": str, "conf": float}, ...],
                "engine": "easyocr"
            }
        """
        reader = self._get_easyocr_reader(lang)
        if reader is None:
            raise RuntimeError("EasyOCR could not be initialized.")

        # Ensure image is in numpy format (RGB)
        if isinstance(image_input, Image.Image):
            img_np = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_np = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 3:
                img_np = image_input  # assumes RGB or BGR, EasyOCR handles both
            else:
                img_np = image_input
        elif isinstance(image_input, str):
            img_np = image_input
        else:
            raise TypeError("Unsupported image format for EasyOCR.")

        results = reader.readtext(img_np, detail=1)

        extracted_lines = []
        blocks = []
        confidences = []

        for bbox, text, conf in results:
            clean_t = text.strip()
            if clean_t:
                extracted_lines.append(clean_t)
                confidences.append(conf)
                blocks.append({
                    "bbox": [[int(coord) for coord in pt] for pt in bbox],
                    "text": clean_t,
                    "confidence": round(float(conf) * 100, 2)
                })

        full_text = "\n".join(extracted_lines)
        avg_conf = round(float(np.mean(confidences)) * 100, 2) if confidences else 0.0

        return {
            "text": full_text,
            "confidence": avg_conf,
            "blocks": blocks,
            "engine": "easyocr",
            "block_count": len(blocks)
        }

    def extract_with_tesseract(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
        lang: str = "eng"
    ) -> Dict[str, Any]:
        """Runs PyTesseract OCR if available."""
        try:
            import pytesseract
            if isinstance(image_input, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            elif isinstance(image_input, str):
                pil_img = Image.open(image_input)
            else:
                pil_img = image_input

            text = pytesseract.image_to_string(pil_img, lang=lang).strip()
            # PyTesseract data for confidence
            data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)
            valid_confs = [float(c) for c in data.get("conf", []) if str(c) not in ("-1", "")]
            avg_conf = round(float(np.mean(valid_confs)), 2) if valid_confs else 80.0

            return {
                "text": text,
                "confidence": avg_conf,
                "blocks": [],
                "engine": "tesseract",
                "block_count": len(text.splitlines())
            }
        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "blocks": [],
                "engine": "tesseract_failed",
                "error": str(e)
            }

    def process_image(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
        preferred_engine: str = "easyocr",
        lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Executes OCR with automatic fallback strategy.
        Preferred engines: 'easyocr', 'tesseract'
        """
        if preferred_engine == "easyocr":
            try:
                return self.extract_with_easyocr(image_input, lang=lang)
            except Exception as e:
                print(f"[OCREngine] EasyOCR failed: {e}. Falling back to Tesseract...")
                tess_res = self.extract_with_tesseract(image_input)
                if tess_res.get("text"):
                    return tess_res
                raise RuntimeError(f"OCR failed across available engines: {e}")
        elif preferred_engine == "tesseract":
            tess_res = self.extract_with_tesseract(image_input)
            if tess_res.get("text"):
                return tess_res
            # Fallback to easyocr
            return self.extract_with_easyocr(image_input, lang=lang)
        else:
            return self.extract_with_easyocr(image_input, lang=lang)
