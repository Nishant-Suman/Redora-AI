"""
PDF Voice Reader AI - Flask Application & REST API.
High-performance REST API and web server for Computer Vision Document Preprocessing,
Multi-Engine OCR, NLP Text Cleaning, and Neural Speech Synthesis.
"""

from typing import Dict, Any, Optional
import os
import io
import time
import base64
import uuid
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
from PIL import Image

# Import Core Pipeline Modules
from core.pdf_processor import PDFProcessor
from core.image_preprocessor import ImagePreprocessor
from core.ocr_engine import OCREngine
from core.text_cleaner import TextCleaner
from core.tts_engine import TTSEngine
from utils.audio_visualizer import AudioVisualizer
from utils.export_manager import ExportManager

# -----------------------------------------------------------------------------
# Flask App Setup
# -----------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "pdf-voice-reader-ai-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit
CORS(app)

# Initialize engines
image_preprocessor = ImagePreprocessor()
ocr_engine = OCREngine(default_lang="en", use_gpu=False)
text_cleaner = TextCleaner()
tts_engine = TTSEngine()

# Persistent & In-memory session / document storage
CACHE_DIR = os.path.join(os.path.dirname(__file__), "tmp_doc_store")
os.makedirs(CACHE_DIR, exist_ok=True)
DOC_STORE: Dict[str, Dict[str, Any]] = {}
LAST_ACTIVE_DOC_ID: Optional[str] = None


def save_doc_data(doc_id: str, doc_data: Dict[str, Any]) -> None:
    """Stores document in memory and persists to disk for zero-loss survival across restarts."""
    global LAST_ACTIVE_DOC_ID
    DOC_STORE[doc_id] = doc_data
    LAST_ACTIVE_DOC_ID = doc_id

    try:
        doc_dir = os.path.join(CACHE_DIR, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        if doc_data.get("file_bytes"):
            with open(os.path.join(doc_dir, "document.bin"), "wb") as f:
                f.write(doc_data["file_bytes"])

        import json
        meta = {
            "id": doc_id,
            "filename": doc_data.get("filename", "document"),
            "is_pdf": doc_data.get("is_pdf", False),
            "total_pages": doc_data.get("total_pages", 1),
            "current_page": doc_data.get("current_page", 1),
            "cv_meta": doc_data.get("cv_meta", {}),
            "page_info": doc_data.get("page_info", {}),
        }
        with open(os.path.join(doc_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception as e:
        print(f"[DocStore] Disk persist warning: {e}")


def get_doc_data(doc_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Retrieves document from in-memory cache, disk storage, or last active document fallback."""
    global LAST_ACTIVE_DOC_ID
    target_id = doc_id or LAST_ACTIVE_DOC_ID

    if not target_id:
        if DOC_STORE:
            target_id = list(DOC_STORE.keys())[-1]

    if target_id and target_id in DOC_STORE:
        return DOC_STORE[target_id]

    # Try restoring from disk cache
    if target_id:
        doc_dir = os.path.join(CACHE_DIR, target_id)
        meta_path = os.path.join(doc_dir, "meta.json")
        bin_path = os.path.join(doc_dir, "document.bin")

        if os.path.exists(meta_path) and os.path.exists(bin_path):
            try:
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                with open(bin_path, "rb") as f:
                    file_bytes = f.read()

                is_pdf = meta.get("is_pdf", False)
                curr_page = meta.get("current_page", 1)

                if is_pdf:
                    pdf_proc = PDFProcessor(file_bytes)
                    total_pages = pdf_proc.page_count
                    p_img = pdf_proc.render_page_to_image(curr_page, dpi=200)
                    p_info = pdf_proc.get_page_info(curr_page)
                    pdf_proc.close()
                else:
                    p_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    total_pages = 1
                    p_info = {"page_number": 1, "width": p_img.width, "height": p_img.height, "is_scanned": True}

                prep_img, cv_meta = image_preprocessor.preprocess_pipeline(p_img)

                doc_data = {
                    "id": target_id,
                    "filename": meta.get("filename", "document"),
                    "is_pdf": is_pdf,
                    "file_bytes": file_bytes,
                    "total_pages": total_pages,
                    "current_page": curr_page,
                    "current_image": p_img,
                    "preprocessed_image": prep_img,
                    "cv_meta": cv_meta,
                    "page_info": p_info,
                    "raw_text": "",
                    "cleaned_text": "",
                    "audio_bytes": None,
                    "audio_format": "mp3",
                    "subtitle_events": [],
                }
                DOC_STORE[target_id] = doc_data
                LAST_ACTIVE_DOC_ID = target_id
                return doc_data
            except Exception as e:
                print(f"[DocStore] Disk restore failed for {target_id}: {e}")

    # Fallback to any recent disk cache directory if available
    try:
        if os.path.exists(CACHE_DIR):
            subdirs = [os.path.join(CACHE_DIR, d) for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                recent_id = os.path.basename(latest_dir)
                if recent_id != target_id:
                    return get_doc_data(recent_id)
    except Exception:
        pass

    return None


def pil_to_data_uri(pil_img: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a base64 data URI string."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"


def data_uri_to_pil(data_uri: str) -> Optional[Image.Image]:
    """Converts a base64 data URI to a PIL Image."""
    try:
        if "," in data_uri:
            data_uri = data_uri.split(",", 1)[1]
        img_bytes = base64.b64decode(data_uri)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"[DocStore] Base64 decode failed: {e}")
        return None


# -----------------------------------------------------------------------------
# Routes & API Endpoints
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """Renders the main single-page application dashboard."""
    voices = TTSEngine.get_available_voices()
    return render_template("index.html", voices=voices)


@app.route("/api/voices", methods=["GET"])
def get_voices():
    """Returns the list of all supported neural voices and dialects."""
    return jsonify({
        "status": "success",
        "voices": TTSEngine.get_available_voices()
    })


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Handles PDF or image uploads.
    Renders first page, applies preprocessing, and returns metadata & images.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    file_bytes = file.read()
    filename = file.filename
    doc_id = str(uuid.uuid4())

    is_pdf = filename.lower().endswith(".pdf")
    doc_data: Dict[str, Any] = {
        "id": doc_id,
        "filename": filename,
        "is_pdf": is_pdf,
        "file_bytes": file_bytes,
        "total_pages": 1,
        "current_page": 1,
        "current_image": None,
        "preprocessed_image": None,
        "cv_meta": {},
        "raw_text": "",
        "cleaned_text": "",
        "audio_bytes": None,
        "audio_format": "mp3",
        "subtitle_events": [],
    }

    try:
        if is_pdf:
            pdf_proc = PDFProcessor(file_bytes)
            doc_data["total_pages"] = pdf_proc.page_count
            p1_img = pdf_proc.render_page_to_image(1, dpi=200)
            doc_data["current_image"] = p1_img
            p_info = pdf_proc.get_page_info(1)
            doc_data["page_info"] = p_info
            pdf_proc.close()
        else:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            doc_data["current_image"] = img
            doc_data["total_pages"] = 1
            doc_data["page_info"] = {
                "page_number": 1,
                "width": img.width,
                "height": img.height,
                "is_scanned": True
            }

        # Apply default computer vision preprocessing
        prep_img, cv_meta = image_preprocessor.preprocess_pipeline(
            doc_data["current_image"],
            do_deskew=True,
            do_denoise=True,
            do_enhance=True,
            do_binarize=False
        )
        doc_data["preprocessed_image"] = prep_img
        doc_data["cv_meta"] = cv_meta

        # Persist document to memory & disk
        save_doc_data(doc_id, doc_data)

        return jsonify({
            "status": "success",
            "doc_id": doc_id,
            "filename": filename,
            "is_pdf": is_pdf,
            "total_pages": doc_data["total_pages"],
            "current_page": 1,
            "page_info": doc_data["page_info"],
            "cv_meta": cv_meta,
            "original_image_url": pil_to_data_uri(doc_data["current_image"]),
            "preprocessed_image_url": pil_to_data_uri(prep_img)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to process document: {str(e)}"}), 500


@app.route("/api/page/<doc_id>/<int:page_num>", methods=["GET"])
def get_page(doc_id: str, page_num: int):
    """Renders and returns a specific page for a multi-page document."""
    doc_data = get_doc_data(doc_id)
    if not doc_data:
        return jsonify({"status": "error", "message": "Document not found or expired."}), 404

    if not doc_data["is_pdf"]:
        return jsonify({"status": "error", "message": "Document is a single image."}), 400

    if page_num < 1 or page_num > doc_data["total_pages"]:
        return jsonify({"status": "error", "message": f"Page {page_num} out of bounds."}), 400

    try:
        pdf_proc = PDFProcessor(doc_data["file_bytes"])
        p_img = pdf_proc.render_page_to_image(page_num, dpi=200)
        p_info = pdf_proc.get_page_info(page_num)
        pdf_proc.close()

        # Preprocess
        prep_img, cv_meta = image_preprocessor.preprocess_pipeline(
            p_img,
            do_deskew=True,
            do_denoise=True,
            do_enhance=True,
            do_binarize=False
        )

        doc_data["current_page"] = page_num
        doc_data["current_image"] = p_img
        doc_data["preprocessed_image"] = prep_img
        doc_data["page_info"] = p_info
        doc_data["cv_meta"] = cv_meta

        save_doc_data(doc_id, doc_data)

        return jsonify({
            "status": "success",
            "doc_id": doc_id,
            "current_page": page_num,
            "page_info": p_info,
            "cv_meta": cv_meta,
            "original_image_url": pil_to_data_uri(p_img),
            "preprocessed_image_url": pil_to_data_uri(prep_img)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/preprocess", methods=["POST"])
def reprocess_image():
    """Re-runs image preprocessing with custom parameters."""
    data = request.get_json() or {}
    doc_id = data.get("doc_id")
    image_data_uri = data.get("image_data")

    doc_data = get_doc_data(doc_id)
    source_img = None

    if doc_data and doc_data.get("current_image") is not None:
        source_img = doc_data["current_image"]
    elif image_data_uri:
        source_img = data_uri_to_pil(image_data_uri)

    if source_img is None:
        return jsonify({"status": "error", "message": "No image available to preprocess."}), 400

    do_deskew = data.get("do_deskew", True)
    do_denoise = data.get("do_denoise", True)
    do_enhance = data.get("do_enhance", True)
    do_binarize = data.get("do_binarize", False)
    binarize_mode = data.get("binarize_mode", "otsu")

    prep_img, cv_meta = image_preprocessor.preprocess_pipeline(
        source_img,
        do_deskew=do_deskew,
        do_denoise=do_denoise,
        do_enhance=do_enhance,
        do_binarize=do_binarize,
        binarize_method=binarize_mode
    )

    if doc_data:
        doc_data["preprocessed_image"] = prep_img
        doc_data["cv_meta"] = cv_meta
        if doc_id:
            save_doc_data(doc_id, doc_data)

    return jsonify({
        "status": "success",
        "cv_meta": cv_meta,
        "preprocessed_image_url": pil_to_data_uri(prep_img)
    })


@app.route("/api/ocr", methods=["POST"])
def run_ocr():
    """Runs OCR and NLP cleaning pipeline on current page with multiple fallbacks."""
    data = request.get_json() or {}
    doc_id = data.get("doc_id")
    image_data_uri = data.get("image_data")
    lang = data.get("lang", "en")
    preferred_engine = data.get("engine", "easyocr")
    process_all_pages = data.get("process_all_pages", False)

    doc_data = get_doc_data(doc_id)
    extracted_raw = ""
    conf_score = 100.0

    try:
        if doc_data and process_all_pages and doc_data.get("is_pdf"):
            # Process all pages of loaded PDF
            pdf_proc = PDFProcessor(doc_data["file_bytes"])
            all_texts = []
            for p in range(1, doc_data["total_pages"] + 1):
                p_info = pdf_proc.get_page_info(p)
                if not p_info["is_scanned"]:
                    t = pdf_proc.extract_direct_text(p)
                else:
                    p_img = pdf_proc.render_page_to_image(p, dpi=200)
                    p_prep, _ = image_preprocessor.preprocess_pipeline(p_img)
                    ocr_res = ocr_engine.process_image(p_prep, lang=lang)
                    t = ocr_res["text"]
                all_texts.append(f"--- Page {p} ---\n{t}")
            pdf_proc.close()
            extracted_raw = "\n\n".join(all_texts)
        else:
            # Single page extraction
            if doc_data and doc_data.get("is_pdf"):
                try:
                    pdf_proc = PDFProcessor(doc_data["file_bytes"])
                    p_info = pdf_proc.get_page_info(doc_data["current_page"])
                    if not p_info["is_scanned"]:
                        extracted_raw = pdf_proc.extract_direct_text(doc_data["current_page"])
                    pdf_proc.close()
                except Exception:
                    pass

            if not extracted_raw or len(extracted_raw.strip()) < 30:
                # Find image to OCR
                img_to_ocr = None
                if doc_data:
                    img_to_ocr = doc_data.get("preprocessed_image") or doc_data.get("current_image")
                if img_to_ocr is None and image_data_uri:
                    img_to_ocr = data_uri_to_pil(image_data_uri)

                if img_to_ocr is None:
                    # Try loading default sample image if no document found
                    sample_path = "sample_documents/sample_scanned_page.png"
                    if os.path.exists(sample_path):
                        img_to_ocr = Image.open(sample_path).convert("RGB")

                if img_to_ocr is None:
                    return jsonify({"status": "error", "message": "No document or image available to extract."}), 400

                ocr_res = ocr_engine.process_image(img_to_ocr, preferred_engine=preferred_engine, lang=lang)
                extracted_raw = ocr_res.get("text", "")
                conf_score = ocr_res.get("confidence", 85.0)

        # NLP Cleaning Pipeline
        cleaned_text = text_cleaner.clean_text(extracted_raw)
        stats = text_cleaner.calculate_statistics(cleaned_text)
        sentences = text_cleaner.split_into_sentences(cleaned_text)

        if doc_data:
            doc_data["raw_text"] = extracted_raw
            doc_data["cleaned_text"] = cleaned_text
            if doc_id:
                save_doc_data(doc_id, doc_data)

        return jsonify({
            "status": "success",
            "raw_text": extracted_raw,
            "cleaned_text": cleaned_text,
            "confidence": conf_score,
            "stats": stats,
            "sentences": sentences
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"OCR extraction failed: {str(e)}"}), 500


@app.route("/api/clean-text", methods=["POST"])
def clean_custom_text():
    """Cleans and calculates stats for arbitrary text input."""
    data = request.get_json() or {}
    raw_text = data.get("text", "")
    cleaned = text_cleaner.clean_text(raw_text)
    stats = text_cleaner.calculate_statistics(cleaned)
    sentences = text_cleaner.split_into_sentences(cleaned)

    return jsonify({
        "status": "success",
        "cleaned_text": cleaned,
        "stats": stats,
        "sentences": sentences
    })


@app.route("/api/tts", methods=["POST"])
def synthesize_speech():
    """
    Synthesizes speech from text using specified engine, voice, speed, and pitch.
    Returns base64 audio data URI, duration, and subtitle events.
    """
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    doc_id = data.get("doc_id")
    engine = data.get("engine", "edge-tts")
    voice = data.get("voice", "en-US-JennyNeural")
    speed = float(data.get("speed", 1.0))
    pitch = int(data.get("pitch", 0))
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"status": "error", "message": "Text is empty."}), 400

    try:
        start_time = time.time()
        audio_bytes, subtitle_events, audio_fmt = tts_engine.synthesize(
            text=text,
            engine=engine,
            voice=voice,
            speed=speed,
            pitch_hz=pitch,
            lang=lang
        )
        elapsed = round(time.time() - start_time, 2)

        # Audio metadata, Waveform, and Mel Spectrogram
        audio_info = AudioVisualizer.get_audio_info(audio_bytes, audio_fmt)
        waveform_png = AudioVisualizer.generate_waveform_image(audio_bytes, audio_fmt)
        waveform_b64 = f"data:image/png;base64,{base64.b64encode(waveform_png).decode('utf-8')}"

        spectrogram_png = AudioVisualizer.generate_melspectrogram_image(audio_bytes, audio_fmt)
        spectrogram_b64 = f"data:image/png;base64,{base64.b64encode(spectrogram_png).decode('utf-8')}"

        # Cache audio in doc store if doc_id provided
        if doc_id and doc_id in DOC_STORE:
            DOC_STORE[doc_id]["audio_bytes"] = audio_bytes
            DOC_STORE[doc_id]["audio_format"] = audio_fmt
            DOC_STORE[doc_id]["subtitle_events"] = subtitle_events
            DOC_STORE[doc_id]["cleaned_text"] = text

        # Subtitle SRT generation
        if subtitle_events:
            srt_content = text_cleaner.generate_srt_subtitles(subtitle_events)
        else:
            # Synthetic sentence timings
            sentences = text_cleaner.split_into_sentences(text)
            dur = audio_info["duration_seconds"]
            per_s = dur / max(1, len(sentences))
            synth_timings = [
                {"start_sec": i * per_s, "end_sec": (i + 1) * per_s, "text": s}
                for i, s in enumerate(sentences)
            ]
            srt_content = text_cleaner.generate_srt_subtitles(synth_timings)
            subtitle_events = synth_timings

        # Format audio as base64 data URI
        mime = "audio/mp3" if audio_fmt == "mp3" else "audio/wav"
        audio_data_uri = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('utf-8')}"

        return jsonify({
            "status": "success",
            "audio_data_uri": audio_data_uri,
            "audio_format": audio_fmt,
            "duration_seconds": audio_info["duration_seconds"],
            "formatted_duration": audio_info["formatted_duration"],
            "file_size_kb": audio_info["file_size_kb"],
            "waveform_image_url": waveform_b64,
            "spectrogram_image_url": spectrogram_b64,
            "subtitle_events": subtitle_events,
            "srt_content": srt_content,
            "synthesis_time_seconds": elapsed
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Speech synthesis failed: {str(e)}"}), 500


@app.route("/api/spectrogram", methods=["GET", "POST"])
def get_spectrogram():
    """Generates an acoustic Mel-spectrogram image for default or requested audio."""
    spec_png = AudioVisualizer.generate_melspectrogram_image(None)
    spec_b64 = f"data:image/png;base64,{base64.b64encode(spec_png).decode('utf-8')}"
    return jsonify({
        "status": "success",
        "spectrogram_image_url": spec_b64
    })


@app.route("/api/sample/<sample_type>", methods=["GET"])
def load_sample(sample_type: str):
    """Loads bundled sample documents for 1-click evaluation."""
    if sample_type == "pdf":
        sample_path = "sample_documents/sample_document.pdf"
    elif sample_type == "image":
        sample_path = "sample_documents/sample_scanned_page.png"
    else:
        return jsonify({"status": "error", "message": "Unknown sample type."}), 400

    if not os.path.exists(sample_path):
        from sample_documents.create_samples import create_sample_scanned_image, create_sample_pdf
        create_sample_scanned_image()
        create_sample_pdf()

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    doc_id = str(uuid.uuid4())
    filename = os.path.basename(sample_path)
    is_pdf = filename.endswith(".pdf")

    doc_data: Dict[str, Any] = {
        "id": doc_id,
        "filename": filename,
        "is_pdf": is_pdf,
        "file_bytes": file_bytes,
        "total_pages": 1,
        "current_page": 1,
        "current_image": None,
        "preprocessed_image": None,
        "cv_meta": {},
        "raw_text": "",
        "cleaned_text": "",
        "audio_bytes": None,
        "audio_format": "mp3",
        "subtitle_events": [],
    }

    if is_pdf:
        pdf_proc = PDFProcessor(file_bytes)
        doc_data["total_pages"] = pdf_proc.page_count
        p1_img = pdf_proc.render_page_to_image(1, dpi=200)
        doc_data["current_image"] = p1_img
        doc_data["page_info"] = pdf_proc.get_page_info(1)
        pdf_proc.close()
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        doc_data["current_image"] = img
        doc_data["page_info"] = {"page_number": 1, "width": img.width, "height": img.height, "is_scanned": True}

    prep_img, cv_meta = image_preprocessor.preprocess_pipeline(doc_data["current_image"])
    doc_data["preprocessed_image"] = prep_img
    doc_data["cv_meta"] = cv_meta

    save_doc_data(doc_id, doc_data)

    return jsonify({
        "status": "success",
        "doc_id": doc_id,
        "filename": filename,
        "is_pdf": is_pdf,
        "total_pages": doc_data["total_pages"],
        "current_page": 1,
        "page_info": doc_data["page_info"],
        "cv_meta": cv_meta,
        "original_image_url": pil_to_data_uri(doc_data["current_image"]),
        "preprocessed_image_url": pil_to_data_uri(prep_img)
    })


@app.route("/api/export/<doc_id>/<export_type>", methods=["GET"])
def export_asset(doc_id: str, export_type: str):
    """
    Downloads generated output files:
    export_type: 'audio', 'srt', 'transcript', 'zip'
    """
    doc_data = get_doc_data(doc_id)
    if not doc_data:
        return jsonify({"status": "error", "message": "Document not found."}), 404

    doc_data = DOC_STORE[doc_id]
    audio_bytes = doc_data.get("audio_bytes")
    cleaned_text = doc_data.get("cleaned_text", "")
    fmt = doc_data.get("audio_format", "mp3")

    if export_type == "audio":
        if not audio_bytes:
            return jsonify({"status": "error", "message": "No audio generated yet."}), 400
        mime = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        return send_file(
            io.BytesIO(audio_bytes),
            mimetype=mime,
            as_attachment=True,
            download_name=f"speech_{doc_id[:8]}.{fmt}"
        )

    elif export_type == "transcript":
        return send_file(
            io.BytesIO(cleaned_text.encode("utf-8")),
            mimetype="text/plain; charset=utf-8",
            as_attachment=True,
            download_name=f"transcript_{doc_id[:8]}.txt"
        )

    elif export_type == "srt":
        sub_events = doc_data.get("subtitle_events", [])
        if sub_events:
            srt_str = text_cleaner.generate_srt_subtitles(sub_events)
        else:
            sentences = text_cleaner.split_into_sentences(cleaned_text)
            srt_str = text_cleaner.generate_srt_subtitles([
                {"start_sec": i * 3.0, "end_sec": (i + 1) * 3.0, "text": s}
                for i, s in enumerate(sentences)
            ])
        return send_file(
            io.BytesIO(srt_str.encode("utf-8")),
            mimetype="text/plain; charset=utf-8",
            as_attachment=True,
            download_name=f"subtitles_{doc_id[:8]}.srt"
        )

    elif export_type == "zip":
        if not audio_bytes:
            return jsonify({"status": "error", "message": "Generate audio before downloading bundle."}), 400

        sub_events = doc_data.get("subtitle_events", [])
        srt_str = text_cleaner.generate_srt_subtitles(sub_events) if sub_events else ""

        zip_bytes = ExportManager.create_zip_bundle(
            audio_bytes=audio_bytes,
            audio_ext=fmt,
            clean_text=cleaned_text,
            srt_subtitles=srt_str,
            metadata={"doc_id": doc_id, "filename": doc_data["filename"]}
        )
        return send_file(
            io.BytesIO(zip_bytes),
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"pdf_voice_reader_{doc_id[:8]}.zip"
        )

    return jsonify({"status": "error", "message": "Unknown export type."}), 400


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting PDF Voice Reader Flask App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
