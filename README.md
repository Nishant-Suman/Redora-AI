# 📚 Readora AI — PDF Voice Reader & Speech AI
**Created by R.M.Nishant Suman**  
📧 Email: [rmnishantsuman@gmail.com](mailto:rmnishantsuman@gmail.com) | 📞 Phone: [+91 9508222833](tel:+919508222833)

An enterprise-grade, end-to-end AI system that converts scanned and digital PDF documents or images into natural, human-like voice synthesis with high-DPI document rendering, Computer Vision preprocessing (Deskewing, Denoising, CLAHE enhancement, Binarization), Multi-Engine OCR (EasyOCR & Direct Text Extraction), NLP Text Cleaning, and Neural Text-to-Speech synthesis with acoustic waveform visualization.

---

## ⚡ Zero-Setup Environment with UV

This project is fully managed with **`uv`**. You **do not need** to manually create or activate virtual environments, install Python versions, or run `pip install` by hand. `uv` handles Python version management, dependency resolution, lockfiles, and execution automatically.

### 🚀 Run Instantly with UV:

#### 1. Launch the Flask Web Server (Default)
```powershell
uv run python flask_app.py
```
> Open your browser at **`http://127.0.0.1:5000`**.

#### 2. Or 1-Click Double-Click
Simply double-click **`run_flask.bat`** in Windows Explorer.

#### 3. Run the Automated Test Suite
```powershell
uv run python -m unittest discover tests
```

#### 4. (Optional) Run Streamlit Interface
```powershell
uv run streamlit run app.py
```

---

## 🚀 Key Features

1. **Dual Frontend Options**:
   - **Flask Web Application**: Modern SPA (Single Page Application) with Tailwind CSS, Lucide icons, glassmorphic UI, real-time sentence karaoke playback tracker, and RESTful JSON APIs.
   - **Streamlit Application**: Rapid dashboard with interactive sliders and visual inspection.

2. **Document & Scanned PDF Ingestion**:
   - High-fidelity PDF page rendering up to 300 DPI using `PyMuPDF`.
   - Automatic classification between digital text PDFs and scanned document images.
   - Multi-page navigation and batch page processing.

3. **Computer Vision Document Preprocessing**:
   - **Hough Angle Deskewing**: Automatically detects skew angles and rotates documents upright.
   - **Bilateral / Median Denoising**: Removes scanner speckles, paper grain, and compression artifacts.
   - **CLAHE Contrast Enhancement**: Balances uneven lighting, faint ink, and shadows.
   - **Adaptive & Otsu Binarization**: Produces crisp black-and-white character separation for maximum OCR accuracy.

4. **Multi-Engine OCR & NLP Cleaner**:
   - Deep learning OCR (`EasyOCR` PyTorch pipeline) + PyMuPDF digital parser.
   - Line-break hyphenation repair (`inter-\nconnected` ➜ `interconnected`).
   - Page number, recurring header, and footer filtering.
   - Punctuation restoration and smart sentence segmentation.
   - Real-time word count, character count, and listening duration metrics.

5. **Neural Voice Synthesis (Edge-TTS + Fallbacks)**:
   - 50+ ultra-realistic Microsoft Edge Neural voices (Male / Female / Regional accents: US, UK, India, Australia, Canada, Spain, France, Germany, Hindi, etc.).
   - Granular speed control (`0.5x` to `2.0x`) and pitch adjustment (`-50Hz` to `+50Hz`).
   - Resilient fallbacks to Google `gTTS` and 100% offline `pyttsx3`.

6. **Audio Studio & Export Suite**:
   - Native audio playback with acoustic waveform visualizer.
   - Interactive sentence-synchronized subtitle highlight (Karaoke tracking).
   - 1-Click download buttons for MP3/WAV, SRT subtitles, Clean text, and full ZIP bundles.

---

## 🛠️ Architecture

```
📄 Scanned PDF / Image / Digital PDF
              │
              ▼
   ┌───────────────────────┐
   │ PDF & Image Ingestion │  (PyMuPDF / PIL high-DPI rasterization)
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ Computer Vision Prep  │  (Deskewing, Denoising, CLAHE, Binarization)
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ Multi-Engine OCR      │  (EasyOCR Deep Learning / Digital Text Layer)
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ NLP Cleaning & Chunks │  (Dehyphenation, Header/Footer Strip, Metrics)
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ Neural TTS Synthesizer│  (Edge-TTS HD Voices / gTTS / pyttsx3)
   └──────────┬────────────┘
              ▼
       🔊 Audio Output
   (MP3 / WAV + Waveform + SRT Subtitles + Transcript)
```

---

## 📦 Project Structure

```
Text-Voice Ai/
├── pyproject.toml              # UV project configuration and dependencies
├── uv.lock                     # UV deterministic lockfile
├── .python-version             # Pinned Python version (3.10)
├── run_flask.bat               # 1-Click Windows batch runner for Flask
├── run_streamlit.bat           # 1-Click Windows batch runner for Streamlit
├── flask_app.py                # Main Flask Server & REST API
├── app.py                      # Alternative Streamlit Web Application
├── templates/
│   └── index.html              # Modern Flask SPA Dashboard (Tailwind + Lucide)
├── static/
│   ├── css/
│   │   ├── style.css           # Glassmorphic dark theme styles
│   │   └── custom.css          # Streamlit theme styles
│   └── js/
│       └── app.js              # Client-side controller with audio & sentence sync
├── core/
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF page extraction & high-DPI rendering
│   ├── image_preprocessor.py   # OpenCV deskewing, denoising, CLAHE, binarization
│   ├── ocr_engine.py           # EasyOCR & Direct PDF OCR engine
│   ├── text_cleaner.py         # Advanced NLP cleaner, hyphenation repair & stats
│   └── tts_engine.py           # Neural Edge-TTS, gTTS, pyttsx3 synthesizer
├── utils/
│   ├── __init__.py
│   ├── audio_visualizer.py     # Waveform plotting & acoustic analysis
│   └── export_manager.py       # ZIP bundling, SRT & transcript export
├── sample_documents/           # Sample PDF & Scanned image test files
│   ├── sample_document.pdf
│   ├── sample_scanned_page.png
│   └── create_samples.py
├── tests/
│   ├── test_pipeline.py        # Core pipeline test suite
│   └── test_flask_app.py       # Flask REST API test suite
└── requirements.txt            # Python dependencies
```

---

## 🌐 Flask REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main Web Application Dashboard |
| `/api/voices` | `GET` | Returns list of all 50+ available neural voices and accents |
| `/api/upload` | `POST` | Uploads PDF/image, renders page, applies initial CV preprocessing |
| `/api/page/<doc_id>/<page_num>` | `GET` | Fetches rendered image and preprocessed version for specific page |
| `/api/preprocess` | `POST` | Re-executes CV filters with custom settings (deskew, denoise, binarize) |
| `/api/ocr` | `POST` | Runs EasyOCR / digital text extraction + NLP cleaning |
| `/api/clean-text` | `POST` | Cleans arbitrary text string and returns metrics |
| `/api/tts` | `POST` | Synthesizes speech using Edge-TTS with speed & pitch controls |
| `/api/sample/<sample_type>` | `GET` | Loads 1-click sample document (`pdf` or `image`) |
| `/api/export/<doc_id>/<type>` | `GET` | Downloads generated assets (`audio`, `srt`, `transcript`, `zip`) |
