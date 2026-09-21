"""
PDF Voice Reader AI - Advanced Streamlit Application.
Converts Scanned PDFs, Images, and Digital Documents into Natural Human-like Speech
with Computer Vision Preprocessing, Multi-Engine OCR, NLP Cleaning, and Neural TTS.
"""

import os
import io
import time
import streamlit as st
from PIL import Image

# Import Core Modules
from core.pdf_processor import PDFProcessor
from core.image_preprocessor import ImagePreprocessor
from core.ocr_engine import OCREngine
from core.text_cleaner import TextCleaner
from core.tts_engine import TTSEngine
from utils.audio_visualizer import AudioVisualizer
from utils.export_manager import ExportManager

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Voice Reader AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "static", "css", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "current_page_idx" not in st.session_state:
    st.session_state.current_page_idx = 1
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "audio_format" not in st.session_state:
    st.session_state.audio_format = "mp3"
if "subtitle_events" not in st.session_state:
    st.session_state.subtitle_events = []
if "ocr_meta" not in st.session_state:
    st.session_state.ocr_meta = {}
if "cv_meta" not in st.session_state:
    st.session_state.cv_meta = {}
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

# Instantiate Engines (cached via st.cache_resource)
@st.cache_resource
def get_engines():
    image_prep = ImagePreprocessor()
    ocr = OCREngine(default_lang="en", use_gpu=False)
    cleaner = TextCleaner()
    tts = TTSEngine()
    return image_prep, ocr, cleaner, tts

image_prep_engine, ocr_engine, text_cleaner, tts_engine = get_engines()

# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div>
            <h1 class="hero-title">📚 PDF Voice Reader AI</h1>
            <p class="hero-subtitle">
                Computer Vision Preprocessing • Multi-Engine OCR • Intelligent NLP Cleaning • Neural Speech Synthesis
            </p>
        </div>
        <div>
            <span class="badge-tag">⚡ Neural AI Powered</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Voice & Synthesis Settings")

    # Synthesis Engine
    tts_engine_choice = st.selectbox(
        "TTS Engine",
        options=["Edge-TTS (Neural HD Voices)", "gTTS (Google Cloud)", "pyttsx3 (Offline Local)"],
        index=0,
        help="Edge-TTS provides high-fidelity natural neural speech with pitch & rate controls."
    )
    engine_key_map = {
        "Edge-TTS (Neural HD Voices)": "edge-tts",
        "gTTS (Google Cloud)": "gtts",
        "pyttsx3 (Offline Local)": "pyttsx3",
    }
    selected_engine = engine_key_map[tts_engine_choice]

    # Voice Selection
    all_voices = TTSEngine.get_available_voices()
    voice_labels = [f"{v['lang']} - {v['name']}" for v in all_voices.values()]
    voice_keys = list(all_voices.keys())

    selected_voice_label = st.selectbox(
        "Voice Character",
        options=voice_labels,
        index=1,  # Jenny US Natural
        help="Select AI speaker profile, dialect, and gender."
    )
    selected_voice_key = voice_keys[voice_labels.index(selected_voice_label)]

    # Speech Controls
    st.subheader("🎚️ Audio Tuning")
    speed_rate = st.slider(
        "Speech Speed",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        format="%.2fx",
        help="Adjust reading playback pace."
    )
    pitch_val = st.slider(
        "Pitch Shift (Hz)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        format="%d Hz",
        help="Adjust voice pitch frequency."
    )

    st.markdown("---")
    st.header("🔬 Preprocessing & OCR")
    
    with st.expander("🛠️ Computer Vision Tuning", expanded=False):
        do_deskew = st.checkbox("Auto-Deskew (Rotation Correction)", value=True)
        do_denoise = st.checkbox("Bilateral Denoising (Clean Speckles)", value=True)
        do_enhance = st.checkbox("CLAHE Contrast Boost", value=True)
        do_binarize = st.checkbox("Adaptive Binarization", value=False)
        binarize_mode = st.selectbox(
            "Binarize Algorithm",
            options=["otsu", "adaptive_gaussian", "adaptive_mean"],
            index=0
        )

    with st.expander("🔤 OCR Engine Options", expanded=False):
        ocr_lang = st.selectbox("OCR Language", options=["en", "es", "fr", "de", "hi"], index=0)
        ocr_pref = st.selectbox("Preferred OCR", options=["EasyOCR (Deep Learning)", "Direct Text (Digital PDF)"], index=0)
        direct_only = (ocr_pref == "Direct Text (Digital PDF)")

    st.markdown("---")
    st.caption("PDF Voice Reader AI v2.0 • Antigravity Edition")

# -----------------------------------------------------------------------------
# Input Source Selection
# -----------------------------------------------------------------------------
st.markdown("### 📥 1. Select Document Source")
input_mode = st.radio(
    "Choose input method:",
    options=["📄 Upload PDF / Image File", "⚡ Load Interactive Sample Demo", "✍️ Direct Text Input"],
    horizontal=True,
)

current_doc_image: Image.Image = None
current_pdf_processor: PDFProcessor = None
total_pages = 1

if input_mode == "📄 Upload PDF / Image File":
    uploaded_file = st.file_uploader(
        "Upload your scanned or digital document (PDF, PNG, JPG, JPEG, TIFF, BMP, WebP)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        help="Drag and drop your PDF or image here."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            try:
                current_pdf_processor = PDFProcessor(file_bytes)
                total_pages = current_pdf_processor.page_count
                
                col_pg, col_info = st.columns([2, 3])
                with col_pg:
                    page_sel = st.number_input(
                        f"Select Page (Total: {total_pages})",
                        min_value=1,
                        max_value=max(1, total_pages),
                        value=st.session_state.current_page_idx,
                        step=1,
                    )
                    st.session_state.current_page_idx = page_sel
                
                with col_info:
                    p_info = current_pdf_processor.get_page_info(page_sel)
                    st.info(
                        f"📄 **Page {page_sel} of {total_pages}** | Type: "
                        f"{'Scanned Image' if p_info['is_scanned'] else 'Digital Text PDF'} | "
                        f"Dimensions: {int(p_info['width'])}x{int(p_info['height'])}"
                    )

                current_doc_image = current_pdf_processor.render_page_to_image(page_sel, dpi=200)
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        else:
            # Direct Image upload
            try:
                current_doc_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                st.success(f"🖼️ Image loaded: {uploaded_file.name} ({current_doc_image.width}x{current_doc_image.height} px)")
            except Exception as e:
                st.error(f"Error reading image: {e}")

elif input_mode == "⚡ Load Interactive Sample Demo":
    sample_col1, sample_col2 = st.columns(2)
    sample_choice = None
    with sample_col1:
        if st.button("📑 Load Sample Multi-Page PDF", use_container_width=True):
            sample_choice = "sample_documents/sample_document.pdf"
    with sample_col2:
        if st.button("🖼️ Load Sample Scanned Page (Skewed & Grainy)", use_container_width=True):
            sample_choice = "sample_documents/sample_scanned_page.png"

    if sample_choice or st.session_state.get("last_sample"):
        if sample_choice:
            st.session_state.last_sample = sample_choice
        path = st.session_state.last_sample

        if path.endswith(".pdf") and os.path.exists(path):
            current_pdf_processor = PDFProcessor(path)
            total_pages = current_pdf_processor.page_count
            page_sel = st.number_input(
                f"Select Page (Total: {total_pages})",
                min_value=1,
                max_value=max(1, total_pages),
                value=st.session_state.current_page_idx,
                step=1,
            )
            st.session_state.current_page_idx = page_sel
            current_doc_image = current_pdf_processor.render_page_to_image(page_sel, dpi=200)
            st.success(f"Loaded Sample PDF: Page {page_sel} of {total_pages}")
        elif os.path.exists(path):
            current_doc_image = Image.open(path).convert("RGB")
            st.success("Loaded Sample Scanned Document with simulated skew & paper noise.")

elif input_mode == "✍️ Direct Text Input":
    st.session_state.extracted_text = st.text_area(
        "Enter or paste text directly for speech synthesis:",
        value=st.session_state.extracted_text or "Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        height=180,
    )

# -----------------------------------------------------------------------------
# Step 2: Image Preprocessing & OCR Extraction
# -----------------------------------------------------------------------------
if current_doc_image is not None:
    st.markdown("---")
    st.markdown("### 🔍 2. Document Preprocessing & OCR Extraction")

    # Run CV Preprocessing
    with st.spinner("Applying Computer Vision enhancement pipeline..."):
        preprocessed_img, cv_meta = image_prep_engine.preprocess_pipeline(
            current_doc_image,
            do_deskew=do_deskew,
            do_denoise=do_denoise,
            do_enhance=do_enhance,
            do_binarize=do_binarize,
            binarize_method=binarize_mode,
        )
        st.session_state.cv_meta = cv_meta

    # Display Side-by-Side Images
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original Document Page**")
        st.image(current_doc_image, use_container_width=True)
    with img_col2:
        st.markdown(f"**Enhanced & Deskewed Image** (Skew Angle: `{cv_meta.get('skew_angle', 0.0)}°`)")
        st.image(preprocessed_img, use_container_width=True)

    # Extraction Button
    col_btn, col_btn_all = st.columns([1, 1])
    with col_btn:
        extract_clicked = st.button("🔍 Extract Text from Current Page", type="primary", use_container_width=True)
    with col_btn_all:
        extract_all_clicked = False
        if current_pdf_processor and total_pages > 1:
            extract_all_clicked = st.button(f"📑 Extract All {total_pages} Pages", use_container_width=True)

    if extract_clicked:
        with st.spinner("Running Multi-Engine OCR and Intelligent NLP Cleaning..."):
            extracted_raw = ""
            conf_score = 100.0

            if direct_only and current_pdf_processor:
                extracted_raw = current_pdf_processor.extract_direct_text(st.session_state.current_page_idx)
            else:
                # If digital text exists in PDF and is rich, try it first or use OCR
                if current_pdf_processor and not current_pdf_processor.get_page_info(st.session_state.current_page_idx)["is_scanned"]:
                    extracted_raw = current_pdf_processor.extract_direct_text(st.session_state.current_page_idx)
                
                if not extracted_raw or len(extracted_raw.strip()) < 30:
                    ocr_res = ocr_engine.process_image(preprocessed_img, preferred_engine="easyocr", lang=ocr_lang)
                    extracted_raw = ocr_res["text"]
                    conf_score = ocr_res.get("confidence", 85.0)

            # NLP Cleaning Pipeline
            cleaned = text_cleaner.clean_text(extracted_raw)
            st.session_state.raw_text = extracted_raw
            st.session_state.extracted_text = cleaned
            st.session_state.ocr_meta = {"confidence": conf_score}
            st.success(f"Extraction complete! OCR Confidence: {conf_score:.1f}%")

    if extract_all_clicked and current_pdf_processor:
        with st.spinner(f"Extracting and processing all {total_pages} pages..."):
            all_raw = []
            progress_bar = st.progress(0)
            for p in range(1, total_pages + 1):
                p_info = current_pdf_processor.get_page_info(p)
                if not p_info["is_scanned"]:
                    t = current_pdf_processor.extract_direct_text(p)
                else:
                    p_img = current_pdf_processor.render_page_to_image(p, dpi=200)
                    p_prep, _ = image_prep_engine.preprocess_pipeline(p_img, do_deskew=do_deskew, do_denoise=do_denoise)
                    ocr_res = ocr_engine.process_image(p_prep, lang=ocr_lang)
                    t = ocr_res["text"]
                all_raw.append(f"--- Page {p} ---\n{t}")
                progress_bar.progress(p / total_pages)

            full_raw = "\n\n".join(all_raw)
            cleaned = text_cleaner.clean_text(full_raw)
            st.session_state.raw_text = full_raw
            st.session_state.extracted_text = cleaned
            st.success(f"Extracted all {total_pages} pages successfully!")

# -----------------------------------------------------------------------------
# Step 3: Extracted Text & NLP Inspection
# -----------------------------------------------------------------------------
if st.session_state.extracted_text:
    st.markdown("---")
    st.markdown("### 📝 3. Extracted & Cleaned Text")

    stats = text_cleaner.calculate_statistics(st.session_state.extracted_text)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><p class="metric-val">{stats["word_count"]}</p><p class="metric-lbl">Words</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><p class="metric-val">{stats["character_count"]}</p><p class="metric-lbl">Characters</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><p class="metric-val">{stats["sentence_count"]}</p><p class="metric-lbl">Sentences</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><p class="metric-val">{stats["formatted_duration"]}</p><p class="metric-lbl">Est. Audio Time</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_edit, tab_raw, tab_sentences = st.tabs(["✍️ Editable Clean Text", "📄 Raw OCR Output", "🧩 Sentence Chunks"])
    with tab_edit:
        edited_text = st.text_area(
            "Review and edit the cleaned text prior to voice synthesis:",
            value=st.session_state.extracted_text,
            height=200,
            key="edited_text_area",
        )
        if edited_text != st.session_state.extracted_text:
            st.session_state.extracted_text = edited_text

    with tab_raw:
        st.text_area("Raw Uncleaned OCR / PDF Extraction:", value=st.session_state.raw_text, height=200, disabled=True)

    with tab_sentences:
        sentences = text_cleaner.split_into_sentences(st.session_state.extracted_text)
        for idx, s in enumerate(sentences, 1):
            st.markdown(f'<div class="sentence-item"><strong>{idx}.</strong> {s}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Step 4: Neural TTS Voice Generation & Audio Player Studio
# -----------------------------------------------------------------------------
if st.session_state.extracted_text:
    st.markdown("---")
    st.markdown("### 🔊 4. Neural Speech Synthesis & Audio Studio")

    synth_col1, synth_col2 = st.columns([2, 1])
    with synth_col1:
        generate_audio_clicked = st.button(
            "🔊 Generate Neural Voice Audio",
            type="primary",
            use_container_width=True,
            help="Synthesizes the cleaned text into high-fidelity neural audio.",
        )
    with synth_col2:
        voice_info = all_voices.get(selected_voice_key, {})
        st.caption(f"🎤 **Engine:** {tts_engine_choice} | **Voice:** {voice_info.get('name', selected_voice_key)} | **Speed:** {speed_rate}x | **Pitch:** {pitch_val:+d}Hz")

    if generate_audio_clicked:
        with st.spinner("Synthesizing neural voice stream with acoustic modeling..."):
            try:
                start_time = time.time()
                audio_bytes, sub_events, fmt = tts_engine.synthesize(
                    text=st.session_state.extracted_text,
                    engine=selected_engine,
                    voice=selected_voice_key,
                    speed=speed_rate,
                    pitch_hz=pitch_val,
                    lang=ocr_lang,
                )
                elapsed = round(time.time() - start_time, 2)
                st.session_state.audio_bytes = audio_bytes
                st.session_state.audio_format = fmt
                st.session_state.subtitle_events = sub_events
                st.success(f"Audio synthesized successfully in {elapsed}s! Ready for playback.")
            except Exception as e:
                st.error(f"Voice generation failed: {e}")

    # Audio Playback & Studio Display
    if st.session_state.audio_bytes:
        audio_info = AudioVisualizer.get_audio_info(st.session_state.audio_bytes, st.session_state.audio_format)
        
        st.markdown(
            f"""
            <div class="audio-studio-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; color: #FFFFFF;">🎧 Master Audio Output</h3>
                    <span class="badge-tag">Duration: {audio_info['formatted_duration']} ({audio_info['duration_seconds']}s) • {audio_info['file_size_kb']} KB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Native Audio Player
        mime_type = "audio/mp3" if st.session_state.audio_format == "mp3" else "audio/wav"
        st.audio(st.session_state.audio_bytes, format=mime_type)

        # Waveform Graphic
        st.markdown("**Acoustic Waveform Analysis**")
        waveform_png = AudioVisualizer.generate_waveform_image(st.session_state.audio_bytes, st.session_state.audio_format)
        st.image(waveform_png, use_container_width=True)

        # Generate SRT Subtitles
        srt_content = ""
        if st.session_state.subtitle_events:
            srt_content = text_cleaner.generate_srt_subtitles(st.session_state.subtitle_events)

        # Export & Download Buttons
        st.markdown("### 💾 Export & Download Assets")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        with d_col1:
            st.download_button(
                label=f"⬇️ Download Audio ({st.session_state.audio_format.upper()})",
                data=st.session_state.audio_bytes,
                file_name=f"voice_reader_speech.{st.session_state.audio_format}",
                mime=mime_type,
                use_container_width=True,
            )

        with d_col2:
            if srt_content:
                st.download_button(
                    label="📝 Download SRT Subtitles",
                    data=srt_content,
                    file_name="voice_reader_subtitles.srt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                # Generate synthetic timings
                sentences = text_cleaner.split_into_sentences(st.session_state.extracted_text)
                dur = audio_info["duration_seconds"]
                per_s = dur / max(1, len(sentences))
                synth_timings = [
                    {"start_sec": i * per_s, "end_sec": (i + 1) * per_s, "text": s}
                    for i, s in enumerate(sentences)
                ]
                srt_synth = text_cleaner.generate_srt_subtitles(synth_timings)
                st.download_button(
                    label="📝 Download SRT Subtitles",
                    data=srt_synth,
                    file_name="voice_reader_subtitles.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with d_col3:
            st.download_button(
                label="📄 Download Clean Transcript",
                data=st.session_state.extracted_text,
                file_name="voice_reader_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with d_col4:
            # Create Full ZIP Bundle
            metadata = {
                "voice": selected_voice_key,
                "engine": selected_engine,
                "speed": speed_rate,
                "pitch": pitch_val,
                "word_count": stats["word_count"],
                "duration_seconds": audio_info["duration_seconds"],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            zip_bundle = ExportManager.create_zip_bundle(
                audio_bytes=st.session_state.audio_bytes,
                audio_ext=st.session_state.audio_format,
                clean_text=st.session_state.extracted_text,
                srt_subtitles=srt_content or srt_synth,
                metadata=metadata,
            )
            st.download_button(
                label="📦 Download Complete ZIP Bundle",
                data=zip_bundle,
                file_name="pdf_voice_reader_package.zip",
                mime="application/zip",
                use_container_width=True,
            )
