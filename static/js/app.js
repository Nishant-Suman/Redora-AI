/**
 * PDF Voice Reader AI - Interactive Frontend Controller
 * Handles document upload, CV preprocessing, OCR extraction,
 * neural TTS synthesis, acoustic waveform visualizer, and sentence-highlight sync.
 */

// Application State
const state = {
    docId: null,
    filename: null,
    isPdf: false,
    currentPage: 1,
    totalPages: 1,
    originalImageUrl: null,
    preprocessedImageUrl: null,
    rawText: "",
    cleanedText: "",
    stats: {},
    audioDataUri: null,
    audioFormat: "mp3",
    subtitleEvents: [],
    audioDuration: 0,
};

// DOM Elements Cache
const elements = {
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("fileInput"),
    uploadStatus: document.getElementById("uploadStatus"),
    samplePdfBtn: document.getElementById("samplePdfBtn"),
    sampleImgBtn: document.getElementById("sampleImgBtn"),
    
    // Page Navigator
    pageNavigator: document.getElementById("pageNavigator"),
    prevPageBtn: document.getElementById("prevPageBtn"),
    nextPageBtn: document.getElementById("nextPageBtn"),
    pageIndicator: document.getElementById("pageIndicator"),
    
    // Preprocessing Images
    originalImg: document.getElementById("originalImg"),
    preprocessedImg: document.getElementById("preprocessedImg"),
    skewAngleBadge: document.getElementById("skewAngleBadge"),
    
    // CV Toggles
    deskewToggle: document.getElementById("deskewToggle"),
    denoiseToggle: document.getElementById("denoiseToggle"),
    enhanceToggle: document.getElementById("enhanceToggle"),
    binarizeToggle: document.getElementById("binarizeToggle"),
    binarizeMode: document.getElementById("binarizeMode"),
    applyCvBtn: document.getElementById("applyCvBtn"),
    
    // OCR & Text
    ocrEngineSelect: document.getElementById("ocrEngineSelect"),
    ocrLangSelect: document.getElementById("ocrLangSelect"),
    extractTextBtn: document.getElementById("extractTextBtn"),
    extractAllBtn: document.getElementById("extractAllBtn"),
    ocrSpinner: document.getElementById("ocrSpinner"),
    
    // Text Viewers
    cleanedTextArea: document.getElementById("cleanedTextArea"),
    rawTextArea: document.getElementById("rawTextArea"),
    sentenceList: document.getElementById("sentenceList"),
    
    // Stats
    statWords: document.getElementById("statWords"),
    statChars: document.getElementById("statChars"),
    statSentences: document.getElementById("statSentences"),
    statDuration: document.getElementById("statDuration"),
    
    // TTS Controls
    ttsEngineSelect: document.getElementById("ttsEngineSelect"),
    ttsVoiceSelect: document.getElementById("ttsVoiceSelect"),
    speedSlider: document.getElementById("speedSlider"),
    speedVal: document.getElementById("speedVal"),
    pitchSlider: document.getElementById("pitchSlider"),
    pitchVal: document.getElementById("pitchVal"),
    generateSpeechBtn: document.getElementById("generateSpeechBtn"),
    ttsSpinner: document.getElementById("ttsSpinner"),
    
    // Audio Player & Visualizer
    audioStudio: document.getElementById("audioStudio"),
    audioPlayer: document.getElementById("audioPlayer"),
    waveformImg: document.getElementById("waveformImg"),
    audioMetaBadge: document.getElementById("audioMetaBadge"),
    
    // Export Buttons
    downloadAudioBtn: document.getElementById("downloadAudioBtn"),
    downloadSrtBtn: document.getElementById("downloadSrtBtn"),
    downloadTxtBtn: document.getElementById("downloadTxtBtn"),
    downloadZipBtn: document.getElementById("downloadZipBtn"),
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    setupDropzone();
});

// Setup All Event Listeners
function setupEventListeners() {
    // Sliders
    elements.speedSlider.addEventListener("input", (e) => {
        elements.speedVal.textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
    });
    elements.pitchSlider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        elements.pitchVal.textContent = val >= 0 ? `+${val}Hz` : `${val}Hz`;
    });

    // Sample Loaders
    elements.samplePdfBtn.addEventListener("click", () => loadSample("pdf"));
    elements.sampleImgBtn.addEventListener("click", () => loadSample("image"));

    // Page Navigation
    elements.prevPageBtn.addEventListener("click", () => changePage(state.currentPage - 1));
    elements.nextPageBtn.addEventListener("click", () => changePage(state.currentPage + 1));

    // CV Preprocessing
    elements.applyCvBtn.addEventListener("click", reapplyPreprocessing);

    // OCR
    elements.extractTextBtn.addEventListener("click", () => runOCR(false));
    elements.extractAllBtn.addEventListener("click", () => runOCR(true));

    // TTS
    elements.generateSpeechBtn.addEventListener("click", generateSpeech);

    // Real-time Text edits
    elements.cleanedTextArea.addEventListener("input", debounce(handleTextEdit, 500));

    // Audio Playback Time Update (Karaoke Sentence Tracker)
    elements.audioPlayer.addEventListener("timeupdate", handleAudioTimeUpdate);

    // Export Buttons
    elements.downloadAudioBtn.addEventListener("click", () => exportAsset("audio"));
    elements.downloadSrtBtn.addEventListener("click", () => exportAsset("srt"));
    elements.downloadTxtBtn.addEventListener("click", () => exportAsset("transcript"));
    elements.downloadZipBtn.addEventListener("click", () => exportAsset("zip"));
}

// Drag & Drop Setup
function setupDropzone() {
    const dz = elements.dropzone;
    ["dragenter", "dragover"].forEach(name => {
        dz.addEventListener(name, (e) => {
            e.preventDefault();
            dz.classList.add("dragover");
        });
    });
    ["dragleave", "drop"].forEach(name => {
        dz.addEventListener(name, (e) => {
            e.preventDefault();
            dz.classList.remove("dragover");
        });
    });
    dz.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFile(files[0]);
    });
    elements.fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) uploadFile(e.target.files[0]);
    });
}

// Upload Handler
async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    elements.uploadStatus.innerHTML = `<span style="color: #818CF8;">⏳ Uploading and processing ${file.name}...</span>`;
    
    try {
        const res = await fetch("/api/upload", { method: "POST", body: formData });
        const data = await res.json();

        if (data.status === "success") {
            handleDocumentLoaded(data);
            elements.uploadStatus.innerHTML = `<span style="color: #34D399;">✓ Loaded: ${data.filename} (${data.total_pages} page${data.total_pages > 1 ? 's' : ''})</span>`;
        } else {
            elements.uploadStatus.innerHTML = `<span style="color: #F87171;">✕ Upload failed: ${data.message}</span>`;
        }
    } catch (err) {
        elements.uploadStatus.innerHTML = `<span style="color: #F87171;">✕ Network error: ${err.message}</span>`;
    }
}

// Load Interactive Sample
async function loadSample(sampleType) {
    elements.uploadStatus.innerHTML = `<span style="color: #818CF8;">⏳ Loading sample ${sampleType}...</span>`;
    try {
        const res = await fetch(`/api/sample/${sampleType}`);
        const data = await res.json();
        if (data.status === "success") {
            handleDocumentLoaded(data);
            elements.uploadStatus.innerHTML = `<span style="color: #34D399;">✓ Loaded Sample: ${data.filename}</span>`;
        }
    } catch (err) {
        elements.uploadStatus.innerHTML = `<span style="color: #F87171;">✕ Error loading sample: ${err.message}</span>`;
    }
}

// Populate State & UI from Loaded Document
function handleDocumentLoaded(data) {
    state.docId = data.doc_id;
    state.filename = data.filename;
    state.isPdf = data.is_pdf;
    state.currentPage = data.current_page || 1;
    state.totalPages = data.total_pages || 1;

    try {
        sessionStorage.setItem("active_doc_id", data.doc_id);
    } catch (e) {}

    // Update images
    elements.originalImg.src = data.original_image_url;
    elements.preprocessedImg.src = data.preprocessed_image_url;

    // Update skew badge
    const angle = data.cv_meta?.skew_angle || 0;
    elements.skewAngleBadge.textContent = `Detected Skew: ${angle > 0 ? '+' : ''}${angle}°`;

    // Show/hide page navigation
    if (state.isPdf && state.totalPages > 1) {
        elements.pageNavigator.style.display = "flex";
        elements.pageIndicator.textContent = `Page ${state.currentPage} of ${state.totalPages}`;
        elements.extractAllBtn.style.display = "inline-flex";
    } else {
        elements.pageNavigator.style.display = "none";
        elements.extractAllBtn.style.display = "none";
    }

    // Scroll to processing section smoothly
    document.getElementById("processingSection").scrollIntoView({ behavior: "smooth" });
}

// Page Navigation
async function changePage(newPage) {
    const docId = state.docId || sessionStorage.getItem("active_doc_id");
    if (newPage < 1 || newPage > state.totalPages || !docId) return;

    try {
        const res = await fetch(`/api/page/${docId}/${newPage}`);
        const data = await res.json();
        if (data.status === "success") {
            state.currentPage = newPage;
            elements.pageIndicator.textContent = `Page ${state.currentPage} of ${state.totalPages}`;
            elements.originalImg.src = data.original_image_url;
            elements.preprocessedImg.src = data.preprocessed_image_url;
            elements.skewAngleBadge.textContent = `Detected Skew: ${data.cv_meta?.skew_angle || 0}°`;
        }
    } catch (err) {
        console.error("Page change error:", err);
    }
}

// Reapply Computer Vision Preprocessing
async function reapplyPreprocessing() {
    const docId = state.docId || sessionStorage.getItem("active_doc_id");

    const payload = {
        doc_id: docId,
        image_data: elements.originalImg.src || elements.preprocessedImg.src,
        do_deskew: elements.deskewToggle.checked,
        do_denoise: elements.denoiseToggle.checked,
        do_enhance: elements.enhanceToggle.checked,
        do_binarize: elements.binarizeToggle.checked,
        binarize_mode: elements.binarizeMode.value,
    };

    elements.applyCvBtn.textContent = "Processing...";
    try {
        const res = await fetch("/api/preprocess", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === "success") {
            elements.preprocessedImg.src = data.preprocessed_image_url;
            elements.skewAngleBadge.textContent = `Detected Skew: ${data.cv_meta?.skew_angle || 0}°`;
        }
    } catch (err) {
        alert("Preprocessing update failed: " + err.message);
    } finally {
        elements.applyCvBtn.textContent = "Apply Preprocessing";
    }
}

// Run OCR & NLP Pipeline
async function runOCR(processAll = false) {
    const docId = state.docId || sessionStorage.getItem("active_doc_id");
    const currentImgSrc = elements.preprocessedImg.src || elements.originalImg.src;

    if (!docId && (!currentImgSrc || currentImgSrc.includes("data:image/svg") || currentImgSrc.length < 50)) {
        alert("Please upload a document or load a sample first!");
        return;
    }

    elements.ocrSpinner.style.display = "inline-block";
    elements.extractTextBtn.disabled = true;

    const payload = {
        doc_id: docId,
        image_data: currentImgSrc,
        lang: elements.ocrLangSelect.value,
        engine: elements.ocrEngineSelect.value,
        process_all_pages: processAll
    };

    try {
        const res = await fetch("/api/ocr", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.status === "success") {
            state.rawText = data.raw_text;
            state.cleanedText = data.cleaned_text;
            state.stats = data.stats;

            elements.cleanedTextArea.value = data.cleaned_text;
            elements.rawTextArea.value = data.raw_text;

            updateStatsDisplay(data.stats);
            renderSentenceList(data.sentences);

            document.getElementById("textSection").scrollIntoView({ behavior: "smooth" });
        } else {
            alert("OCR Extraction Notice: " + data.message);
        }
    } catch (err) {
        alert("OCR Network Error: " + err.message);
    } finally {
        elements.ocrSpinner.style.display = "none";
        elements.extractTextBtn.disabled = false;
    }
}

// Real-time Text Edit Handler
async function handleTextEdit() {
    const text = elements.cleanedTextArea.value;
    state.cleanedText = text;

    try {
        const res = await fetch("/api/clean-text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        if (data.status === "success") {
            state.stats = data.stats;
            updateStatsDisplay(data.stats);
            renderSentenceList(data.sentences);
        }
    } catch (err) {
        console.error("Text stats error:", err);
    }
}

// Render Stats Cards
function updateStatsDisplay(stats) {
    elements.statWords.textContent = stats.word_count || 0;
    elements.statChars.textContent = stats.character_count || 0;
    elements.statSentences.textContent = stats.sentence_count || 0;
    elements.statDuration.textContent = stats.formatted_duration || "00:00";
}

// Render Sentence List for Playback Sync
function renderSentenceList(sentences) {
    elements.sentenceList.innerHTML = "";
    if (!sentences || sentences.length === 0) return;

    sentences.forEach((s, idx) => {
        const bubble = document.createElement("div");
        bubble.className = "sentence-bubble";
        bubble.dataset.idx = idx;
        bubble.innerHTML = `<strong>${idx + 1}.</strong> ${s}`;
        elements.sentenceList.appendChild(bubble);
    });
}

// Generate Neural Speech
async function generateSpeech() {
    const text = elements.cleanedTextArea.value.trim();
    if (!text) {
        alert("Please extract or enter text first!");
        return;
    }

    elements.ttsSpinner.style.display = "inline-block";
    elements.generateSpeechBtn.disabled = true;

    const payload = {
        doc_id: state.docId,
        text: text,
        engine: elements.ttsEngineSelect.value,
        voice: elements.ttsVoiceSelect.value,
        speed: parseFloat(elements.speedSlider.value),
        pitch: parseInt(elements.pitchSlider.value),
        lang: elements.ocrLangSelect.value
    };

    try {
        const res = await fetch("/api/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.status === "success") {
            state.audioDataUri = data.audio_data_uri;
            state.audioFormat = data.audio_format;
            state.subtitleEvents = data.subtitle_events || [];
            state.audioDuration = data.duration_seconds;

            // Load audio into HTML5 player
            elements.audioPlayer.src = data.audio_data_uri;
            elements.audioStudio.style.display = "block";
            elements.waveformImg.src = data.waveform_image_url;
            elements.audioMetaBadge.textContent = `${data.formatted_duration} (${data.duration_seconds}s) • ${data.file_size_kb} KB • Formatted in ${data.synthesis_time_seconds}s`;

            elements.audioStudio.scrollIntoView({ behavior: "smooth" });
            elements.audioPlayer.play().catch(() => {});
        } else {
            alert("TTS Synthesis Failed: " + data.message);
        }
    } catch (err) {
        alert("TTS Network Error: " + err.message);
    } finally {
        elements.ttsSpinner.style.display = "none";
        elements.generateSpeechBtn.disabled = false;
    }
}

// Real-time Sentence Highlighter (Karaoke tracker)
function handleAudioTimeUpdate() {
    const currentTime = elements.audioPlayer.currentTime;
    if (!state.subtitleEvents || state.subtitleEvents.length === 0) return;

    let activeIdx = -1;
    for (let i = 0; i < state.subtitleEvents.length; i++) {
        const ev = state.subtitleEvents[i];
        if (currentTime >= ev.start_sec && currentTime <= ev.end_sec) {
            activeIdx = i;
            break;
        }
    }

    // Highlight corresponding bubble
    const bubbles = elements.sentenceList.querySelectorAll(".sentence-bubble");
    bubbles.forEach((b, idx) => {
        if (idx === activeIdx) {
            b.classList.add("active-speech");
            b.scrollIntoView({ behavior: "smooth", block: "nearest" });
        } else {
            b.classList.remove("active-speech");
        }
    });
}

// Export Asset Helper
function exportAsset(exportType) {
    if (!state.docId) {
        alert("Please upload a document and generate speech first!");
        return;
    }
    window.location.href = `/api/export/${state.docId}/${exportType}`;
}

// Deep Dive Interactive Tab Switcher
function switchDeepDive(tabKey) {
    const tabs = ["tts-model", "mel-spec", "text-proc", "vocoder", "ocr", "audio-out"];
    
    tabs.forEach(t => {
        const btn = document.getElementById(`tab-${t}`);
        const content = document.getElementById(`content-${t}`);
        if (t === tabKey) {
            btn?.classList.add("active");
            content?.classList.remove("hidden");
        } else {
            btn?.classList.remove("active");
            content?.classList.add("hidden");
        }
    });

    if (tabKey === "mel-spec") {
        loadDefaultSpectrogram();
    }

    if (window.lucide) {
        lucide.createIcons();
    }
}

// Demo Audio Samples Cache & Player
const demoAudioCache = {};
async function playDemoSample(sampleId) {
    const demoAudio = document.getElementById("demoAudioPlayer");
    const btn = document.getElementById(`demoBtn${sampleId}`);
    
    const samples = {
        1: {
            text: "Machine learning has revolutionized optical character recognition and neural speech synthesis.",
            voice: "en-US-JennyNeural"
        },
        2: {
            text: "By combining computer vision with neural audio processing, digital assistants convert physical textbooks into immersive audiobooks.",
            voice: "en-US-ChristopherNeural"
        }
    };

    const cfg = samples[sampleId];
    if (!cfg) return;

    if (!demoAudio.paused && demoAudio.dataset.activeSample == sampleId) {
        demoAudio.pause();
        btn.innerHTML = `<i data-lucide="play" class="w-3.5 h-3.5 fill-current"></i><span>Play Audio</span>`;
        if (window.lucide) lucide.createIcons();
        return;
    }

    btn.innerHTML = `<span>⏳ Loading...</span>`;

    try {
        if (!demoAudioCache[sampleId]) {
            const res = await fetch("/api/tts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: cfg.text,
                    voice: cfg.voice,
                    speed: 1.0,
                    pitch: 0
                })
            });
            const data = await res.json();
            if (data.status === "success") {
                demoAudioCache[sampleId] = data.audio_data_uri;
            }
        }

        demoAudio.src = demoAudioCache[sampleId];
        demoAudio.dataset.activeSample = sampleId;
        demoAudio.play();

        btn.innerHTML = `<i data-lucide="pause" class="w-3.5 h-3.5 fill-current"></i><span>Pause</span>`;
        demoAudio.onended = () => {
            btn.innerHTML = `<i data-lucide="play" class="w-3.5 h-3.5 fill-current"></i><span>Play Audio</span>`;
            if (window.lucide) lucide.createIcons();
        };

        if (window.lucide) lucide.createIcons();
    } catch (err) {
        console.error("Demo playback error:", err);
        btn.innerHTML = `<i data-lucide="play" class="w-3.5 h-3.5 fill-current"></i><span>Play Audio</span>`;
        if (window.lucide) lucide.createIcons();
    }
}

// Load Default Spectrogram Graphic
async function loadDefaultSpectrogram() {
    const imgEl = document.getElementById("deepdiveSpectrogramImg");
    if (imgEl && !imgEl.src) {
        try {
            const res = await fetch("/api/spectrogram");
            const data = await res.json();
            if (data.status === "success") {
                imgEl.src = data.spectrogram_image_url;
            }
        } catch (err) {
            console.error("Spectrogram load error:", err);
        }
    }
}

// Preload Spectrogram on Init
document.addEventListener("DOMContentLoaded", () => {
    loadDefaultSpectrogram();
});

// Debounce helper
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
