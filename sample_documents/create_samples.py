import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf


def create_sample_scanned_image(output_path: str = "sample_documents/sample_scanned_page.png"):
    """Generates a realistic scanned document image with skewed text and subtle paper grain."""
    w, h = 1000, 1300
    img = np.ones((h, w, 3), dtype=np.uint8) * 245  # Off-white background

    # Convert to PIL for crisp typography
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Title
    draw.text((80, 80), "Artificial Intelligence & Voice Synthesis", fill=(20, 20, 30))
    draw.line((80, 115, 920, 115), fill=(100, 100, 120), width=2)

    # Paragraph text
    lines = [
        "Machine learning has revolutionized optical character recognition and neural speech synthesis.",
        "Modern deep neural networks convert raw document pixels directly into semantic character embeddings,",
        "which can then be normalized, dehyphenated, and synthesized into natural human-like speech.",
        "",
        "In this automated pipeline, document preprocessing plays a pivotal role. Deskewing algorithms",
        "estimate the document angle and rotate the image back to its upright alignment. Denoising filters",
        "and adaptive thresholding eliminate unwanted background scanner noise and shadow artifacts.",
        "",
        "Once extracted, the text is processed by a neural acoustic model to generate realistic speech audio.",
        "Users can adjust playback speed, pitch, and voice characteristics across multiple languages seamlessly.",
        "",
        "Key Benefits of Voice AI:",
        "• High accessibility for visually impaired readers and multitaskers.",
        "• Fast audio generation with ultra-low latency.",
        "• Multi-language and multi-dialect voice variety.",
    ]

    y = 150
    for line in lines:
        if line.startswith("•") or line.startswith("Key"):
            draw.text((80, y), line, fill=(30, 30, 40))
        else:
            draw.text((80, y), line, fill=(40, 40, 50))
        y += 40

    draw.text((450, 1220), "- Page 1 -", fill=(120, 120, 130))

    # Add light noise / scanner grain
    np_img = np.array(pil_img)
    noise = np.random.normal(0, 5, np_img.shape).astype(np.int16)
    noisy_img = np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add artificial skew (e.g. 2.5 degrees)
    center = (w // 2, h // 2)
    angle = 2.5
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    skewed = cv2.warpAffine(noisy_img, rot_mat, (w, h), borderValue=(255, 255, 255))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, skewed)
    print(f"Created sample scanned image at: {output_path}")


def create_sample_pdf(output_path: str = "sample_documents/sample_document.pdf"):
    """Generates a sample multi-page PDF."""
    doc = pymupdf.open()

    # Page 1
    page1 = doc.new_page(width=595, height=842)  # A4
    text1 = """Machine Learning & Speech Technology Overview

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

Text-to-Speech (TTS) models have evolved dramatically from early formant synthesizers to deep neural architectures like Tacotron, FastSpeech, and VITS. Neural vocoders like HiFi-GAN convert mel-spectrograms into high-fidelity audio waveforms.

Key capabilities of modern voice pipelines:
1. Optical Character Recognition (OCR) for scanned documents.
2. Intelligent NLP cleaning and line-break dehyphenation.
3. Neural text-to-speech with natural human prosody and emotion."""
    page1.insert_text((50, 70), text1, fontsize=12)

    # Page 2
    page2 = doc.new_page(width=595, height=842)
    text2 = """Audio Processing and Future Innovations

The next frontier of speech synthesis lies in zero-shot voice cloning, expressive speech control, and real-time streaming translation.

By combining computer vision with neural audio processing, digital assistants can convert physical textbooks, research papers, and legal documents into immersive audiobooks on the fly.

Features:
• Cross-platform compatibility.
• Local offline and cloud-accelerated TTS.
• Synchronized subtitle generation for immersive reading."""
    page2.insert_text((50, 70), text2, fontsize=12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"Created sample PDF at: {output_path}")


if __name__ == "__main__":
    create_sample_scanned_image()
    create_sample_pdf()
