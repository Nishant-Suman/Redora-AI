"""
Advanced NLP Text Cleaning, Normalization, and Chunking Engine.
Repairs OCR hyphenation, strips headers/footers, cleans punctuation, and generates sentence chunks.
"""

from typing import List, Dict, Any, Tuple
import re


class TextCleaner:
    """Intelligent text cleaner designed specifically for OCR outputs and TTS ingestion."""

    # Common English ligatures to normalize
    LIGATURES = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "æ": "ae",
        "œ": "oe",
        "—": " - ",
        "–": " - ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
    }

    def __init__(self, avg_words_per_minute: int = 150):
        self.wpm = avg_words_per_minute

    def fix_ligatures_and_symbols(self, text: str) -> str:
        """Replaces unusual unicode ligatures and smart quotes with standard characters."""
        for orig, replacement in self.LIGATURES.items():
            text = text.replace(orig, replacement)
        return text

    def fix_line_break_hyphenations(self, text: str) -> str:
        """
        Reconstructs words broken across lines with hyphens.
        Example: 'recon-\nstructs' -> 'reconstructs'
        """
        # Matches lowercase letter + hyphen + newline + lowercase letter
        pattern = r"([a-zA-Z]+)-\s*[\r\n]+\s*([a-zA-Z]+)"
        return re.sub(pattern, r"\1\2", text)

    def remove_headers_footers(self, text: str) -> str:
        """Removes common header/footer patterns and page numbers."""
        lines = text.splitlines()
        cleaned_lines = []

        page_num_patterns = [
            r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
            r"^\s*\d+\s*/\s*\d+\s*$",
            r"^\s*-\s*\d+\s*-\s*$",
            r"^\s*\d+\s*$",
        ]

        for line in lines:
            stripped = line.strip().lower()
            if not stripped:
                cleaned_lines.append("")
                continue

            # Check page number patterns
            is_page_num = any(re.match(p, stripped, re.IGNORECASE) for p in page_num_patterns)
            if is_page_num:
                continue

            # Check common header/footer artifacts
            if re.match(r"^[-=_~*]{4,}$", stripped):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def normalize_whitespace_and_paragraphs(self, text: str) -> str:
        """
        Normalizes intra-paragraph line breaks into spaces while preserving true paragraph breaks.
        """
        # Replace multiple spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n+", text)
        cleaned_paragraphs = []

        for p in paragraphs:
            # Within a single paragraph, replace single newlines with spaces
            p_clean = " ".join(p.splitlines()).strip()
            if p_clean:
                cleaned_paragraphs.append(p_clean)

        return "\n\n".join(cleaned_paragraphs)

    def clean_text(
        self,
        raw_text: str,
        fix_hyphens: bool = True,
        remove_page_nums: bool = True,
        normalize_spacing: bool = True,
    ) -> str:
        """
        Executes comprehensive text cleaning pipeline.
        """
        if not raw_text or not raw_text.strip():
            return ""

        text = self.fix_ligatures_and_symbols(raw_text)

        if fix_hyphens:
            text = self.fix_line_break_hyphenations(text)

        if remove_page_nums:
            text = self.remove_headers_footers(text)

        if normalize_spacing:
            text = self.normalize_whitespace_and_paragraphs(text)

        # Final cleanup: normalize ellipses, trailing spaces
        text = re.sub(r"\s+([.,;:?!])", r"\1", text)
        text = re.sub(r"([.,;:?!])([a-zA-Z])", r"\1 \2", text)
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into discrete, natural sentences using regex punctuation boundaries.
        Handles abbreviations (e.g., Dr., Prof., e.g., i.e., etc.) without premature splitting.
        """
        if not text.strip():
            return []

        # Temporarily protect common abbreviations
        protected = text
        abbreviations = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "vs.", "etc.", "e.g.", "i.e.", "Inc.", "Ltd.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]
        for i, abbr in enumerate(abbreviations):
            protected = protected.replace(abbr, f"__ABBR_{i}__")

        # Split on sentence terminals followed by whitespace or quotes
        sentence_end = re.compile(r"([.!?]+[\"']?\s+)")
        tokens = sentence_end.split(protected)

        sentences = []
        current = ""
        for i, token in enumerate(tokens):
            current += token
            if i % 2 == 1 or i == len(tokens) - 1:
                # Restore abbreviations
                for j, abbr in enumerate(abbreviations):
                    current = current.replace(f"__ABBR_{j}__", abbr)
                s = current.strip()
                if s:
                    sentences.append(s)
                current = ""

        return sentences if sentences else [text.strip()]

    def calculate_statistics(self, text: str) -> Dict[str, Any]:
        """Calculates rich metrics including word count, reading time, and reading level."""
        clean = text.strip()
        if not clean:
            return {
                "character_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "estimated_speech_seconds": 0,
                "formatted_duration": "00:00",
            }

        words = clean.split()
        word_count = len(words)
        char_count = len(clean)
        sentences = self.split_into_sentences(clean)
        sentence_count = len(sentences)
        paragraphs = [p for p in clean.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        # Estimate speech duration
        duration_minutes = word_count / self.wpm
        total_seconds = int(duration_minutes * 60)
        mins, secs = divmod(total_seconds, 60)
        formatted_duration = f"{mins:02d}:{secs:02d}"

        return {
            "character_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "estimated_speech_seconds": total_seconds,
            "formatted_duration": formatted_duration,
        }

    def generate_srt_subtitles(self, sentence_timings: List[Dict[str, Any]]) -> str:
        """
        Generates standard SRT subtitle file string from sentence timing list.
        sentence_timings: [{"start_sec": float, "end_sec": float, "text": str}, ...]
        """
        srt_lines = []

        def format_timestamp(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

        for idx, item in enumerate(sentence_timings, 1):
            start_str = format_timestamp(item["start_sec"])
            end_str = format_timestamp(item["end_sec"])
            srt_lines.append(f"{idx}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(item["text"])
            srt_lines.append("")

        return "\n".join(srt_lines)
