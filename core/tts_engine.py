"""
Multi-Engine Text-to-Speech (TTS) Synthesizer.
Supports Microsoft Edge Neural HD Voices, gTTS, and pyttsx3 offline synthesis with pitch & rate controls.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import io
import os
import tempfile
import edge_tts


class TTSEngine:
    """Advanced Text-to-Speech Engine with Neural AI Voices and Offline Fallbacks."""

    # Curated selection of top-tier natural neural voices
    VOICE_REGISTRY: Dict[str, Dict[str, Any]] = {
        "en-US-GuyNeural": {
            "name": "Guy (Male, US Natural)",
            "gender": "Male",
            "lang": "English (US)",
            "short_name": "en-US-GuyNeural",
        },
        "en-US-JennyNeural": {
            "name": "Jenny (Female, US Natural)",
            "gender": "Female",
            "lang": "English (US)",
            "short_name": "en-US-JennyNeural",
        },
        "en-US-ChristopherNeural": {
            "name": "Christopher (Male, US Deep / Storyteller)",
            "gender": "Male",
            "lang": "English (US)",
            "short_name": "en-US-ChristopherNeural",
        },
        "en-US-AriaNeural": {
            "name": "Aria (Female, US Clear / Expressive)",
            "gender": "Female",
            "lang": "English (US)",
            "short_name": "en-US-AriaNeural",
        },
        "en-GB-RyanNeural": {
            "name": "Ryan (Male, UK British)",
            "gender": "Male",
            "lang": "English (UK)",
            "short_name": "en-GB-RyanNeural",
        },
        "en-GB-SoniaNeural": {
            "name": "Sonia (Female, UK British)",
            "gender": "Female",
            "lang": "English (UK)",
            "short_name": "en-GB-SoniaNeural",
        },
        "en-IN-PrabhatNeural": {
            "name": "Prabhat (Male, Indian English)",
            "gender": "Male",
            "lang": "English (India)",
            "short_name": "en-IN-PrabhatNeural",
        },
        "en-IN-NeerjaNeural": {
            "name": "Neerja (Female, Indian English)",
            "gender": "Female",
            "lang": "English (India)",
            "short_name": "en-IN-NeerjaNeural",
        },
        "en-AU-WilliamNeural": {
            "name": "William (Male, Australian)",
            "gender": "Male",
            "lang": "English (Australia)",
            "short_name": "en-AU-WilliamNeural",
        },
        "en-AU-NatashaNeural": {
            "name": "Natasha (Female, Australian)",
            "gender": "Female",
            "lang": "English (Australia)",
            "short_name": "en-AU-NatashaNeural",
        },
        "hi-IN-MadhurNeural": {
            "name": "Madhur (Male, Hindi)",
            "gender": "Male",
            "lang": "Hindi (India)",
            "short_name": "hi-IN-MadhurNeural",
        },
        "hi-IN-SwaraNeural": {
            "name": "Swara (Female, Hindi)",
            "gender": "Female",
            "lang": "Hindi (India)",
            "short_name": "hi-IN-SwaraNeural",
        },
        "es-ES-AlvaroNeural": {
            "name": "Alvaro (Male, Spanish)",
            "gender": "Male",
            "lang": "Spanish (Spain)",
            "short_name": "es-ES-AlvaroNeural",
        },
        "es-ES-ElviraNeural": {
            "name": "Elvira (Female, Spanish)",
            "gender": "Female",
            "lang": "Spanish (Spain)",
            "short_name": "es-ES-ElviraNeural",
        },
        "fr-FR-HenriNeural": {
            "name": "Henri (Male, French)",
            "gender": "Male",
            "lang": "French (France)",
            "short_name": "fr-FR-HenriNeural",
        },
        "fr-FR-DeniseNeural": {
            "name": "Denise (Female, French)",
            "gender": "Female",
            "lang": "French (France)",
            "short_name": "fr-FR-DeniseNeural",
        },
        "de-DE-ConradNeural": {
            "name": "Conrad (Male, German)",
            "gender": "Male",
            "lang": "German (Germany)",
            "short_name": "de-DE-ConradNeural",
        },
        "de-DE-KatjaNeural": {
            "name": "Katja (Female, German)",
            "gender": "Female",
            "lang": "German (Germany)",
            "short_name": "de-DE-KatjaNeural",
        },
    }

    def __init__(self, default_voice: str = "en-US-JennyNeural"):
        self.default_voice = default_voice

    @classmethod
    def get_available_voices(cls) -> Dict[str, Dict[str, Any]]:
        """Returns dictionary of all supported neural voices."""
        return cls.VOICE_REGISTRY

    @staticmethod
    def _format_rate(speed_multiplier: float) -> str:
        """Converts speed multiplier (e.g. 1.25) to Edge-TTS rate string (+25%)."""
        pct = int(round((speed_multiplier - 1.0) * 100))
        return f"+{pct}%" if pct >= 0 else f"{pct}%"

    @staticmethod
    def _format_pitch(pitch_hz: int) -> str:
        """Converts pitch offset in Hz to Edge-TTS pitch string."""
        return f"+{pitch_hz}Hz" if pitch_hz >= 0 else f"{pitch_hz}Hz"

    async def _synthesize_edge_async(
        self,
        text: str,
        voice: str,
        rate_str: str,
        pitch_str: str,
        volume_str: str = "+0%"
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """Asynchronous synthesis using edge-tts with boundary event tracking."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str
        )
        audio_chunks = []
        subtitle_events = []

        async for event in communicate.stream():
            if event["type"] == "audio":
                audio_chunks.append(event["data"])
            elif event["type"] == "WordBoundary":
                # Microseconds to seconds
                offset_sec = event["offset"] / 10_000_000.0
                duration_sec = event["duration"] / 10_000_000.0
                subtitle_events.append({
                    "text": event["text"],
                    "start_sec": offset_sec,
                    "end_sec": offset_sec + duration_sec
                })

        audio_bytes = b"".join(audio_chunks)
        return audio_bytes, subtitle_events

    def synthesize_edge(
        self,
        text: str,
        voice: str = "en-US-JennyNeural",
        speed: float = 1.0,
        pitch_hz: int = 0
    ) -> Tuple[bytes, List[Dict[str, Any]], str]:
        """
        Synthesizes text into high-definition neural speech using Edge-TTS.
        Returns: (audio_bytes, subtitle_events, "mp3")
        """
        if not text.strip():
            raise ValueError("Input text is empty.")

        rate_str = self._format_rate(speed)
        pitch_str = self._format_pitch(pitch_hz)

        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                audio_bytes, sub_events = loop.run_until_complete(
                    self._synthesize_edge_async(text, voice, rate_str, pitch_str)
                )
            else:
                audio_bytes, sub_events = loop.run_until_complete(
                    self._synthesize_edge_async(text, voice, rate_str, pitch_str)
                )
            return audio_bytes, sub_events, "mp3"
        except Exception as e:
            print(f"[TTSEngine] Edge-TTS error: {e}. Trying fallback...")
            raise e

    def synthesize_gtts(self, text: str, lang: str = "en", slow: bool = False) -> Tuple[bytes, List[Dict[str, Any]], str]:
        """Google Text-to-Speech (gTTS) cloud fallback."""
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=slow)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read(), [], "mp3"

    def synthesize_pyttsx3(self, text: str, speed_multiplier: float = 1.0) -> Tuple[bytes, List[Dict[str, Any]], str]:
        """Offline pyttsx3 fallback synthesis (WAV output)."""
        import pyttsx3
        engine = pyttsx3.init()
        # Default rate is ~200 WPM
        base_rate = engine.getProperty("rate")
        engine.setProperty("rate", int(base_rate * speed_multiplier))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes, [], "wav"
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def synthesize(
        self,
        text: str,
        engine: str = "edge-tts",
        voice: str = "en-US-JennyNeural",
        speed: float = 1.0,
        pitch_hz: int = 0,
        lang: str = "en"
    ) -> Tuple[bytes, List[Dict[str, Any]], str]:
        """
        Unified synthesis interface with automatic fallback.
        Engines: 'edge-tts', 'gtts', 'pyttsx3'
        """
        if engine == "edge-tts":
            try:
                return self.synthesize_edge(text, voice=voice, speed=speed, pitch_hz=pitch_hz)
            except Exception as e:
                print(f"[TTSEngine] Primary Edge-TTS failed: {e}. Falling back to gTTS...")
                try:
                    return self.synthesize_gtts(text, lang=lang)
                except Exception as e2:
                    print(f"[TTSEngine] gTTS failed: {e2}. Falling back to offline pyttsx3...")
                    return self.synthesize_pyttsx3(text, speed_multiplier=speed)

        elif engine == "gtts":
            try:
                return self.synthesize_gtts(text, lang=lang)
            except Exception:
                return self.synthesize_pyttsx3(text, speed_multiplier=speed)

        elif engine == "pyttsx3":
            return self.synthesize_pyttsx3(text, speed_multiplier=speed)

        else:
            return self.synthesize_edge(text, voice=voice, speed=speed, pitch_hz=pitch_hz)
