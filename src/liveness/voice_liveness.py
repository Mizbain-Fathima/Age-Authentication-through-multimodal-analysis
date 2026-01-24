"""
Voice Liveness Detection
Verifies that the spoken content matches the captcha text using speech-to-text
"""
import numpy as np
import librosa
from pathlib import Path
from loguru import logger
from typing import Dict, Optional, Tuple
import re
from difflib import SequenceMatcher

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import SAMPLE_RATE, CAPTCHA_MATCH_THRESHOLD


class VoiceLivenessDetector:
    """
    Voice liveness detection through captcha verification
    Uses speech-to-text to verify spoken content matches expected text
    """
    
    def __init__(self, match_threshold=CAPTCHA_MATCH_THRESHOLD, use_whisper=True):
        self.match_threshold = match_threshold
        self.use_whisper = use_whisper
        self.model = None
        self.recognizer = None
        
        self._initialize_stt()
        logger.info(f"Initialized VoiceLivenessDetector with threshold={match_threshold}")
    
    def _initialize_stt(self):
        """Initialize speech-to-text engine"""
        if self.use_whisper:
            try:
                import whisper
                # Use small model for faster inference
                self.model = whisper.load_model("base")
                logger.info("Loaded Whisper base model for STT")
            except ImportError:
                logger.warning("Whisper not available, falling back to SpeechRecognition")
                self.use_whisper = False
        
        if not self.use_whisper:
            try:
                import speech_recognition as sr
                self.recognizer = sr.Recognizer()
                logger.info("Using SpeechRecognition library for STT")
            except ImportError:
                logger.error("No STT library available!")
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Preprocess audio for better recognition"""
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize
        audio_data = audio_data.astype(np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Trim silence
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        
        return audio_data
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
        
        Returns:
            Transcribed text
        """
        audio_processed = self._preprocess_audio(audio_data, sample_rate)
        
        if self.use_whisper and self.model is not None:
            try:
                # Whisper expects float32 audio normalized to [-1, 1]
                result = self.model.transcribe(
                    audio_processed,
                    language='en',
                    task='transcribe',
                    fp16=False
                )
                transcription = result['text'].strip()
                logger.debug(f"Whisper transcription: {transcription}")
                return transcription
            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
                return ""
        
        elif self.recognizer is not None:
            try:
                import speech_recognition as sr
                import io
                import soundfile as sf
                
                # Convert to AudioData format
                audio_bytes = io.BytesIO()
                sf.write(audio_bytes, audio_processed, 16000, format='WAV')
                audio_bytes.seek(0)
                
                with sr.AudioFile(audio_bytes) as source:
                    audio = self.recognizer.record(source)
                
                transcription = self.recognizer.recognize_google(audio)
                logger.debug(f"Google STT transcription: {transcription}")
                return transcription
            except Exception as e:
                logger.error(f"SpeechRecognition error: {e}")
                return ""
        
        return ""
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison - handles number words and digits"""
        # Convert to lowercase
        text = text.lower()
        
        # Number word to digit mapping
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        
        # Split into words and convert number words to digits
        words = text.split()
        normalized_words = []
        for word in words:
            # Remove punctuation from word
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in number_words:
                normalized_words.append(number_words[clean_word])
            else:
                normalized_words.append(clean_word)
        
        # Join and remove all spaces, commas, and repeated tokens
        normalized = ''.join(normalized_words)
        
        # Remove any remaining non-digit/non-letter characters (except digits and letters)
        normalized = re.sub(r'[^\w]', '', normalized)
        
        # Remove repeated consecutive characters (e.g., "91529152" -> "9152")
        if len(normalized) > 0:
            deduplicated = normalized[0]
            for char in normalized[1:]:
                if char != deduplicated[-1]:
                    deduplicated += char
            normalized = deduplicated
        
        return normalized
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        Uses normalized digit strings for number-based captchas
        Uses word overlap for text-based captchas (demo-friendly)
        """
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # If both are digit-only strings, use exact or substring matching
        if norm1.isdigit() and norm2.isdigit():
            # Exact match
            if norm1 == norm2:
                return 1.0
            # Check if one contains the other (handles repeated numbers)
            if norm1 in norm2 or norm2 in norm1:
                return 0.9
            # Calculate edit distance similarity
            seq_ratio = SequenceMatcher(None, norm1, norm2).ratio()
            return seq_ratio
        
        # For text captchas (non-numeric), use word overlap ratio
        # Tokenize and normalize words
        def tokenize_text(text):
            # Lowercase, remove punctuation, split into words
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            words = [w for w in text.split() if w]
            return words
        
        words1 = set(tokenize_text(text1))
        words2 = set(tokenize_text(text2))
        
        if not words1 or not words2:
            # Fallback to character-level similarity
            return SequenceMatcher(None, norm1, norm2).ratio()
        
        # Calculate word overlap ratio
        common_words = len(words1 & words2)
        expected_words = len(words1)
        
        if expected_words > 0:
            overlap_ratio = common_words / expected_words
            # Use word overlap for text captchas (more lenient)
            return overlap_ratio
        
        # Fallback to character-level similarity
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def verify_captcha(self, audio_data: np.ndarray, expected_text: str,
                       sample_rate: int = SAMPLE_RATE) -> Dict:
        """
        Verify that spoken audio matches expected captcha text
        
        Args:
            audio_data: Audio waveform
            expected_text: The captcha text that should be spoken
            sample_rate: Audio sample rate
        
        Returns:
            Verification result dictionary
        """
        # Transcribe audio
        transcribed_text = self.transcribe_audio(audio_data, sample_rate)
        
        if not transcribed_text:
            return {
                'verified': False,
                'confidence': 0.0,
                'transcribed_text': '',
                'expected_text': expected_text,
                'similarity': 0.0,
                'reason': 'Failed to transcribe audio'
            }
        
        # Calculate similarity
        similarity = self.calculate_similarity(transcribed_text, expected_text)
        
        # Determine verification result with relaxed rules for text captchas
        # Check if expected text is numeric (digit-only after normalization)
        norm_expected = self._normalize_text(expected_text)
        is_numeric_captcha = norm_expected.isdigit()
        
        if is_numeric_captcha:
            # Numeric captcha: use original threshold
            verified = similarity >= self.match_threshold
            if verified:
                logger.info("Captcha verified via numeric match")
        else:
            # Text captcha: use word overlap (>= 0.6 for demo-friendly)
            verified = similarity >= 0.6
            if verified:
                logger.info(f"Captcha verified via text overlap (overlap: {similarity:.1%})")
        
        # Generate detailed reason
        if verified:
            if is_numeric_captcha:
                reason = f"Speech matches numeric captcha (similarity: {similarity:.1%})"
            else:
                reason = f"Speech matches text captcha (word overlap: {similarity:.1%})"
        elif similarity > 0.3:
            reason = f"Partial match detected (similarity: {similarity:.1%})"
        else:
            reason = "Speech does not match expected captcha"
        
        return {
            'verified': verified,
            'confidence': similarity,
            'transcription': transcribed_text,  # Use 'transcription' for consistency
            'transcribed_text': transcribed_text,  # Keep both for backward compatibility
            'expected_text': expected_text,
            'match_score': similarity,  # Add match_score alias
            'similarity': similarity,
            'reason': reason,
            'is_numeric_captcha': is_numeric_captcha
        }
    
    def analyze_voice_characteristics(self, audio_data: np.ndarray,
                                       sample_rate: int = SAMPLE_RATE) -> Dict:
        """
        Analyze voice characteristics for additional liveness signals
        
        Args:
            audio_data: Audio waveform
            sample_rate: Sample rate
        
        Returns:
            Voice analysis results
        """
        audio_processed = self._preprocess_audio(audio_data, sample_rate)
        
        # Basic voice activity detection
        energy = np.sqrt(np.mean(audio_processed ** 2))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_processed))
        
        # Spectral centroid (brightness)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(
            y=audio_processed, sr=16000
        ))
        
        # Fundamental frequency estimation
        pitches, magnitudes = librosa.piptrack(y=audio_processed, sr=16000)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            if pitches[index, t] > 0:
                pitch_values.append(pitches[index, t])
        
        avg_pitch = np.mean(pitch_values) if pitch_values else 0
        pitch_variance = np.var(pitch_values) if pitch_values else 0
        
        # Speech rate estimation (syllables per second approximation)
        onset_env = librosa.onset.onset_strength(y=audio_processed, sr=16000)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=16000)[0]
        
        # Naturalness score
        # Real speech has certain characteristics:
        # - Pitch variation
        # - Natural speech rate
        # - Spectral properties
        naturalness_score = 0.0
        
        # Check for pitch variation (monotone = suspicious)
        if pitch_variance > 50:
            naturalness_score += 0.3
        elif pitch_variance > 10:
            naturalness_score += 0.15
        
        # Check for reasonable speech rate
        if 80 < tempo < 200:
            naturalness_score += 0.3
        
        # Check energy levels
        if 0.01 < energy < 0.5:
            naturalness_score += 0.2
        
        # Check spectral characteristics
        if 500 < spectral_centroid < 4000:
            naturalness_score += 0.2
        
        return {
            'energy': float(energy),
            'zero_crossing_rate': float(zcr),
            'spectral_centroid': float(spectral_centroid),
            'average_pitch': float(avg_pitch),
            'pitch_variance': float(pitch_variance),
            'estimated_tempo': float(tempo),
            'naturalness_score': naturalness_score,
            'duration': len(audio_processed) / 16000
        }


class AudioQualityChecker:
    """Check audio quality for reliable processing"""
    
    def __init__(self, min_duration=1.0, max_duration=30.0, 
                 min_energy=0.001, max_clipping_ratio=0.1):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_energy = min_energy
        self.max_clipping_ratio = max_clipping_ratio
    
    def check_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Check audio quality
        
        Returns:
            Quality check results with pass/fail and details
        """
        duration = len(audio_data) / sample_rate
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Check for clipping
        max_amplitude = np.max(np.abs(audio_data))
        clipped_samples = np.sum(np.abs(audio_data) > 0.99 * max_amplitude)
        clipping_ratio = clipped_samples / len(audio_data)
        
        # Check for silence
        is_silent = energy < self.min_energy
        
        issues = []
        
        if duration < self.min_duration:
            issues.append(f"Audio too short ({duration:.1f}s < {self.min_duration}s)")
        
        if duration > self.max_duration:
            issues.append(f"Audio too long ({duration:.1f}s > {self.max_duration}s)")
        
        if is_silent:
            issues.append("Audio appears to be silent")
        
        if clipping_ratio > self.max_clipping_ratio:
            issues.append(f"Audio is clipped ({clipping_ratio:.1%} samples)")
        
        passed = len(issues) == 0
        
        return {
            'passed': passed,
            'duration': duration,
            'energy': energy,
            'clipping_ratio': clipping_ratio,
            'is_silent': is_silent,
            'issues': issues
        }

