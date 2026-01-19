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
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        Uses multiple metrics for robust comparison
        """
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Sequence matching ratio
        seq_ratio = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Word overlap (Jaccard similarity)
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Word order similarity
        words1_list = norm1.split()
        words2_list = norm2.split()
        
        # Find common words in order
        order_score = 0
        if words1_list and words2_list:
            i, j = 0, 0
            matches = 0
            while i < len(words1_list) and j < len(words2_list):
                if words1_list[i] == words2_list[j]:
                    matches += 1
                    i += 1
                    j += 1
                elif words1_list[i] in words2_list[j:]:
                    j += 1
                else:
                    i += 1
            order_score = matches / max(len(words1_list), len(words2_list))
        
        # Weighted combination
        similarity = 0.4 * seq_ratio + 0.35 * jaccard + 0.25 * order_score
        
        return similarity
    
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
        
        # Determine verification result
        verified = similarity >= self.match_threshold
        
        # Generate detailed reason
        if verified:
            reason = f"Speech matches captcha (similarity: {similarity:.1%})"
        elif similarity > 0.3:
            reason = f"Partial match detected (similarity: {similarity:.1%})"
        else:
            reason = "Speech does not match expected captcha"
        
        return {
            'verified': verified,
            'confidence': similarity,
            'transcribed_text': transcribed_text,
            'expected_text': expected_text,
            'similarity': similarity,
            'reason': reason
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

