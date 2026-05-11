"""
Captcha Generation for Voice Liveness Verification.

This module generates a NUMERIC captcha (digits) that the user reads aloud.
"""
import random
from typing import List, Tuple
from datetime import datetime
import hashlib
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import CAPTCHA_NUMERIC_LENGTH


class CaptchaGenerator:
    """
    Generate numeric captcha strings for voice verification.
    """
    
    def __init__(self, numeric_length: int = CAPTCHA_NUMERIC_LENGTH):
        self.numeric_length = numeric_length
        logger.info(f"Initialized CaptchaGenerator (numeric, length={numeric_length})")
    
    def generate_predefined(self) -> Tuple[str, str]:
        """
        Generate a captcha (numeric).
        
        Returns:
            Tuple of (captcha_text, captcha_id)
        """
        captcha_text = self._generate_numeric_string()
        captcha_id = self._generate_id(captcha_text)
        return captcha_text, captcha_id
    
    def generate_dynamic(self, complexity: str = 'medium') -> Tuple[str, str]:
        """
        Generate a dynamic captcha (numeric).
        
        Args:
            complexity: unused (kept for backward compatibility)
        
        Returns:
            Tuple of (captcha_text, captcha_id)
        """
        captcha_text = self._generate_numeric_string()
        captcha_id = self._generate_id(captcha_text)
        return captcha_text, captcha_id
    
    def _generate_numeric_string(self) -> str:
        """Generate a digit-only captcha string."""
        return ''.join(random.choices('0123456789', k=self.numeric_length))
    
    def _generate_id(self, sentence: str) -> str:
        """Generate a unique ID for the captcha"""
        timestamp = datetime.now().isoformat()
        data = f"{sentence}_{timestamp}_{random.random()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    
    def validate_captcha(self, captcha_id: str, timestamp: datetime, 
                        max_age_seconds: int = 300) -> bool:
        """
        Validate that a captcha is still valid (not expired)
        
        Args:
            captcha_id: The captcha ID to validate
            timestamp: When the captcha was created
            max_age_seconds: Maximum age in seconds (default 5 minutes)
        
        Returns:
            True if captcha is still valid
        """
        age = (datetime.now() - timestamp).total_seconds()
        return age <= max_age_seconds


class CaptchaSession:
    """Manage captcha sessions for verification"""
    
    def __init__(self, expiry_seconds: int = 300):
        self.expiry_seconds = expiry_seconds
        self.sessions = {}  # captcha_id -> {sentence, created_at, attempts}
        self.generator = CaptchaGenerator()
    
    def create_session(self, complexity: str = 'medium') -> dict:
        """
        Create a new captcha session (numeric).
        
        Returns:
            Session info with captcha text and ID
        """
        captcha_text, captcha_id = self.generator.generate_dynamic(complexity)
        
        self.sessions[captcha_id] = {
            'sentence': captcha_text,
            'created_at': datetime.now(),
            'attempts': 0
        }
        
        # Clean up expired sessions
        self._cleanup_expired()
        
        return {
            'captcha_id': captcha_id,
            'sentence': captcha_text,
            'expires_in': self.expiry_seconds
        }
    
    def validate_session(self, captcha_id: str) -> Tuple[bool, str]:
        """
        Validate a captcha session
        
        Returns:
            Tuple of (is_valid, sentence_or_error)
        """
        if captcha_id not in self.sessions:
            return False, "Invalid captcha ID"
        
        session = self.sessions[captcha_id]
        
        # Check expiry
        age = (datetime.now() - session['created_at']).total_seconds()
        if age > self.expiry_seconds:
            del self.sessions[captcha_id]
            return False, "Captcha expired"
        
        # Check attempts
        if session['attempts'] >= 3:
            del self.sessions[captcha_id]
            return False, "Maximum attempts exceeded"
        
        # Increment attempts
        session['attempts'] += 1
        
        return True, session['sentence']
    
    def complete_session(self, captcha_id: str):
        """Mark a session as completed"""
        if captcha_id in self.sessions:
            del self.sessions[captcha_id]
    
    def _cleanup_expired(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired = [
            cid for cid, session in self.sessions.items()
            if (now - session['created_at']).total_seconds() > self.expiry_seconds
        ]
        for cid in expired:
            del self.sessions[cid]

