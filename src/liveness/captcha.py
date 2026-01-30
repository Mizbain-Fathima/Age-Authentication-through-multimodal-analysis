"""
Captcha Generation for Voice Liveness Verification
Generates random text sentences for users to read aloud
"""
import random
import string
from typing import List, Tuple
from datetime import datetime
import hashlib
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import CAPTCHA_SENTENCES


class CaptchaGenerator:
    """
    Generate text-based captcha sentences for voice verification
    Only produces short sentences (5-7 words) with simple phonetics
    """
    
    def __init__(self, predefined_sentences: List[str] = None):
        self.predefined_sentences = predefined_sentences or CAPTCHA_SENTENCES
        
        # Word lists for dynamic generation (simple, common words)
        self.adjectives = ['clear', 'fresh', 'safe', 'bright', 'calm', 'quick',
                          'gentle', 'kind', 'smart', 'warm', 'bold', 'strong']
        
        self.nouns = ['speech', 'system', 'flower', 'cat', 'dog', 'bird',
                     'tree', 'river', 'cloud', 'star', 'book', 'house']
        
        self.verbs = ['runs', 'walks', 'flies', 'speaks', 'reads', 'writes',
                     'enables', 'requires', 'provides', 'creates', 'shows']
        
        self.prepositions = ['under', 'over', 'through', 'across', 'beside',
                            'near', 'along', 'around', 'into', 'onto']
        
        logger.info("Initialized CaptchaGenerator (text-only)")
    
    def generate_predefined(self) -> Tuple[str, str]:
        """
        Generate a captcha from predefined sentences
        
        Returns:
            Tuple of (sentence, captcha_id)
        """
        sentence = random.choice(self.predefined_sentences)
        # Ensure lowercase and clean
        sentence = sentence.lower().strip()
        captcha_id = self._generate_id(sentence)
        return sentence, captcha_id
    
    def generate_dynamic(self, complexity: str = 'medium') -> Tuple[str, str]:
        """
        Generate a dynamic text sentence (5-7 words)
        
        Args:
            complexity: 'medium' (only option for text captchas)
        
        Returns:
            Tuple of (sentence, captcha_id)
        """
        sentence = self._generate_text_sentence()
        captcha_id = self._generate_id(sentence)
        return sentence, captcha_id
    
    def _generate_text_sentence(self) -> str:
        """Generate a text sentence (5-7 words, simple phonetics)"""
        # Generate 5-7 word sentences with simple structure
        templates = [
            # 5 words
            f"{random.choice(self.adjectives)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.prepositions)} the {random.choice(self.nouns)}",
            f"the {random.choice(self.adjectives)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.prepositions)} {random.choice(self.nouns)}",
            # 6 words
            f"{random.choice(self.adjectives)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.prepositions)} the {random.choice(self.adjectives)} {random.choice(self.nouns)}",
            f"the {random.choice(self.adjectives)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.prepositions)} a {random.choice(self.adjectives)} {random.choice(self.nouns)}",
            # 7 words
            f"{random.choice(self.adjectives)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.prepositions)} the {random.choice(self.adjectives)} {random.choice(self.nouns)} {random.choice(self.verbs)}",
            f"the {random.choice(self.adjectives)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.prepositions)} a {random.choice(self.adjectives)} {random.choice(self.nouns)}",
        ]
        sentence = random.choice(templates)
        # Ensure lowercase and clean
        return sentence.lower().strip()
    
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
        Create a new captcha session (text-only)
        
        Returns:
            Session info with captcha text and ID
        """
        # Choose between predefined and dynamic (text-only)
        method = random.choice(['predefined', 'dynamic'])
        
        if method == 'predefined':
            sentence, captcha_id = self.generator.generate_predefined()
        else:
            sentence, captcha_id = self.generator.generate_dynamic(complexity)
        
        # Ensure sentence is lowercase and clean
        sentence = sentence.lower().strip()
        
        self.sessions[captcha_id] = {
            'sentence': sentence,
            'created_at': datetime.now(),
            'attempts': 0
        }
        
        # Clean up expired sessions
        self._cleanup_expired()
        
        return {
            'captcha_id': captcha_id,
            'sentence': sentence,
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

