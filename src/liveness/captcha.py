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
    Generate random captcha sentences for voice verification
    """
    
    def __init__(self, predefined_sentences: List[str] = None,
                 include_numbers: bool = True, 
                 include_colors: bool = True):
        self.predefined_sentences = predefined_sentences or CAPTCHA_SENTENCES
        self.include_numbers = include_numbers
        self.include_colors = include_colors
        
        # Word lists for dynamic generation
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 
                       'pink', 'white', 'black', 'gray']
        
        self.adjectives = ['quick', 'lazy', 'happy', 'bright', 'calm', 'clear',
                          'fresh', 'gentle', 'kind', 'smart', 'warm', 'bold']
        
        self.nouns = ['fox', 'dog', 'cat', 'bird', 'tree', 'river', 'mountain',
                     'cloud', 'star', 'flower', 'book', 'house', 'garden', 'bridge']
        
        self.verbs = ['jumps', 'runs', 'walks', 'flies', 'swims', 'climbs',
                     'reads', 'writes', 'speaks', 'thinks', 'dreams', 'dances']
        
        self.actions = ['over', 'under', 'around', 'through', 'across', 'beside',
                       'near', 'along', 'toward', 'into', 'onto', 'upon']
        
        # Number words
        self.number_words = {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
            5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'
        }
        
        logger.info("Initialized CaptchaGenerator")
    
    def generate_predefined(self) -> Tuple[str, str]:
        """
        Generate a captcha from predefined sentences
        
        Returns:
            Tuple of (sentence, captcha_id)
        """
        sentence = random.choice(self.predefined_sentences)
        captcha_id = self._generate_id(sentence)
        return sentence, captcha_id
    
    def generate_dynamic(self, complexity: str = 'medium') -> Tuple[str, str]:
        """
        Generate a dynamic random sentence
        
        Args:
            complexity: 'simple', 'medium', or 'complex'
        
        Returns:
            Tuple of (sentence, captcha_id)
        """
        if complexity == 'simple':
            sentence = self._generate_simple_sentence()
        elif complexity == 'complex':
            sentence = self._generate_complex_sentence()
        else:
            sentence = self._generate_medium_sentence()
        
        captcha_id = self._generate_id(sentence)
        return sentence, captcha_id
    
    def _generate_simple_sentence(self) -> str:
        """Generate a simple sentence (3-5 words)"""
        templates = [
            f"The {random.choice(self.adjectives)} {random.choice(self.nouns)}",
            f"A {random.choice(self.colors)} {random.choice(self.nouns)}",
            f"Please say {self._random_number_word()}"
        ]
        return random.choice(templates)
    
    def _generate_medium_sentence(self) -> str:
        """Generate a medium complexity sentence (5-8 words)"""
        templates = [
            f"The {random.choice(self.adjectives)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.actions)} the {random.choice(self.nouns)}",
            f"A {random.choice(self.colors)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.actions)} a {random.choice(self.adjectives)} {random.choice(self.nouns)}",
            f"Please verify by saying {self._random_number_word()} {self._random_number_word()} {self._random_number_word()}",
            f"My favorite {random.choice(self.nouns)} is {random.choice(self.adjectives)} and {random.choice(self.colors)}"
        ]
        return random.choice(templates)
    
    def _generate_complex_sentence(self) -> str:
        """Generate a complex sentence (8+ words)"""
        templates = [
            f"The {random.choice(self.adjectives)} {random.choice(self.colors)} {random.choice(self.nouns)} {random.choice(self.verbs)} {random.choice(self.actions)} the {random.choice(self.adjectives)} {random.choice(self.nouns)} in the {random.choice(self.nouns)}",
            f"I verify my identity by saying {self._random_number_word()} {self._random_number_word()} {self._random_number_word()} {self._random_number_word()}",
            f"Please repeat the following words: {random.choice(self.adjectives)}, {random.choice(self.colors)}, {random.choice(self.nouns)}, {random.choice(self.verbs)}"
        ]
        return random.choice(templates)
    
    def _random_number_word(self) -> str:
        """Generate a random number word"""
        return self.number_words[random.randint(0, 9)]
    
    def _generate_id(self, sentence: str) -> str:
        """Generate a unique ID for the captcha"""
        timestamp = datetime.now().isoformat()
        data = f"{sentence}_{timestamp}_{random.random()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def generate_number_sequence(self, length: int = 4) -> Tuple[str, str]:
        """
        Generate a number sequence captcha
        
        Args:
            length: Number of digits
        
        Returns:
            Tuple of (sentence, captcha_id)
        """
        numbers = [random.randint(0, 9) for _ in range(length)]
        words = [self.number_words[n] for n in numbers]
        sentence = f"Please say the numbers: {' '.join(words)}"
        captcha_id = self._generate_id(sentence)
        return sentence, captcha_id
    
    def generate_word_sequence(self, length: int = 4) -> Tuple[str, str]:
        """
        Generate a word sequence captcha
        
        Args:
            length: Number of words
        
        Returns:
            Tuple of (sentence, captcha_id)
        """
        all_words = self.adjectives + self.nouns + self.colors
        words = random.sample(all_words, min(length, len(all_words)))
        sentence = f"Please say these words: {', '.join(words)}"
        captcha_id = self._generate_id(sentence)
        return sentence, captcha_id
    
    def generate_mixed_captcha(self) -> Tuple[str, str]:
        """
        Generate a mixed captcha (words + numbers)
        
        Returns:
            Tuple of (sentence, captcha_id)
        """
        elements = []
        
        # Add some words
        for _ in range(random.randint(2, 3)):
            word_type = random.choice(['adjective', 'color', 'noun'])
            if word_type == 'adjective':
                elements.append(random.choice(self.adjectives))
            elif word_type == 'color':
                elements.append(random.choice(self.colors))
            else:
                elements.append(random.choice(self.nouns))
        
        # Add some numbers
        for _ in range(random.randint(2, 3)):
            elements.append(self.number_words[random.randint(0, 9)])
        
        random.shuffle(elements)
        sentence = f"Read aloud: {' '.join(elements)}"
        captcha_id = self._generate_id(sentence)
        return sentence, captcha_id
    
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
        Create a new captcha session
        
        Returns:
            Session info with captcha text and ID
        """
        # Choose generation method randomly for variety
        method = random.choice(['predefined', 'dynamic', 'number', 'word', 'mixed'])
        
        if method == 'predefined':
            sentence, captcha_id = self.generator.generate_predefined()
        elif method == 'dynamic':
            sentence, captcha_id = self.generator.generate_dynamic(complexity)
        elif method == 'number':
            sentence, captcha_id = self.generator.generate_number_sequence()
        elif method == 'word':
            sentence, captcha_id = self.generator.generate_word_sequence()
        else:
            sentence, captcha_id = self.generator.generate_mixed_captcha()
        
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

