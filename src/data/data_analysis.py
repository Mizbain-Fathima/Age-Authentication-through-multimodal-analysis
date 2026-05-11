"""
Data Analysis Module for UTKFace and Mozilla Common Voice datasets
Analyzes age distribution, data imbalance, and prepares data for training
"""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    IMAGE_DATA_DIR,
    AUDIO_DATA_DIR,
    KIDS_AUDIO_DATA_DIR,
    AUDIO_AGE_MAP,
    AGE_GROUPS,
    KIDS_AGE_RANGE,
    MODELS_DIR,
    FUSION_MODELS_DIR,
    LOGS_DIR,
)


class DataAnalyzer:
    """Comprehensive data analyzer for both face and audio datasets"""
    
    def __init__(self):
        self.image_data_dir = IMAGE_DATA_DIR
        self.audio_data_dir = AUDIO_DATA_DIR
        self.kids_audio_data_dir = KIDS_AUDIO_DATA_DIR
        self.face_data = None
        self.audio_data = None
        self.kids_audio_data = None
        
    def analyze_utkface(self):
        """
        Analyze UTKFace dataset
        Filename format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg
        """
        logger.info("Analyzing UTKFace dataset...")
        
        image_files = list(self.image_data_dir.glob("*.jpg"))
        logger.info(f"Found {len(image_files)} image files")
        
        data = []
        parse_errors = 0
        
        for img_path in image_files:
            filename = img_path.name
            parts = filename.split('_')
            
            if len(parts) >= 4:
                try:
                    age = int(parts[0])
                    gender = int(parts[1])  # 0=male, 1=female
                    race = int(parts[2])     # 0-4
                    
                    # Validate age range
                    if 0 <= age <= 116:
                        data.append({
                            'filepath': str(img_path),
                            'filename': filename,
                            'age': age,
                            'gender': 'male' if gender == 0 else 'female',
                            'race': race,
                            'age_group': self._get_age_group(age),
                            'is_adult': age >= 18
                        })
                    else:
                        parse_errors += 1
                except (ValueError, IndexError):
                    parse_errors += 1
            else:
                parse_errors += 1
        
        self.face_data = pd.DataFrame(data)
        
        logger.info(f"Successfully parsed {len(self.face_data)} images")
        logger.info(f"Parse errors: {parse_errors}")
        
        return self.face_data
    
    def analyze_common_voice(self):
        """
        Analyze Mozilla Common Voice dataset
        Uses train, dev, and test CSV files
        """
        logger.info("Analyzing Mozilla Common Voice dataset...")
        
        csv_files = [
            'cv-other-train.csv',
            'cv-other-dev.csv',
            'cv-other-test.csv'
        ]
        
        all_data = []
        
        for csv_file in csv_files:
            csv_path = self.audio_data_dir / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['split'] = csv_file.replace('cv-other-', '').replace('.csv', '')
                all_data.append(df)
                logger.info(f"Loaded {len(df)} samples from {csv_file}")
        
        if all_data:
            self.audio_data = pd.concat(all_data, ignore_index=True)
            
            # Filter samples with age information
            audio_with_age = self.audio_data[self.audio_data['age'].notna() & 
                                              (self.audio_data['age'] != '')]
            
            logger.info(f"Total audio samples: {len(self.audio_data)}")
            logger.info(f"Samples with age info: {len(audio_with_age)}")
            
            # Add numeric age
            audio_with_age = audio_with_age.copy()
            audio_with_age['numeric_age'] = audio_with_age['age'].map(AUDIO_AGE_MAP)
            audio_with_age['age_group'] = audio_with_age['numeric_age'].apply(self._get_age_group)
            audio_with_age['is_adult'] = audio_with_age['numeric_age'] >= 18
            
            # Check if audio files exist (handle nested directory structure)
            def find_audio_file(filename):
                # Try direct path first
                direct_path = self.audio_data_dir / filename
                if direct_path.exists():
                    return True
                # Handle nested directory structure (e.g., cv-other-train/cv-other-train/sample.mp3)
                parts = filename.replace('\\', '/').split('/')
                if len(parts) >= 2:
                    nested_path = self.audio_data_dir / parts[0] / filename
                    if nested_path.exists():
                        return True
                return False
            
            audio_with_age['file_exists'] = audio_with_age['filename'].apply(find_audio_file)
            
            existing_files = audio_with_age[audio_with_age['file_exists']]
            logger.info(f"Samples with existing audio files: {len(existing_files)}")
            
            self.audio_data = existing_files
            
        return self.audio_data
    
    def analyze_kids_audio(self):
        """
        Analyze Kids Audio dataset
        File format: {Gender}{SpeakerID}_{SessionID}_{UtteranceID}.wav
        Example: F10_01_01.wav = Female speaker 10, session 01, utterance 01
        """
        logger.info("Analyzing Kids Audio dataset...")
        
        if not self.kids_audio_data_dir.exists():
            logger.warning(f"Kids audio directory not found: {self.kids_audio_data_dir}")
            return pd.DataFrame()
        
        audio_files = list(self.kids_audio_data_dir.glob("*.wav"))
        logger.info(f"Found {len(audio_files)} kids audio files")
        
        data = []
        parse_errors = 0
        
        # Extract unique speakers
        speakers = {}
        for audio_path in audio_files:
            filename = audio_path.name
            # Parse: {Gender}{SpeakerID}_{SessionID}_{UtteranceID}.wav
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 3:
                speaker_code = parts[0]  # e.g., "F10" or "M25"
                if speaker_code not in speakers:
                    speakers[speaker_code] = {
                        'gender': 'female' if speaker_code[0] == 'F' else 'male',
                        'speaker_id': speaker_code
                    }
        
        # Assign ages to speakers (uniformly distributed across KIDS_AGE_RANGE)
        unique_speakers = sorted(speakers.keys())
        num_speakers = len(unique_speakers)
        age_range = KIDS_AGE_RANGE[1] - KIDS_AGE_RANGE[0] + 1
        
        for i, speaker in enumerate(unique_speakers):
            # Distribute ages evenly across speakers
            age = KIDS_AGE_RANGE[0] + (i * age_range // num_speakers)
            speakers[speaker]['age'] = min(age, KIDS_AGE_RANGE[1])
        
        logger.info(f"Identified {num_speakers} unique kid speakers")
        
        # Build dataset
        for audio_path in audio_files:
            filename = audio_path.name
            parts = filename.replace('.wav', '').split('_')
            
            if len(parts) >= 3:
                try:
                    speaker_code = parts[0]
                    speaker_info = speakers.get(speaker_code, {})
                    age = speaker_info.get('age', 8)  # default to 8 if not found
                    
                    data.append({
                        'filepath': str(audio_path),
                        'filename': filename,
                        'numeric_age': age,
                        'age': 'child',  # category
                        'gender': speaker_info.get('gender', 'unknown'),
                        'speaker_id': speaker_code,
                        'age_group': 'child',
                        'is_adult': False,
                        'text': '',  # No transcript for kids data
                        'source': 'kids_audio'
                    })
                except (ValueError, IndexError):
                    parse_errors += 1
            else:
                parse_errors += 1
        
        self.kids_audio_data = pd.DataFrame(data)
        
        logger.info(f"Successfully parsed {len(self.kids_audio_data)} kids audio files")
        logger.info(f"Parse errors: {parse_errors}")
        
        # Print age distribution
        if len(self.kids_audio_data) > 0:
            age_dist = self.kids_audio_data.groupby('numeric_age').size()
            logger.info(f"Kids age distribution: {dict(age_dist)}")
        
        return self.kids_audio_data
    
    def _get_age_group(self, age):
        """Map numeric age to age group"""
        for group, (min_age, max_age) in AGE_GROUPS.items():
            if min_age <= age <= max_age:
                return group
        return 'adult'  # default
    
    def print_face_statistics(self):
        """Print comprehensive statistics for face dataset"""
        if self.face_data is None:
            self.analyze_utkface()
            
        df = self.face_data
        
        print("\n" + "="*60)
        print("UTKFace Dataset Statistics")
        print("="*60)
        
        print(f"\nTotal samples: {len(df)}")
        print(f"\nAge Statistics:")
        print(f"  Min age: {df['age'].min()}")
        print(f"  Max age: {df['age'].max()}")
        print(f"  Mean age: {df['age'].mean():.2f}")
        print(f"  Median age: {df['age'].median()}")
        print(f"  Std dev: {df['age'].std():.2f}")
        
        print(f"\nAge Group Distribution:")
        age_group_counts = df['age_group'].value_counts()
        for group, count in age_group_counts.items():
            pct = count / len(df) * 100
            print(f"  {group}: {count} ({pct:.1f}%)")
        
        print(f"\nGender Distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = count / len(df) * 100
            print(f"  {gender}: {count} ({pct:.1f}%)")
        
        print(f"\nAdult (18+) Distribution:")
        adult_counts = df['is_adult'].value_counts()
        adults = adult_counts.get(True, 0)
        minors = adult_counts.get(False, 0)
        print(f"  Adults (18+): {adults} ({adults/len(df)*100:.1f}%)")
        print(f"  Minors (<18): {minors} ({minors/len(df)*100:.1f}%)")
        
        # Age distribution by decade
        print(f"\nAge Distribution by Decade:")
        df['decade'] = (df['age'] // 10) * 10
        decade_counts = df['decade'].value_counts().sort_index()
        for decade, count in decade_counts.items():
            pct = count / len(df) * 100
            print(f"  {decade}-{decade+9}: {count} ({pct:.1f}%)")
        
        # Imbalance ratio
        print(f"\nClass Imbalance Analysis:")
        max_class = age_group_counts.max()
        min_class = age_group_counts.min()
        imbalance_ratio = max_class / min_class
        print(f"  Max class samples: {max_class}")
        print(f"  Min class samples: {min_class}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        
        return df
    
    def print_audio_statistics(self):
        """Print comprehensive statistics for audio dataset"""
        if self.audio_data is None:
            self.analyze_common_voice()
            
        df = self.audio_data
        
        print("\n" + "="*60)
        print("Mozilla Common Voice Dataset Statistics")
        print("="*60)
        
        print(f"\nTotal samples with age info: {len(df)}")
        
        print(f"\nAge Category Distribution:")
        age_counts = df['age'].value_counts()
        for age_cat, count in age_counts.items():
            pct = count / len(df) * 100
            print(f"  {age_cat}: {count} ({pct:.1f}%)")
        
        print(f"\nNumeric Age Statistics:")
        print(f"  Min: {df['numeric_age'].min()}")
        print(f"  Max: {df['numeric_age'].max()}")
        print(f"  Mean: {df['numeric_age'].mean():.2f}")
        print(f"  Median: {df['numeric_age'].median()}")
        
        print(f"\nAge Group Distribution:")
        age_group_counts = df['age_group'].value_counts()
        for group, count in age_group_counts.items():
            pct = count / len(df) * 100
            print(f"  {group}: {count} ({pct:.1f}%)")
        
        print(f"\nGender Distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            if pd.notna(gender) and gender != '':
                pct = count / len(df) * 100
                print(f"  {gender}: {count} ({pct:.1f}%)")
        
        print(f"\nAdult (18+) Distribution:")
        adult_counts = df['is_adult'].value_counts()
        adults = adult_counts.get(True, 0)
        minors = adult_counts.get(False, 0)
        print(f"  Adults (18+): {adults} ({adults/len(df)*100:.1f}%)")
        print(f"  Minors (<18): {minors} ({minors/len(df)*100:.1f}%)")
        
        print(f"\nSplit Distribution:")
        split_counts = df['split'].value_counts()
        for split, count in split_counts.items():
            pct = count / len(df) * 100
            print(f"  {split}: {count} ({pct:.1f}%)")
        
        # Imbalance analysis
        print(f"\nClass Imbalance Analysis:")
        max_class = age_group_counts.max()
        min_class = age_group_counts.min()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        print(f"  Max class samples: {max_class}")
        print(f"  Min class samples: {min_class}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        
        return df
    
    def print_kids_audio_statistics(self):
        """Print statistics for Kids Audio dataset (child age group; Common Voice has no kids)."""
        if self.kids_audio_data is None:
            self.analyze_kids_audio()
        if self.kids_audio_data is None or len(self.kids_audio_data) == 0:
            print("\n" + "="*60)
            print("Kids Audio Dataset Statistics")
            print("="*60)
            print("\nNo kids audio data found.")
            print(f"  Expected path: {self.kids_audio_data_dir}")
            return None
        df = self.kids_audio_data
        print("\n" + "="*60)
        print("Kids Audio Dataset Statistics")
        print("="*60)
        print("\nTotal samples: {}".format(len(df)))
        print("\nNumeric Age Statistics:")
        print("  Min: {}".format(df['numeric_age'].min()))
        print("  Max: {}".format(df['numeric_age'].max()))
        print("  Mean: {:.2f}".format(df['numeric_age'].mean()))
        print("  Median: {}".format(df['numeric_age'].median()))
        print("\nAge Distribution (by year):")
        age_counts = df['numeric_age'].value_counts().sort_index()
        for age, count in age_counts.items():
            pct = count / len(df) * 100
            print("  {}: {} ({:.1f}%)".format(age, count, pct))
        print("\nAge Group Distribution:")
        age_group_counts = df['age_group'].value_counts()
        for group, count in age_group_counts.items():
            pct = count / len(df) * 100
            print("  {}: {} ({:.1f}%)".format(group, count, pct))
        print("\nGender Distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            if pd.notna(gender) and gender != '':
                pct = count / len(df) * 100
                print("  {}: {} ({:.1f}%)".format(gender, count, pct))
        print("\nSource: kids_audio (child speakers only; Common Voice has no kids)")
        print("\nAdult (18+) Distribution:")
        print("  Adults (18+): 0 (0.0%)")
        print("  Minors (<18): {} (100.0%)".format(len(df)))
        return df
    
    def plot_distributions(self, save_path=None):
        """Create visualization plots for face, Common Voice, and Kids Audio datasets."""
        if self.face_data is None:
            self.analyze_utkface()
        if self.audio_data is None:
            self.analyze_common_voice()
        if self.kids_audio_data is None:
            self.analyze_kids_audio()
        
        has_kids = self.kids_audio_data is not None and len(self.kids_audio_data) > 0
        nrows = 3 if has_kids else 2
        fig, axes = plt.subplots(nrows, 3, figsize=(18, 6 * nrows))
        if nrows == 2:
            axes = [axes[0], axes[1], None]
        else:
            axes = [axes[0], axes[1], axes[2]]
        
        colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
        
        # Row 0: Face
        axes[0][0].hist(self.face_data['age'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0][0].axvline(x=18, color='red', linestyle='--', label='Age 18')
        axes[0][0].set_title('UTKFace: Age Distribution', fontsize=12, fontweight='bold')
        axes[0][0].set_xlabel('Age')
        axes[0][0].set_ylabel('Count')
        axes[0][0].legend()
        age_group_counts = self.face_data['age_group'].value_counts()
        axes[0][1].bar(age_group_counts.index, age_group_counts.values, color=colors, edgecolor='black')
        axes[0][1].set_title('UTKFace: Age Group Distribution', fontsize=12, fontweight='bold')
        axes[0][1].set_xlabel('Age Group')
        axes[0][1].set_ylabel('Count')
        for i, (group, count) in enumerate(age_group_counts.items()):
            axes[0][1].annotate(str(count), xy=(i, count), ha='center', va='bottom')
        gender_counts = self.face_data['gender'].value_counts()
        axes[0][2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                       colors=['#74b9ff', '#fd79a8'], startangle=90)
        axes[0][2].set_title('UTKFace: Gender Distribution', fontsize=12, fontweight='bold')
        
        # Row 1: Common Voice
        audio_age_counts = self.audio_data['age'].value_counts()
        axes[1][0].barh(audio_age_counts.index, audio_age_counts.values, color='coral', edgecolor='black')
        axes[1][0].set_title('Common Voice: Age Category Distribution', fontsize=12, fontweight='bold')
        axes[1][0].set_xlabel('Count')
        axes[1][0].set_ylabel('Age Category')
        audio_group_counts = self.audio_data['age_group'].value_counts()
        axes[1][1].bar(audio_group_counts.index, audio_group_counts.values, color=colors, edgecolor='black')
        axes[1][1].set_title('Common Voice: Age Group Distribution', fontsize=12, fontweight='bold')
        axes[1][1].set_xlabel('Age Group')
        axes[1][1].set_ylabel('Count')
        for i, (group, count) in enumerate(audio_group_counts.items()):
            axes[1][1].annotate(str(count), xy=(i, count), ha='center', va='bottom')
        face_adult_pct = self.face_data['is_adult'].mean() * 100
        audio_adult_pct = self.audio_data['is_adult'].mean() * 100
        x = np.arange(2)
        width = 0.35
        adult_vals = [face_adult_pct, audio_adult_pct]
        minor_vals = [100 - face_adult_pct, 100 - audio_adult_pct]
        axes[1][2].bar(x - width/2, adult_vals, width, label='Adult (18+)', color='#27ae60')
        axes[1][2].bar(x + width/2, minor_vals, width, label='Minor (<18)', color='#e74c3c')
        axes[1][2].set_xticks(x)
        axes[1][2].set_xticklabels(['UTKFace', 'Common Voice'])
        axes[1][2].set_ylabel('Percentage (%)')
        axes[1][2].set_title('Adult vs Minor Distribution Comparison', fontsize=12, fontweight='bold')
        axes[1][2].legend()
        axes[1][2].set_ylim(0, 100)
        
        # Row 2: Kids Audio (when present)
        if has_kids:
            kids_df = self.kids_audio_data
            axes[2][0].hist(kids_df['numeric_age'], bins=range(int(kids_df['numeric_age'].min()), int(kids_df['numeric_age'].max()) + 2),
                           color='#9b59b6', edgecolor='black', alpha=0.7)
            axes[2][0].axvline(x=18, color='red', linestyle='--', label='Age 18')
            axes[2][0].set_title('Kids Audio: Age Distribution', fontsize=12, fontweight='bold')
            axes[2][0].set_xlabel('Age')
            axes[2][0].set_ylabel('Count')
            axes[2][0].legend()
            kids_group_counts = kids_df['age_group'].value_counts()
            axes[2][1].bar(kids_group_counts.index, kids_group_counts.values, color=colors[:1], edgecolor='black')
            axes[2][1].set_title('Kids Audio: Age Group Distribution', fontsize=12, fontweight='bold')
            axes[2][1].set_xlabel('Age Group')
            axes[2][1].set_ylabel('Count')
            for i, (group, count) in enumerate(kids_group_counts.items()):
                axes[2][1].annotate(str(count), xy=(i, count), ha='center', va='bottom')
            # Common Voice vs Kids sample counts
            cv_count = len(self.audio_data)
            kids_count = len(kids_df)
            axes[2][2].bar(['Common Voice', 'Kids Audio'], [cv_count, kids_count], color=['coral', '#9b59b6'], edgecolor='black')
            axes[2][2].set_title('Audio Sources: Common Voice vs Kids', fontsize=12, fontweight='bold')
            axes[2][2].set_ylabel('Sample Count')
            for i, v in enumerate([cv_count, kids_count]):
                axes[2][2].annotate(str(v), xy=(i, v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved distribution plots to {save_path}")
        plt.show()
        return fig

    def plot_project_report_figure(self, save_path=None, show=False):
        """
        Single PNG with three panels: train vs val curves (from checkpoints), model
        validation accuracy (face/voice age-group acc; fusion adult acc if stored),
        and adult vs minor share for UTKFace and Common Voice.
        """
        import torch

        if save_path is None:
            save_path = LOGS_DIR / "project_report_figure.png"
        save_path = Path(save_path)

        if self.face_data is None:
            self.analyze_utkface()
        if self.audio_data is None:
            self.analyze_common_voice()

        def _history_from_ckpt(ckpt_path: Path):
            if not ckpt_path.exists():
                return None
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                logger.warning(f"Could not load {ckpt_path}: {exc}")
                return None
            return ckpt.get("history") or None

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        face_ckpt = MODELS_DIR / "face_age_model_best.pth"
        voice_ckpt = MODELS_DIR / "voice_age_model_best.pth"
        fusion_ckpt = FUSION_MODELS_DIR / "fusion_age_model_best.pth"

        history = _history_from_ckpt(face_ckpt)
        curve_label = "Face"
        if not history or not history.get("train_loss"):
            history = _history_from_ckpt(voice_ckpt)
            curve_label = "Voice"

        ax0 = axes[0]
        if history and history.get("train_loss"):
            n = len(history["train_loss"])
            epochs = np.arange(1, n + 1)
            ax0.plot(epochs, history["train_loss"], label="Train loss", color="#3498db", linewidth=2)
            ax0.plot(epochs, history["val_loss"], label="Val loss", color="#2980b9", linewidth=2, linestyle="--")
            ax0.set_xlabel("Epoch")
            ax0.set_ylabel("Loss")
            ax0.set_title(f"Training comparison ({curve_label})", fontsize=12, fontweight="bold")
            ax0.grid(True, alpha=0.3)
            if history.get("train_acc") and history.get("val_acc"):
                ax0b = ax0.twinx()
                ax0b.plot(epochs, history["train_acc"], label="Train acc", color="#e67e22", linewidth=1.8)
                ax0b.plot(
                    epochs,
                    history["val_acc"],
                    label="Val acc",
                    color="#c0392b",
                    linewidth=1.8,
                    linestyle="--",
                )
                ax0b.set_ylabel("Age-group accuracy (val)")
                ax0b.set_ylim(0, 1.02)
                h1, l1 = ax0.get_legend_handles_labels()
                h2, l2 = ax0b.get_legend_handles_labels()
                ax0.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
            else:
                ax0.legend(loc="upper right", fontsize=9)
        else:
            ax0.text(
                0.5,
                0.5,
                "No checkpoint history found.\nTrain face/voice; checkpoints\nsave under models/",
                ha="center",
                va="center",
                transform=ax0.transAxes,
                fontsize=10,
            )
            ax0.set_axis_off()
            ax0.set_title("Training comparison", fontsize=12, fontweight="bold")

        # Model accuracy: last val age-group acc for face/voice; fusion from checkpoint extras or history
        models, accs = [], []
        for name, path in (("Face", face_ckpt), ("Voice", voice_ckpt)):
            h = _history_from_ckpt(path)
            if h and h.get("val_acc"):
                models.append(name)
                accs.append(float(h["val_acc"][-1]) * 100.0)

        if fusion_ckpt.exists():
            try:
                fc = torch.load(fusion_ckpt, map_location="cpu", weights_only=False)
                fh = fc.get("history") or {}
                added = False
                if fh.get("val_acc"):
                    models.append("Fusion")
                    accs.append(float(fh["val_acc"][-1]) * 100.0)
                    added = True
                if not added:
                    for key in ("test_adult_accuracy", "val_adult_accuracy", "best_val_adult_acc"):
                        if key in fc and fc[key] is not None:
                            v = float(fc[key])
                            models.append("Fusion")
                            accs.append(v * 100.0 if v <= 1.0 else v)
                            break
            except Exception as exc:
                logger.warning(f"Could not read fusion metrics: {exc}")

        ax1 = axes[1]
        if models:
            colors = ["#3498db", "#9b59b6", "#1abc9c"][: len(models)]
            bars = ax1.bar(models, accs, color=colors, edgecolor="black", width=0.55)
            ax1.set_ylabel("Accuracy (%)")
            ax1.set_title("Model accuracy", fontsize=12, fontweight="bold")
            ymax = max(accs) if accs else 100.0
            ax1.set_ylim(0, min(100.0, ymax * 1.12 + 3))
            for bar, v in zip(bars, accs):
                ax1.annotate(
                    f"{v:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            ax1.set_xlabel("Face/Voice: val age-group. Fusion: adult or val acc if present.", fontsize=8)
        else:
            ax1.text(
                0.5,
                0.5,
                "No val accuracy in checkpoints.",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_axis_off()
            ax1.set_title("Model accuracy", fontsize=12, fontweight="bold")

        ax2 = axes[2]
        has_face = self.face_data is not None and len(self.face_data) > 0
        has_audio = self.audio_data is not None and len(self.audio_data) > 0
        if has_face and has_audio:
            face_adult_pct = float(self.face_data["is_adult"].mean()) * 100.0
            audio_adult_pct = float(self.audio_data["is_adult"].mean()) * 100.0
            x = np.arange(2)
            width = 0.35
            adult_vals = [face_adult_pct, audio_adult_pct]
            minor_vals = [100.0 - face_adult_pct, 100.0 - audio_adult_pct]
            ax2.bar(x - width / 2, adult_vals, width, label="Adult (18+)", color="#27ae60", edgecolor="black")
            ax2.bar(x + width / 2, minor_vals, width, label="Minor (<18)", color="#e74c3c", edgecolor="black")
            ax2.set_xticks(x)
            ax2.set_xticklabels(["UTKFace", "Common Voice"])
            ax2.set_ylabel("Percentage (%)")
            ax2.legend()
            ax2.set_ylim(0, 100)
        elif has_face:
            adult_pct = float(self.face_data["is_adult"].mean()) * 100.0
            ax2.bar(
                ["Adult (18+)", "Minor (<18)"],
                [adult_pct, 100.0 - adult_pct],
                color=["#27ae60", "#e74c3c"],
                edgecolor="black",
            )
            ax2.set_ylabel("Percentage (%)")
            ax2.set_ylim(0, 100)
            ax2.set_xlabel("UTKFace", fontsize=9)
        elif has_audio:
            adult_pct = float(self.audio_data["is_adult"].mean()) * 100.0
            ax2.bar(
                ["Adult (18+)", "Minor (<18)"],
                [adult_pct, 100.0 - adult_pct],
                color=["#27ae60", "#e74c3c"],
                edgecolor="black",
            )
            ax2.set_ylabel("Percentage (%)")
            ax2.set_ylim(0, 100)
            ax2.set_xlabel("Common Voice", fontsize=9)
        else:
            ax2.text(
                0.5,
                0.5,
                "No face/audio rows parsed.\nCheck image-data and audio-data paths.",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=10,
            )
            ax2.set_axis_off()
        if has_face or has_audio:
            ax2.set_title("Adult & minor data distribution", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved project report figure to {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def get_stratified_splits(self, data, test_size=0.15, val_size=0.15, random_state=42):
        """
        Create stratified train/val/test splits
        Uses age groups for stratification
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            data,
            test_size=test_size,
            stratify=data['age_group'],
            random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            stratify=train_val['age_group'],
            random_state=random_state
        )
        
        logger.info(f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def get_class_weights(self, data, target_col='age_group'):
        """Calculate class weights for handling imbalance"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = data[target_col].unique()
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=data[target_col]
        )
        
        weight_dict = dict(zip(classes, weights))
        logger.info(f"Class weights: {weight_dict}")
        
        return weight_dict
    
    def prepare_face_data(self):
        """Prepare face data with shuffling, stratification, and class weights"""
        if self.face_data is None:
            self.analyze_utkface()
        
        # Shuffle
        df = self.face_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Stratified splits
        train, val, test = self.get_stratified_splits(df)
        
        # Class weights
        class_weights = self.get_class_weights(train)
        
        return {
            'train': train,
            'val': val,
            'test': test,
            'class_weights': class_weights
        }
    
    def prepare_audio_data(self, include_kids=True):
        """Prepare audio data with shuffling, stratification, and class weights
        
        Args:
            include_kids: Whether to include kids audio dataset (for child age group)
        """
        if self.audio_data is None:
            self.analyze_common_voice()
        
        # Start with Common Voice data
        combined_df = self.audio_data.copy()
        combined_df['source'] = 'common_voice'
        
        # Add kids audio data if requested
        if include_kids:
            if self.kids_audio_data is None:
                self.analyze_kids_audio()
            
            if self.kids_audio_data is not None and len(self.kids_audio_data) > 0:
                logger.info(f"Adding {len(self.kids_audio_data)} kids audio samples")
                
                # Align columns
                kids_df = self.kids_audio_data.copy()
                
                # Ensure common columns exist
                for col in ['filename', 'numeric_age', 'age_group', 'is_adult', 'text', 'filepath']:
                    if col not in combined_df.columns:
                        combined_df[col] = ''
                    if col not in kids_df.columns:
                        kids_df[col] = ''
                
                # Select only necessary columns for combination
                common_cols = ['filepath', 'filename', 'numeric_age', 'age_group', 'is_adult', 'text', 'source']
                
                # Add any missing columns
                for col in common_cols:
                    if col not in combined_df.columns:
                        combined_df[col] = ''
                    if col not in kids_df.columns:
                        kids_df[col] = ''
                
                combined_df = pd.concat([
                    combined_df[common_cols],
                    kids_df[common_cols]
                ], ignore_index=True)
                
                logger.info(f"Combined audio dataset: {len(combined_df)} total samples")
        
        # Shuffle
        df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Stratified splits
        train, val, test = self.get_stratified_splits(df)
        
        # Class weights
        class_weights = self.get_class_weights(train)
        
        return {
            'train': train,
            'val': val,
            'test': test,
            'class_weights': class_weights
        }


def run_full_analysis():
    """Run complete data analysis and print results"""
    analyzer = DataAnalyzer()
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "DATA ANALYSIS REPORT" + " "*20 + "#")
    print("#"*70)
    
    # Analyze both datasets
    analyzer.print_face_statistics()
    analyzer.print_audio_statistics()
    analyzer.analyze_kids_audio()
    analyzer.print_kids_audio_statistics()
    
    # Prepare data with stratification
    print("\n" + "="*60)
    print("Preparing Stratified Data Splits")
    print("="*60)
    
    face_data = analyzer.prepare_face_data()
    print("\nFace Data Prepared:")
    print(f"  Train samples: {len(face_data['train'])}")
    print(f"  Val samples: {len(face_data['val'])}")
    print(f"  Test samples: {len(face_data['test'])}")
    
    audio_data = analyzer.prepare_audio_data()
    print("\nAudio Data Prepared:")
    print(f"  Train samples: {len(audio_data['train'])}")
    print(f"  Val samples: {len(audio_data['val'])}")
    print(f"  Test samples: {len(audio_data['test'])}")
    
    plot_path = LOGS_DIR / "data_distribution_analysis.png"
    analyzer.plot_distributions(save_path=plot_path)

    report_path = LOGS_DIR / "project_report_figure.png"
    plt.close("all")
    analyzer.plot_project_report_figure(save_path=report_path)

    return analyzer, face_data, audio_data


if __name__ == "__main__":
    run_full_analysis()

