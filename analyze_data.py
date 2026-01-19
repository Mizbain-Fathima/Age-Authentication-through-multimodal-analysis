#!/usr/bin/env python
"""
Data Analysis Script for Age Authentication System
Run this script to analyze the UTKFace and Mozilla Common Voice datasets.

Usage:
    python analyze_data.py
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMAGE_DATA_DIR = Path(__file__).parent / "image-data" / "UTKFace"
AUDIO_DATA_DIR = Path(__file__).parent / "audio-data"

# Age mappings
AUDIO_AGE_MAP = {
    'teens': 15,
    'twenties': 24,
    'thirties': 34,
    'fourties': 44,
    'fifties': 54,
    'sixties': 64,
    'seventies': 74,
    'eighties': 84,
    'nineties': 92
}

AGE_GROUPS = {
    'child': (0, 12),
    'teen': (13, 17),
    'adult': (18, 59),
    'senior': (60, 120)
}


def get_age_group(age):
    """Map numeric age to age group"""
    for group, (min_age, max_age) in AGE_GROUPS.items():
        if min_age <= age <= max_age:
            return group
    return 'adult'


def analyze_utkface():
    """Analyze UTKFace dataset"""
    print("\n" + "="*70)
    print("📸 UTKFace Dataset Analysis")
    print("="*70)
    
    if not IMAGE_DATA_DIR.exists():
        print(f"❌ Directory not found: {IMAGE_DATA_DIR}")
        return None
    
    image_files = list(IMAGE_DATA_DIR.glob("*.jpg"))
    print(f"\n📁 Found {len(image_files)} image files")
    
    data = []
    parse_errors = 0
    
    for img_path in image_files:
        filename = img_path.name
        parts = filename.split('_')
        
        if len(parts) >= 4:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                
                if 0 <= age <= 116:
                    data.append({
                        'filepath': str(img_path),
                        'age': age,
                        'gender': 'male' if gender == 0 else 'female',
                        'race': race,
                        'age_group': get_age_group(age),
                        'is_adult': age >= 18
                    })
                else:
                    parse_errors += 1
            except (ValueError, IndexError):
                parse_errors += 1
        else:
            parse_errors += 1
    
    df = pd.DataFrame(data)
    
    print(f"✅ Successfully parsed: {len(df)} images")
    print(f"⚠️  Parse errors: {parse_errors}")
    
    # Basic statistics
    print(f"\n📊 Age Statistics:")
    print(f"   Min age: {df['age'].min()}")
    print(f"   Max age: {df['age'].max()}")
    print(f"   Mean age: {df['age'].mean():.2f}")
    print(f"   Median age: {df['age'].median():.0f}")
    print(f"   Std deviation: {df['age'].std():.2f}")
    
    # Age group distribution
    print(f"\n👥 Age Group Distribution:")
    age_group_counts = df['age_group'].value_counts()
    total = len(df)
    for group in ['child', 'teen', 'adult', 'senior']:
        count = age_group_counts.get(group, 0)
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"   {group:8s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    # Gender distribution
    print(f"\n♀️♂️ Gender Distribution:")
    gender_counts = df['gender'].value_counts()
    for gender, count in gender_counts.items():
        pct = count / total * 100
        print(f"   {gender:8s}: {count:6d} ({pct:5.1f}%)")
    
    # Adult vs Minor
    print(f"\n🔞 Adult (18+) vs Minor Distribution:")
    adult_counts = df['is_adult'].value_counts()
    adults = adult_counts.get(True, 0)
    minors = adult_counts.get(False, 0)
    print(f"   Adults (18+): {adults:6d} ({adults/total*100:5.1f}%)")
    print(f"   Minors (<18): {minors:6d} ({minors/total*100:5.1f}%)")
    
    # Age by decade
    print(f"\n📅 Age Distribution by Decade:")
    df['decade'] = (df['age'] // 10) * 10
    decade_counts = df['decade'].value_counts().sort_index()
    for decade, count in decade_counts.items():
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"   {decade:2.0f}-{decade+9:<2.0f}: {count:6d} ({pct:5.1f}%) {bar}")
    
    # Class imbalance
    print(f"\n⚖️  Class Imbalance Analysis:")
    max_class = age_group_counts.max()
    min_class = age_group_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"   Max class samples: {max_class}")
    print(f"   Min class samples: {min_class}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
    
    return df


def analyze_common_voice():
    """Analyze Mozilla Common Voice dataset"""
    print("\n" + "="*70)
    print("🎤 Mozilla Common Voice Dataset Analysis")
    print("="*70)
    
    if not AUDIO_DATA_DIR.exists():
        print(f"❌ Directory not found: {AUDIO_DATA_DIR}")
        return None
    
    csv_files = [
        'cv-other-train.csv',
        'cv-other-dev.csv',
        'cv-other-test.csv'
    ]
    
    all_data = []
    
    for csv_file in csv_files:
        csv_path = AUDIO_DATA_DIR / csv_file
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['split'] = csv_file.replace('cv-other-', '').replace('.csv', '')
            all_data.append(df)
            print(f"📁 Loaded {len(df):,} samples from {csv_file}")
    
    if not all_data:
        print("❌ No CSV files found")
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total samples: {len(df):,}")
    
    # Filter samples with age
    df_with_age = df[df['age'].notna() & (df['age'] != '')]
    print(f"✅ Samples with age info: {len(df_with_age):,}")
    
    # Add numeric age
    df_with_age = df_with_age.copy()
    df_with_age['numeric_age'] = df_with_age['age'].map(AUDIO_AGE_MAP)
    df_with_age['age_group'] = df_with_age['numeric_age'].apply(get_age_group)
    df_with_age['is_adult'] = df_with_age['numeric_age'] >= 18
    
    # Check file existence (sample)
    sample_paths = df_with_age['filename'].head(100)
    exists_count = sum(1 for f in sample_paths if (AUDIO_DATA_DIR / f).exists())
    print(f"📁 Audio files verified (sample): {exists_count}/100")
    
    # Age category distribution
    print(f"\n📊 Age Category Distribution:")
    age_counts = df_with_age['age'].value_counts()
    total = len(df_with_age)
    for age_cat in ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 
                    'sixties', 'seventies', 'eighties', 'nineties']:
        count = age_counts.get(age_cat, 0)
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"   {age_cat:10s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    # Age group distribution
    print(f"\n👥 Age Group Distribution:")
    age_group_counts = df_with_age['age_group'].value_counts()
    for group in ['child', 'teen', 'adult', 'senior']:
        count = age_group_counts.get(group, 0)
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"   {group:8s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    # Gender distribution
    print(f"\n♀️♂️ Gender Distribution:")
    df_with_gender = df_with_age[df_with_age['gender'].notna() & (df_with_age['gender'] != '')]
    gender_counts = df_with_gender['gender'].value_counts()
    for gender, count in gender_counts.items():
        pct = count / len(df_with_gender) * 100
        print(f"   {gender:8s}: {count:6d} ({pct:5.1f}%)")
    
    # Adult vs Minor
    print(f"\n🔞 Adult (18+) vs Minor Distribution:")
    adult_counts = df_with_age['is_adult'].value_counts()
    adults = adult_counts.get(True, 0)
    minors = adult_counts.get(False, 0)
    print(f"   Adults (18+): {adults:6d} ({adults/total*100:5.1f}%)")
    print(f"   Minors (<18): {minors:6d} ({minors/total*100:5.1f}%)")
    
    # Split distribution
    print(f"\n📂 Data Split Distribution:")
    split_counts = df_with_age['split'].value_counts()
    for split, count in split_counts.items():
        pct = count / total * 100
        print(f"   {split:8s}: {count:6d} ({pct:5.1f}%)")
    
    # Class imbalance
    print(f"\n⚖️  Class Imbalance Analysis:")
    max_class = age_group_counts.max()
    min_class = age_group_counts.min()
    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
    print(f"   Max class samples: {max_class}")
    print(f"   Min class samples: {min_class}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
    
    return df_with_age


def print_summary(face_df, audio_df):
    """Print combined summary"""
    print("\n" + "="*70)
    print("📈 COMBINED SUMMARY")
    print("="*70)
    
    print("\n┌─────────────────────────┬────────────┬────────────┐")
    print("│ Metric                  │  UTKFace   │ Common V.  │")
    print("├─────────────────────────┼────────────┼────────────┤")
    
    if face_df is not None and audio_df is not None:
        print(f"│ Total samples           │ {len(face_df):>10,} │ {len(audio_df):>10,} │")
        print(f"│ Children (0-12)         │ {(face_df['age_group']=='child').sum():>10,} │ {(audio_df['age_group']=='child').sum():>10,} │")
        print(f"│ Teens (13-17)           │ {(face_df['age_group']=='teen').sum():>10,} │ {(audio_df['age_group']=='teen').sum():>10,} │")
        print(f"│ Adults (18-59)          │ {(face_df['age_group']=='adult').sum():>10,} │ {(audio_df['age_group']=='adult').sum():>10,} │")
        print(f"│ Seniors (60+)           │ {(face_df['age_group']=='senior').sum():>10,} │ {(audio_df['age_group']=='senior').sum():>10,} │")
        print(f"│ Adults (18+)            │ {face_df['is_adult'].sum():>10,} │ {audio_df['is_adult'].sum():>10,} │")
        print(f"│ Minors (<18)            │ {(~face_df['is_adult']).sum():>10,} │ {(~audio_df['is_adult']).sum():>10,} │")
    
    print("└─────────────────────────┴────────────┴────────────┘")
    
    print("\n✨ Data Preparation Recommendations:")
    print("   1. Apply stratified sampling by age group for train/val/test splits")
    print("   2. Use class weights in loss function to handle imbalance")
    print("   3. Apply data augmentation for underrepresented classes")
    print("   4. Consider oversampling teens/seniors in audio data")


def main():
    print("\n" + "🔬 "*20)
    print("     AGE AUTHENTICATION SYSTEM - DATA ANALYSIS")
    print("🔬 "*20)
    
    # Analyze both datasets
    face_df = analyze_utkface()
    audio_df = analyze_common_voice()
    
    # Print combined summary
    print_summary(face_df, audio_df)
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70 + "\n")
    
    return face_df, audio_df


if __name__ == "__main__":
    main()

