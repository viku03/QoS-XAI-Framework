# Data Preprocessing Script for CICIDS2017
# data/preprocess_cicids.py

"""
Preprocess CICIDS2017 dataset for XAI-QoS Framework
Merges CSV files, maps labels, and prepares data for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Label mapping from CICIDS2017 to your 10 application classes
LABEL_MAPPING = {
    'BENIGN': 'http_browsing',
    'FTP-Patator': 'ftp',
    'SSH-Patator': 'background',
    'DoS slowloris': 'background',
    'DoS Slowhttptest': 'background',
    'DoS Hulk': 'background',
    'DoS GoldenEye': 'background',
    'Heartbleed': 'background',
    'Web Attack – Brute Force': 'http_browsing',
    'Web Attack – XSS': 'http_browsing',
    'Web Attack – Sql Injection': 'http_browsing',
    'Infiltration': 'background',
    'Bot': 'background',
    'PortScan': 'background',
    'DDoS': 'background',
}

# Features to keep (matching your classifier's expected input)
SELECTED_FEATURES = [
    # Flow identifiers
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Bwd URG Flags',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
]


def load_cicids_files(data_dir: Path):
    """Load all CICIDS2017 CSV files"""
    csv_files = list(data_dir.glob('*.csv'))
    logger.info(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
            # Clean column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            logger.info(f"  Loaded {len(df)} rows")
        except Exception as e:
            logger.error(f"  Error loading {csv_file.name}: {e}")
    
    # Combine all dataframes
    logger.info("Combining all dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows: {len(combined_df)}")
    
    return combined_df


def clean_and_map_labels(df: pd.DataFrame):
    """Clean and map labels to application classes"""
    # Find label column (might be 'Label' or ' Label')
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Label column not found in dataset")
    
    logger.info(f"Found label column: '{label_col}'")
    logger.info(f"Original labels: {df[label_col].unique()}")
    
    # Map labels
    df['label'] = df[label_col].map(LABEL_MAPPING)
    
    # Remove unmapped labels
    original_count = len(df)
    df = df.dropna(subset=['label'])
    removed_count = original_count - len(df)
    
    logger.info(f"Mapped labels: {df['label'].unique()}")
    logger.info(f"Removed {removed_count} rows with unmapped labels")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def select_and_clean_features(df: pd.DataFrame):
    """Select relevant features and handle missing values"""
    # Check which features are available
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    logger.info(f"Using {len(available_features)} features")
    
    # Select features + label
    df_selected = df[available_features + ['label']].copy()
    
    # Handle missing values
    logger.info("Handling missing values...")
    
    # Replace infinity with NaN
    df_selected = df_selected.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with 0 (conservative approach for network features)
    df_selected = df_selected.fillna(0)
    
    # Remove any remaining problematic rows
    df_selected = df_selected.dropna()
    
    logger.info(f"Final dataset shape: {df_selected.shape}")
    
    return df_selected


def balance_dataset(df: pd.DataFrame, samples_per_class: int = 10000):
    """Balance dataset by sampling equal number from each class"""
    logger.info(f"Balancing dataset to {samples_per_class} samples per class...")
    
    balanced_dfs = []
    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        if len(class_df) > samples_per_class:
            class_df = class_df.sample(n=samples_per_class, random_state=42)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Balanced dataset shape: {balanced_df.shape}")
    logger.info(f"Class distribution:\n{balanced_df['label'].value_counts()}")
    
    return balanced_df


def main():
    # Paths
    data_dir = Path('data/cicids2017')
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("CICIDS2017 Data Preprocessing")
    logger.info("="*60)
    
    # Step 1: Load all CSV files
    logger.info("\n[Step 1] Loading CICIDS2017 CSV files...")
    df = load_cicids_files(data_dir)
    
    # Step 2: Clean and map labels
    logger.info("\n[Step 2] Cleaning and mapping labels...")
    df = clean_and_map_labels(df)
    
    # Step 3: Select and clean features
    logger.info("\n[Step 3] Selecting and cleaning features...")
    df = select_and_clean_features(df)
    
    # Step 4: Balance dataset (optional but recommended)
    logger.info("\n[Step 4] Balancing dataset...")
    df = balance_dataset(df, samples_per_class=10000)
    
    # Step 5: Save processed data
    output_file = output_dir / 'traffic_data.csv'
    logger.info(f"\n[Step 5] Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    logger.info("\n" + "="*60)
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Processed data saved to: {output_file}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Features: {len(df.columns) - 1}")
    logger.info(f"Classes: {df['label'].nunique()}")
    logger.info("="*60)
    
    # Print sample
    logger.info("\nSample data:")
    print(df.head())
    
    # Print statistics
    logger.info("\nFeature statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()