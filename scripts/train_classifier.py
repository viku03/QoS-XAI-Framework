# Training Script for XAI-QoS Traffic Classifier
# scripts/train_classifier.py

"""
Train the attention-based traffic classifier with progress bars and checkpoints
Optimized for Mac M1/M2 with MPS support
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from xai_qos.traffic_classifier.attention_classifier import AttentionTrafficClassifier
from xai_qos.utils.config_utils import ConfigManager, setup_logging

def load_and_preprocess_data(data_path: str, config: dict):
    """Load and preprocess traffic dataset"""
    logging.info(f"Loading data from {data_path}")
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Separate features and labels
    label_column = 'label'
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    y = df[label_column].values
    X = df.drop(columns=[label_column]).values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Update class names in config
    class_names = label_encoder.classes_.tolist()
    config['classifier']['class_names'] = class_names
    config['classifier']['n_classes'] = len(class_names)
    config['classifier']['input_dim'] = X.shape[1]
    
    logging.info(f"Dataset shape: {X.shape}")
    logging.info(f"Classes: {class_names}")
    logging.info(f"Class distribution: {np.bincount(y_encoded)}")
    
    return X, y_encoded, label_encoder

def create_data_loaders(X, y, test_size=0.2, validation_size=0.1, batch_size=32):
    """Create PyTorch data loaders with progress tracking"""
    
    logging.info("Creating data loaders...")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + validation_size, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size + validation_size), 
        random_state=42, stratify=y_temp
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info(f"Train set: {len(train_dataset)} samples")
    logging.info(f"Validation set: {len(val_dataset)} samples")
    logging.info(f"Test set: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, scaler

def train_model(model, train_loader, val_loader, config, device, checkpoint_dir='checkpoints'):
    """Train the attention-based traffic classifier with progress bars"""
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    
    num_epochs = config.get('num_epochs', 100)
    checkpoint_freq = config.get('checkpoint_freq', 10)  # Save every N epochs
    best_val_accuracy = 0.0
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    logging.info(f"Starting training for {num_epochs} epochs on {device}")
    logging.info(f"Checkpoints will be saved every {checkpoint_freq} epochs to {checkpoint_dir}/")
    
    # Main training loop with progress bar
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training batch progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         leave=False, unit="batch")
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs, attention_weights = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
            
        train_accuracy = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Validation batch progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                       leave=False, unit="batch")
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                outputs, attention_weights = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
                
        val_accuracy = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Log epoch summary
        tqdm.write(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        tqdm.write(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        tqdm.write(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        tqdm.write(f"  Best Val Acc: {max(val_accuracies):.2f}%")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': avg_val_loss,
                'config': config
            }
            torch.save(best_checkpoint, f'{checkpoint_dir}/best_model.pth')
            tqdm.write(f"  ✅ New best model saved!")
        
        # Save periodic checkpoints
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'config': config
            }
            checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            tqdm.write(f"  💾 Checkpoint saved: {checkpoint_path}")
            
    logging.info(f"\n{'='*60}")
    logging.info(f"Training completed!")
    logging.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    logging.info(f"{'='*60}")
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, class_names, device):
    """Evaluate the trained model"""
    logging.info("Evaluating model on test set...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Evaluation progress bar
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    with torch.no_grad():
        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)
            outputs, attention_weights = model(data)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100.0 * np.sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Test Accuracy: {accuracy:.2f}%")
    logging.info(f"{'='*60}")
    
    # Classification report
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    logging.info("\nClassification Report:")
    logging.info(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return accuracy, report, cm

def plot_training_history(train_losses, val_losses, val_accuracies, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Training history saved to {save_path}")
    
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Confusion matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train XAI-QoS Traffic Classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--config', type=str, default='config/xai-qos-config.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='models/xai_qos_classifier.pth', help='Output model path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/mps/auto)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    logging.info("="*60)
    logging.info("XAI-QoS Traffic Classifier Training")
    logging.info("="*60)
    
    # Load configuration
    config = ConfigManager.load_config(args.config)
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size  
    config['learning_rate'] = args.learning_rate
    config['checkpoint_freq'] = args.checkpoint_freq
    
    # Set device with Mac M1/M2 MPS support
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'  # Mac M1/M2 GPU
        else:
            device = 'cpu'
    else:
        device = args.device
    
    if device == 'mps' and not torch.backends.mps.is_available():
        logging.warning("MPS not available on this Mac, using CPU")
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logging.info(f"Using device: {device.upper()}")
    if device == 'mps':
        logging.info("🚀 Mac M1/M2 GPU acceleration enabled!")
    
    try:
        # Load and preprocess data
        X, y, label_encoder = load_and_preprocess_data(args.data, config)
        
        # Create data loaders
        train_loader, val_loader, test_loader, scaler = create_data_loaders(
            X, y, batch_size=args.batch_size
        )
        
        # Create model
        model = AttentionTrafficClassifier(config['classifier'], device)
        model.scaler = scaler
        model.scaler_fitted = True
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"\nModel Architecture:")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Train model
        train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, config, device
        )
        
        # Load best model for evaluation
        best_checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Evaluate model
        accuracy, report, cm = evaluate_model(
            model, test_loader, config['classifier']['class_names'], device
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save final model
        model.save_model(args.output)
        logging.info(f"\n✅ Final model saved to {args.output}")
        
        # Plot results
        plot_training_history(train_losses, val_losses, val_accuracies)
        plot_confusion_matrix(cm, config['classifier']['class_names'])
        
        # Save training results
        import json
        results = {
            'test_accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': config['classifier']['class_names'],
            'training_config': config,
            'device': device,
            'total_epochs': args.epochs,
            'best_val_accuracy': max(val_accuracies)
        }
        
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"\n{'='*60}")
        logging.info("Training completed successfully! 🎉")
        logging.info(f"{'='*60}")
        logging.info(f"Best validation accuracy: {max(val_accuracies):.2f}%")
        logging.info(f"Final test accuracy: {accuracy:.2f}%")
        logging.info(f"Model saved to: {args.output}")
        logging.info(f"Checkpoints saved to: checkpoints/")
        logging.info(f"{'='*60}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()