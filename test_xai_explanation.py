# test_xai_explanation_real.py
"""
Test XAI explanations using REAL data from the preprocessed dataset
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from xai_qos.traffic_classifier.attention_classifier import AttentionTrafficClassifier
from xai_qos.xai_engine.attention_explainer import AttentionExplainer
from xai_qos.utils.config_utils import ConfigManager


def print_separator(char='=', length=70):
    print(char * length)


def print_explanation(test_name, explanation, classification, actual_label):
    """Pretty print explanation results"""
    print_separator()
    print(f"🧪 TEST: {test_name}")
    print_separator()
    
    # Show actual vs predicted
    correct = "✅" if classification['predicted_class'] == actual_label else "❌"
    print(f"\n{correct} Actual Label: {actual_label}")
    print(f"   Predicted: {classification['predicted_class']}")
    print(f"   Confidence: {classification['confidence']:.2%}")
    
    # Show all class probabilities
    print(f"\n📊 All Class Probabilities:")
    for cls, prob in sorted(classification['class_probabilities'].items(), 
                           key=lambda x: x[1], reverse=True):
        print(f"   {cls}: {prob:.2%}")
    
    # Key features
    print(f"\n🔑 Key Features:")
    for feature in explanation.get('key_features', [])[:5]:
        print(f"   • {feature['name']}: {feature['importance']:.3f} (value: {feature['value']:.2f})")
    
    # QoS
    qos = explanation.get('qos_requirements', {})
    print(f"\n🎯 QoS: Priority={qos.get('priority_level')}, "
          f"Latency={qos.get('latency_requirement')}, "
          f"Bandwidth={qos.get('bandwidth_requirement')}")
    print()


def main():
    print_separator('=', 70)
    print("🚀 XAI-QoS Framework - Real Data Testing")
    print_separator('=', 70)
    
    # Load model
    print("\n📥 Loading model...")
    config = ConfigManager.load_config('config/xai-qos-config.yaml')
    classifier = AttentionTrafficClassifier(config['classifier'], 'mps')
    classifier.load_model('models/xai_qos_classifier.pth')
    explainer = AttentionExplainer(classifier, config['explainer'])
    
    # Load test data (from your preprocessed dataset)
    print("📂 Loading test data from preprocessed dataset...")
    df = pd.read_csv('data/processed/traffic_data.csv')
    
    print(f"   Total samples: {len(df)}")
    print(f"   Classes: {df['label'].unique()}")
    print(f"\n   Class distribution:")
    print(df['label'].value_counts())
    
    # Sample flows from each class
    print("\n🔍 Testing with real flows from each class...\n")
    
    results = []
    for label in df['label'].unique():
        # Get 3 random samples from this class
        samples = df[df['label'] == label].sample(n=min(3, len(df[df['label'] == label])), 
                                                   random_state=42)
        
        # NEW CODE - Map DataFrame columns to expected feature names:
        for idx, row in samples.iterrows():
            # Create flow_data with COLUMN NAMES as keys (not synthetic feature names)
            flow_data = {}
            
            # Copy all features from DataFrame row
            for col in row.index:
                if col != 'label':
                    flow_data[col] = row[col]
            
            flow_data['flow_id'] = f"{label}_{idx}"

            
            # Classify and explain
            classification = classifier.classify_flow(flow_data)
            explanation = explainer.explain_classification(flow_data, classification)
            
            # Print result
            print_explanation(
                f"{label.upper()} Sample #{idx}", 
                explanation, 
                classification,
                label
            )
            
            # Store for summary
            results.append({
                'actual': label,
                'predicted': classification['predicted_class'],
                'confidence': classification['confidence'],
                'correct': classification['predicted_class'] == label
            })
    
    # Calculate accuracy
    print_separator('=', 70)
    print("📊 REAL DATA TEST SUMMARY")
    print_separator('=', 70)
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"\n✅ Overall Accuracy: {accuracy:.2%}")
    
    # Per-class accuracy
    print(f"\n📈 Per-Class Performance:")
    for label in df['label'].unique():
        class_results = [r for r in results if r['actual'] == label]
        class_acc = sum(r['correct'] for r in class_results) / len(class_results)
        print(f"   {label}: {class_acc:.2%} ({sum(r['correct'] for r in class_results)}/{len(class_results)} correct)")
    
    # Confusion matrix
    print(f"\n🔀 Confusion Matrix:")
    print(f"{'':15} ", end='')
    for label in sorted(df['label'].unique()):
        print(f"{label:15}", end='')
    print()
    
    for actual in sorted(df['label'].unique()):
        print(f"{actual:15} ", end='')
        for predicted in sorted(df['label'].unique()):
            count = sum(1 for r in results if r['actual'] == actual and r['predicted'] == predicted)
            print(f"{count:15}", end='')
        print()
    
    print_separator('=', 70)


if __name__ == "__main__":
    main()
