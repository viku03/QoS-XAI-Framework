# xai_qos/traffic_classifier/__init__.py

"""
Traffic Classifier Module
Attention-based neural network for fine-grained traffic classification
"""

from .attention_classifier import (
    AttentionTrafficClassifier,
    MultiHeadAttention,
    TrafficFeatureExtractor
)

__all__ = [
    'AttentionTrafficClassifier',
    'MultiHeadAttention',
    'TrafficFeatureExtractor'
]
