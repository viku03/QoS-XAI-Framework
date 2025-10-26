# xai_qos/xai_engine/__init__.py

"""
XAI Engine Module
Generates real-time explanations using attention weights
"""

from .attention_explainer import AttentionExplainer

__all__ = ['AttentionExplainer']
