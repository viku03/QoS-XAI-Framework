# xai_qos/sdn_controller/__init__.py

"""
SDN Controller Module
Policy translation and enforcement via Ryu/OpenFlow
"""

from .policy_translator import (
    PolicyTranslator,
    QoSPolicy,
    QoSPriority
)

# Make Ryu controller import optional
try:
    from .ryu_controller import XAIQoSController
    __all__ = [
        'PolicyTranslator',
        'QoSPolicy',
        'QoSPriority',
        'XAIQoSController'
    ]
except ImportError:
    # Ryu not available or dependencies missing
    __all__ = [
        'PolicyTranslator',
        'QoSPolicy',
        'QoSPriority'
    ]
