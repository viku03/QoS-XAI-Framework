# Policy Translator: XAI Explanations to QoS Policies
"""
Translates XAI explanations into concrete SDN/OpenFlow policies
Converts semantic understanding into actionable network control
"""

import json
from typing import Dict, List, Any, Tuple
import logging
from enum import Enum
from dataclasses import dataclass

class QoSPriority(Enum):
    CRITICAL = 1
    HIGH = 2  
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class QoSPolicy:
    """Represents a QoS policy to be enforced"""
    flow_id: str
    priority: QoSPriority
    min_bandwidth: int  # in Kbps
    max_bandwidth: int  # in Kbps  
    max_latency: int    # in ms
    max_jitter: int     # in ms
    max_packet_loss: float  # percentage
    queue_id: int
    dscp_marking: int
    actions: List[str]
    explanation: str

class PolicyTranslator:
    """
    Translates XAI explanations into concrete QoS policies for SDN enforcement
    """
    
    def __init__(self, config: Dict[str, Any], qos_policies: Dict[str, Any]):
        self.config = config
        self.qos_policies = qos_policies
        self.logger = logging.getLogger(__name__)
        
        # Policy templates based on application types
        self.policy_templates = {
            'youtube': {
                'priority': QoSPriority.HIGH,
                'min_bandwidth': 2000,  # 2 Mbps
                'max_bandwidth': 25000,  # 25 Mbps  
                'max_latency': 150,
                'max_jitter': 50,
                'max_packet_loss': 0.1,
                'queue_id': 2,
                'dscp_marking': 34,  # AF41
                'traffic_shaping': True,
                'burst_allowance': 5000  # KB
            },
            'netflix': {
                'priority': QoSPriority.HIGH,
                'min_bandwidth': 3000,  # 3 Mbps
                'max_bandwidth': 50000,  # 50 Mbps
                'max_latency': 200,
                'max_jitter': 100,
                'max_packet_loss': 0.1,
                'queue_id': 2,
                'dscp_marking': 34,  # AF41
                'traffic_shaping': True,
                'burst_allowance': 10000  # KB
            },
            'zoom': {
                'priority': QoSPriority.CRITICAL,
                'min_bandwidth': 1500,  # 1.5 Mbps
                'max_bandwidth': 5000,  # 5 Mbps
                'max_latency': 50,
                'max_jitter': 20,
                'max_packet_loss': 0.01,
                'queue_id': 1,
                'dscp_marking': 46,  # EF
                'traffic_shaping': False,
                'preemption_allowed': True
            },
            'teams': {
                'priority': QoSPriority.CRITICAL,
                'min_bandwidth': 1200,  # 1.2 Mbps
                'max_bandwidth': 4000,  # 4 Mbps
                'max_latency': 50,
                'max_jitter': 20,
                'max_packet_loss': 0.01,
                'queue_id': 1,
                'dscp_marking': 46,  # EF
                'traffic_shaping': False,
                'preemption_allowed': True
            },
            'gaming': {
                'priority': QoSPriority.CRITICAL,
                'min_bandwidth': 100,   # 100 Kbps
                'max_bandwidth': 1000,  # 1 Mbps
                'max_latency': 20,
                'max_jitter': 5,
                'max_packet_loss': 0.001,
                'queue_id': 1,
                'dscp_marking': 46,  # EF
                'traffic_shaping': False,
                'preemption_allowed': True
            },
            'voip': {
                'priority': QoSPriority.CRITICAL,
                'min_bandwidth': 64,    # 64 Kbps
                'max_bandwidth': 128,   # 128 Kbps
                'max_latency': 30,
                'max_jitter': 10,
                'max_packet_loss': 0.001,
                'queue_id': 1,
                'dscp_marking': 46,  # EF
                'traffic_shaping': False,
                'preemption_allowed': True
            },
            'ftp': {
                'priority': QoSPriority.LOW,
                'min_bandwidth': 100,   # 100 Kbps
                'max_bandwidth': 100000,  # 100 Mbps
                'max_latency': 5000,
                'max_jitter': 1000,
                'max_packet_loss': 1.0,
                'queue_id': 4,
                'dscp_marking': 10,  # AF11
                'traffic_shaping': True,
                'burst_allowance': 50000  # KB
            },
            'http_browsing': {
                'priority': QoSPriority.NORMAL,
                'min_bandwidth': 500,   # 500 Kbps
                'max_bandwidth': 10000, # 10 Mbps
                'max_latency': 500,
                'max_jitter': 200,
                'max_packet_loss': 0.5,
                'queue_id': 3,
                'dscp_marking': 0,   # Best effort
                'traffic_shaping': True,
                'burst_allowance': 2000  # KB
            },
            'streaming': {
                'priority': QoSPriority.HIGH,
                'min_bandwidth': 2000,  # 2 Mbps
                'max_bandwidth': 30000, # 30 Mbps
                'max_latency': 200,
                'max_jitter': 100,
                'max_packet_loss': 0.1,
                'queue_id': 2,
                'dscp_marking': 34,  # AF41
                'traffic_shaping': True,
                'burst_allowance': 8000  # KB
            },
            'background': {
                'priority': QoSPriority.BACKGROUND,
                'min_bandwidth': 50,    # 50 Kbps
                'max_bandwidth': 1000,  # 1 Mbps
                'max_latency': 10000,
                'max_jitter': 5000,
                'max_packet_loss': 5.0,
                'queue_id': 5,
                'dscp_marking': 8,   # CS1
                'traffic_shaping': True,
                'burst_allowance': 500   # KB
            }
        }
        
        # OpenFlow action mappings
        self.openflow_actions = {
            'set_queue': self._create_queue_action,
            'set_dscp': self._create_dscp_action,
            'rate_limit': self._create_rate_limit_action,
            'priority_path': self._create_priority_path_action,
            'drop_excessive': self._create_drop_action
        }
        
    def translate_explanation_to_policy(self, explanation: Dict[str, Any],
                                      classification_result: Dict[str, Any],
                                      flow_data: Dict[str, Any]) -> List[QoSPolicy]:
        """
        Main method to translate XAI explanation into QoS policies
        """
        try:
            predicted_class = explanation['classification']['predicted_class']
            confidence = explanation['classification']['confidence']
            qos_requirements = explanation['qos_requirements']
            
            # Get base policy template
            base_template = self.policy_templates.get(predicted_class, 
                                                    self.policy_templates['http_browsing'])
            
            # Adjust policy based on explanation insights
            adjusted_policy = self._adjust_policy_from_explanation(
                base_template, explanation, flow_data
            )
            
            # Create QoS policy object
            qos_policy = QoSPolicy(
                flow_id=flow_data.get('flow_id', 'unknown'),
                priority=adjusted_policy['priority'],
                min_bandwidth=adjusted_policy['min_bandwidth'],
                max_bandwidth=adjusted_policy['max_bandwidth'],
                max_latency=adjusted_policy['max_latency'],
                max_jitter=adjusted_policy['max_jitter'],
                max_packet_loss=adjusted_policy['max_packet_loss'],
                queue_id=adjusted_policy['queue_id'],
                dscp_marking=adjusted_policy['dscp_marking'],
                actions=self._generate_openflow_actions(adjusted_policy, flow_data),
                explanation=self._generate_policy_explanation(explanation, adjusted_policy)
            )
            
            self.logger.debug(f"Generated QoS policy for {predicted_class} flow: {qos_policy}")
            
            return [qos_policy]
            
        except Exception as e:
            self.logger.error(f"Error translating explanation to policy: {e}")
            return []
            
    def _adjust_policy_from_explanation(self, base_template: Dict[str, Any],
                                       explanation: Dict[str, Any],
                                       flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust base policy template based on XAI explanation insights
        """
        adjusted = base_template.copy()
        
        # Get key features from explanation
        key_features = explanation.get('key_features', [])
        confidence = explanation['classification']['confidence']
        
        # Adjust based on confidence level
        if confidence < 0.7:
            # Lower confidence - be more conservative
            adjusted['priority'] = QoSPriority(min(adjusted['priority'].value + 1, 5))
            adjusted['min_bandwidth'] = int(adjusted['min_bandwidth'] * 0.8)
            
        elif confidence > 0.9:
            # High confidence - can be more aggressive with QoS
            if adjusted['priority'].value > 1:
                adjusted['priority'] = QoSPriority(adjusted['priority'].value - 1)
                
        # Adjust based on specific feature insights
        for feature in key_features:
            feature_name = feature.get('name', '').lower()
            feature_value = feature.get('value', 0)
            importance = feature.get('importance', 0)
            
            if 'byte rate' in feature_name and importance > 0.15:
                if feature_value > 10000000:  # > 10 MB/s
                    # High bandwidth flow
                    adjusted['min_bandwidth'] = max(adjusted['min_bandwidth'], 
                                                  int(feature_value / 1000))
                    adjusted['max_bandwidth'] = max(adjusted['max_bandwidth'],
                                                  int(feature_value / 500))
                    
            elif 'packet rate' in feature_name and importance > 0.15:
                if feature_value > 100:  # High packet rate
                    # Likely real-time - reduce latency requirements
                    adjusted['max_latency'] = min(adjusted['max_latency'], 100)
                    adjusted['max_jitter'] = min(adjusted['max_jitter'], 30)
                    
            elif 'duration' in feature_name and importance > 0.15:
                if feature_value > 300:  # Long duration flow
                    # Sustained transfer - may need higher bandwidth guarantee
                    adjusted['min_bandwidth'] = int(adjusted['min_bandwidth'] * 1.2)
                    
        # Apply network congestion considerations
        network_load = self._estimate_network_load()
        if network_load > 0.8:  # High network load
            # Be more conservative with bandwidth allocations
            adjusted['min_bandwidth'] = int(adjusted['min_bandwidth'] * 0.9)
            adjusted['max_bandwidth'] = int(adjusted['max_bandwidth'] * 0.8)
            
        return adjusted
        
    def _generate_openflow_actions(self, policy: Dict[str, Any], 
                                  flow_data: Dict[str, Any]) -> List[str]:
        """
        Generate OpenFlow actions based on policy requirements
        """
        actions = []
        
        # Queue assignment
        actions.append(f"set_queue:{policy['queue_id']}")
        
        # DSCP marking
        if policy['dscp_marking'] > 0:
            actions.append(f"set_dscp:{policy['dscp_marking']}")
            
        # Rate limiting if required
        if policy.get('traffic_shaping', False):
            actions.append(f"rate_limit:{policy['max_bandwidth']}")
            
        # Priority path selection for critical traffic
        if policy['priority'] in [QoSPriority.CRITICAL, QoSPriority.HIGH]:
            actions.append("priority_path:enable")
            
        # Preemption for ultra-critical traffic
        if policy.get('preemption_allowed', False):
            actions.append("preemption:allow")
            
        # Burst allowance for streaming applications
        if policy.get('burst_allowance'):
            actions.append(f"burst_allowance:{policy['burst_allowance']}")
            
        return actions
        
    def _generate_policy_explanation(self, explanation: Dict[str, Any],
                                   policy: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for the policy decision
        """
        predicted_class = explanation['classification']['predicted_class']
        confidence = explanation['classification']['confidence']
        
        policy_explanation = [
            f"Applied {policy['priority'].name.lower()} priority QoS policy for {predicted_class} traffic",
            f"(confidence: {confidence:.1%})."
        ]
        
        # Add specific policy reasoning
        if policy['priority'] == QoSPriority.CRITICAL:
            policy_explanation.append(
                "Critical priority ensures minimal latency and jitter for real-time applications."
            )
            
        if policy.get('traffic_shaping', False):
            policy_explanation.append(
                f"Traffic shaping applied with {policy['max_bandwidth']} Kbps limit to prevent congestion."
            )
            
        if policy['min_bandwidth'] > 1000:
            policy_explanation.append(
                f"Guaranteed minimum bandwidth of {policy['min_bandwidth']} Kbps for quality assurance."
            )
            
        # Add explanation reasoning
        reasoning = explanation.get('reasoning', '')
        if reasoning:
            policy_explanation.append(f"Reasoning: {reasoning}")
            
        return ' '.join(policy_explanation)
        
    def _create_queue_action(self, queue_id: int) -> str:
        """Create OpenFlow queue action"""
        return f"SET_QUEUE:{queue_id}"
        
    def _create_dscp_action(self, dscp_value: int) -> str:
        """Create DSCP marking action"""
        return f"SET_NW_TOS:{dscp_value << 2}"  # DSCP is in upper 6 bits
        
    def _create_rate_limit_action(self, bandwidth_kbps: int) -> str:
        """Create rate limiting action"""
        return f"RATE_LIMIT:{bandwidth_kbps}"
        
    def _create_priority_path_action(self, enable: bool) -> str:
        """Create priority path selection action"""
        return f"PRIORITY_PATH:{'ENABLE' if enable else 'DISABLE'}"
        
    def _create_drop_action(self, threshold: float) -> str:
        """Create packet drop action for excessive traffic"""
        return f"DROP_EXCESSIVE:{threshold}"
        
    def _estimate_network_load(self) -> float:
        """
        Estimate current network load (simplified)
        In real implementation, this would query network statistics
        """
        # Placeholder - would integrate with network monitoring
        return 0.5  # 50% load
        
    def get_policy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available policy templates"""
        return self.policy_templates
        
    def update_policy_template(self, app_type: str, template: Dict[str, Any]):
        """Update policy template for specific application type"""
        self.policy_templates[app_type] = template
        self.logger.info(f"Updated policy template for {app_type}")
        
    def validate_policy(self, policy: QoSPolicy) -> Tuple[bool, List[str]]:
        """Validate policy parameters"""
        errors = []
        
        if policy.min_bandwidth > policy.max_bandwidth:
            errors.append("Minimum bandwidth cannot exceed maximum bandwidth")
            
        if policy.max_latency < 1:
            errors.append("Maximum latency must be positive")
            
        if policy.max_packet_loss < 0 or policy.max_packet_loss > 100:
            errors.append("Packet loss must be between 0 and 100 percent")
            
        if policy.queue_id < 1 or policy.queue_id > 8:
            errors.append("Queue ID must be between 1 and 8")
            
        return len(errors) == 0, errors