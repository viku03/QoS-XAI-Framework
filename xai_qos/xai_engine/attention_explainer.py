# xai_qos/xai_engine/attention_explainer.py

"""
Real-time explainable AI engine that uses attention weights 
to generate human-readable explanations for traffic classification decisions
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import json
from collections import defaultdict

class AttentionExplainer:
    """
    Generates real-time explanations for traffic classification decisions
    using attention weights from the neural network
    """
    
    def __init__(self, classifier, config: Dict[str, Any]):
        self.classifier = classifier
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Explanation thresholds
        self.importance_threshold = config.get('importance_threshold', 0.05)
        self.top_k_features = config.get('top_k_features', 5)
        
        # Feature semantics for human-readable explanations
        self.feature_semantics = {
            'packet_count': {
                'name': 'Packet Count',
                'description': 'Total number of packets in flow',
                'high_indicates': 'Large data transfer or sustained communication'
            },
            'byte_count': {
                'name': 'Byte Count', 
                'description': 'Total bytes transferred in flow',
                'high_indicates': 'High bandwidth usage'
            },
            'duration': {
                'name': 'Flow Duration',
                'description': 'How long the flow lasted',
                'high_indicates': 'Long-running connection or session'
            },
            'avg_packet_size': {
                'name': 'Average Packet Size',
                'description': 'Mean size of packets in flow',
                'high_indicates': 'Large payload packets, potentially media content'
            },
            'packet_rate': {
                'name': 'Packet Rate',
                'description': 'Packets per second',
                'high_indicates': 'High frequency communication'
            },
            'byte_rate': {
                'name': 'Byte Rate',
                'description': 'Bytes per second throughput',
                'high_indicates': 'High bandwidth application'
            },
            'iat_mean': {
                'name': 'Inter-arrival Time Mean',
                'description': 'Average time between packets',
                'high_indicates': 'Sporadic or batch communication pattern'
            },
            'iat_std': {
                'name': 'Inter-arrival Time Variance', 
                'description': 'Variability in packet timing',
                'high_indicates': 'Irregular traffic pattern'
            },
            'protocol': {
                'name': 'Protocol',
                'description': 'Network protocol used',
                'high_indicates': 'Protocol-specific behavior'
            },
            'dst_port': {
                'name': 'Destination Port',
                'description': 'Target service port',
                'high_indicates': 'Specific service or application'
            }
        }
        
        # Application-specific QoS characteristics
        self.app_qos_profiles = {
            'youtube': {
                'latency_sensitive': True,
                'bandwidth_intensive': True,
                'burst_tolerant': True,
                'priority': 'high'
            },
            'netflix': {
                'latency_sensitive': True,
                'bandwidth_intensive': True,  
                'burst_tolerant': True,
                'priority': 'high'
            },
            'zoom': {
                'latency_sensitive': True,
                'bandwidth_intensive': False,
                'burst_tolerant': False,
                'priority': 'critical'
            },
            'teams': {
                'latency_sensitive': True,
                'bandwidth_intensive': False,
                'burst_tolerant': False, 
                'priority': 'critical'
            },
            'ftp': {
                'latency_sensitive': False,
                'bandwidth_intensive': True,
                'burst_tolerant': True,
                'priority': 'low'
            },
            'gaming': {
                'latency_sensitive': True,
                'bandwidth_intensive': False,
                'burst_tolerant': False,
                'priority': 'critical'
            },
            'voip': {
                'latency_sensitive': True,
                'bandwidth_intensive': False,
                'burst_tolerant': False,
                'priority': 'critical'
            },
            'http_browsing': {
                'latency_sensitive': False,
                'bandwidth_intensive': False,
                'burst_tolerant': True,
                'priority': 'normal'
            },
            'background': {
                'latency_sensitive': False,
                'bandwidth_intensive': False,
                'burst_tolerant': True,
                'priority': 'low'
            }
        }
        
    def explain_classification(self, flow_data: Dict[str, Any], 
                             classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for classification decision
        """
        try:
            # Get attention weights and feature importance
            feature_importance = self.classifier.get_feature_importance()
            
            if not feature_importance:
                return self._fallback_explanation(classification_result)
            
            # Generate explanation components
            explanation = {
                'classification': {
                    'predicted_class': classification_result['predicted_class'],
                    'confidence': classification_result['confidence'],
                    'alternative_classes': self._get_alternative_classes(
                        classification_result['class_probabilities']
                    )
                },
                'key_features': self._identify_key_features(feature_importance, flow_data),
                'reasoning': self._generate_reasoning(
                    classification_result['predicted_class'],
                    feature_importance,
                    flow_data
                ),
                'qos_requirements': self._infer_qos_requirements(
                    classification_result['predicted_class']
                ),
                'confidence_factors': self._analyze_confidence_factors(
                    classification_result, feature_importance
                ),
                'explanation_metadata': {
                    'explanation_method': 'attention_weights',
                    'feature_count': len(feature_importance),
                    'top_k_features': self.top_k_features,
                    'timestamp': str(np.datetime64('now'))
                }
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_explanation(classification_result)
            
    def _identify_key_features(self, feature_importance: Dict[str, float], 
                              flow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify the most important features for the decision"""
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        key_features = []
        for feature_name, importance in sorted_features[:self.top_k_features]:
            # Convert numpy types to Python types
            importance = float(importance) if hasattr(importance, 'item') else float(importance)
            
            if importance < self.importance_threshold:
                break
                
            feature_info = self.feature_semantics.get(feature_name, {
                'name': feature_name.replace('_', ' ').title(),
                'description': f'Network feature: {feature_name}',
                'high_indicates': 'Specific traffic characteristic'
            })
            
            # Get actual feature value
            feature_value = self._get_feature_value(feature_name, flow_data)
            
            key_features.append({
                'name': feature_info['name'],
                'importance': float(importance),
                'value': float(feature_value) if feature_value is not None else 0.0,
                'description': feature_info['description'],
                'interpretation': self._interpret_feature_value(
                    feature_name, feature_value, feature_info
                )
            })
            
        return key_features
        
    def _generate_reasoning(self, predicted_class: str, 
                           feature_importance: Dict[str, float],
                           flow_data: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the classification"""
        
        top_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        reasoning_parts = [
            f"This flow was classified as '{predicted_class}' based on the following key indicators:"
        ]
        
        for feature_name, importance in top_features:
            importance = float(importance) if hasattr(importance, 'item') else float(importance)
            
            if importance < self.importance_threshold:
                continue
                
            feature_value = self._get_feature_value(feature_name, flow_data)
            feature_info = self.feature_semantics.get(feature_name, {})
            
            if feature_name == 'dst_port':
                reasoning_parts.append(
                    f"• Destination port {int(feature_value)} is commonly associated with {predicted_class} applications"
                )
            elif feature_name == 'avg_packet_size' or 'packet' in feature_name.lower():
                if feature_value > 1000:
                    reasoning_parts.append(
                        f"• Large average packet size ({feature_value:.0f} bytes) indicates media or file transfer content"
                    )
                else:
                    reasoning_parts.append(
                        f"• Small average packet size ({feature_value:.0f} bytes) suggests control or messaging traffic"
                    )
            elif 'rate' in feature_name.lower():
                if feature_value > 100:
                    reasoning_parts.append(
                        f"• High {feature_name.replace('_', ' ')} ({feature_value:.1f}) indicates real-time or streaming application"
                    )
                else:
                    reasoning_parts.append(
                        f"• Moderate {feature_name.replace('_', ' ')} ({feature_value:.1f}) suggests batch or background transfer"
                    )
            elif feature_name == 'duration':
                if feature_value > 60:
                    reasoning_parts.append(
                        f"• Long flow duration ({feature_value:.1f}s) indicates sustained session typical of {predicted_class}"
                    )
                else:
                    reasoning_parts.append(
                        f"• Short flow duration ({feature_value:.1f}s) suggests quick transaction or setup"
                    )
            else:
                reasoning_parts.append(
                    f"• {feature_info.get('name', feature_name)} pattern matches {predicted_class} characteristics (importance: {importance:.2f})"
                )
                
        return ' '.join(reasoning_parts)
        
    def _infer_qos_requirements(self, predicted_class: str) -> Dict[str, Any]:
        """Infer QoS requirements based on application classification"""
        
        profile = self.app_qos_profiles.get(predicted_class, {
            'latency_sensitive': False,
            'bandwidth_intensive': False, 
            'burst_tolerant': True,
            'priority': 'normal'
        })
        
        qos_requirements = {
            'priority_level': profile['priority'],
            'latency_requirement': 'low' if profile['latency_sensitive'] else 'normal',
            'bandwidth_requirement': 'high' if profile['bandwidth_intensive'] else 'normal', 
            'jitter_tolerance': 'low' if not profile['burst_tolerant'] else 'high',
            'packet_loss_tolerance': 'very_low' if profile['latency_sensitive'] else 'normal'
        }
        
        # Generate specific recommendations
        recommendations = []
        
        if profile['latency_sensitive']:
            recommendations.extend([
                'Prioritize in traffic queues',
                'Use shortest path routing',
                'Consider dedicated bandwidth allocation'
            ])
            
        if profile['bandwidth_intensive']:
            recommendations.extend([
                'Reserve sufficient bandwidth',
                'Monitor for congestion',
                'Enable traffic shaping if needed'
            ])
            
        if not profile['burst_tolerant']:
            recommendations.extend([
                'Implement strict jitter control',
                'Use priority queuing',
                'Monitor packet loss closely'
            ])
            
        qos_requirements['recommendations'] = recommendations
        
        return qos_requirements
        
    def _analyze_confidence_factors(self, classification_result: Dict[str, Any],
                                   feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze factors affecting classification confidence"""
        
        confidence = float(classification_result['confidence'])
        class_probs = classification_result['class_probabilities']
        
        # Find second highest probability
        sorted_probs = sorted(class_probs.values(), reverse=True)
        second_highest = sorted_probs[1] if len(sorted_probs) > 1 else 0
        
        margin = confidence - second_highest
        
        # Analyze confidence factors
        factors = {
            'confidence_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
            'decision_margin': float(margin),
            'uncertainty_sources': [],
            'confidence_drivers': []
        }
        
        # Identify uncertainty sources
        if margin < 0.3:
            factors['uncertainty_sources'].append('Similar probability for multiple classes')
            
        if confidence < 0.7:
            factors['uncertainty_sources'].append('Low overall prediction confidence')
            
        # Identify confidence drivers
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:2]
        
        for feature_name, importance in top_features:
            importance = float(importance) if hasattr(importance, 'item') else float(importance)
            if importance > 0.15:  # High importance threshold
                factors['confidence_drivers'].append(
                    f'Strong {feature_name.replace("_", " ")} signature'
                )
                
        return factors
        
    def _get_alternative_classes(self, class_probabilities: Dict[str, float]) -> List[Dict[str, float]]:
        """Get alternative classification possibilities"""
        
        sorted_classes = sorted(
            class_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[1:4]  # Top 3 alternatives
        
        return [
            {'class': class_name, 'probability': float(prob)}
            for class_name, prob in sorted_classes
            if prob > 0.1  # Only show alternatives with reasonable probability
        ]
        
    def _get_feature_value(self, feature_name: str, flow_data: Dict[str, Any]) -> float:
        """Extract feature value from flow data"""
        
        # Direct mapping for most features
        if feature_name in flow_data:
            value = flow_data[feature_name]
            return float(value) if value is not None else 0.0
            
        # Handle computed features
        if feature_name == 'avg_packet_size':
            packet_count = flow_data.get('packet_count', 1)
            byte_count = flow_data.get('byte_count', 0)
            return float(byte_count) / max(float(packet_count), 1.0)
            
        if feature_name == 'packet_rate':
            packet_count = flow_data.get('packet_count', 0)
            duration = flow_data.get('duration', 1)
            return float(packet_count) / max(float(duration), 0.001)
            
        if feature_name == 'byte_rate':
            byte_count = flow_data.get('byte_count', 0)
            duration = flow_data.get('duration', 1)
            return float(byte_count) / max(float(duration), 0.001)
            
        # Handle statistical features
        if feature_name.startswith('iat_'):
            iat_data = flow_data.get('inter_arrival_times', [])
            if not iat_data:
                return 0.0
                
            if feature_name == 'iat_mean':
                return float(np.mean(iat_data))
            elif feature_name == 'iat_std':
                return float(np.std(iat_data))
            elif feature_name == 'iat_min':
                return float(np.min(iat_data))
            elif feature_name == 'iat_max':
                return float(np.max(iat_data))
                
        return 0.0
        
    def _interpret_feature_value(self, feature_name: str, value: float, 
                                feature_info: Dict[str, str]) -> str:
        """Provide interpretation of feature value"""
        
        if value is None:
            return "Value not available"
            
        value = float(value) if hasattr(value, 'item') else float(value)
        
        if feature_name == 'avg_packet_size' or 'packet_size' in feature_name.lower():
            if value > 1200:
                return "Large packets suggest media content or file transfer"
            elif value < 100:
                return "Small packets indicate control messages or chat"
            else:
                return "Medium packet size suggests mixed content"
                
        elif 'rate' in feature_name.lower():
            if value > 100:
                return "High rate indicates real-time application"
            elif value < 10:
                return "Low rate suggests batch transfer"
            else:
                return "Moderate rate indicates normal application"
                
        elif feature_name == 'duration':
            if value > 300:
                return "Long duration suggests streaming or persistent connection"
            elif value < 10:
                return "Short duration indicates quick transaction"
            else:
                return "Medium duration suggests normal session"
                
        elif feature_name == 'dst_port':
            common_ports = {
                80: "HTTP web traffic",
                443: "HTTPS secure web traffic", 
                22: "SSH secure shell",
                21: "FTP file transfer",
                25: "SMTP email",
                53: "DNS name resolution"
            }
            
            port_desc = common_ports.get(int(value), f"Port {int(value)}")
            return f"Traffic to {port_desc}"
            
        else:
            return f"Value {value:.2f} - {feature_info.get('high_indicates', 'characteristic pattern')}"
            
    def _fallback_explanation(self, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Provide basic explanation when detailed analysis fails"""
        
        return {
            'classification': {
                'predicted_class': classification_result['predicted_class'],
                'confidence': float(classification_result['confidence']),
                'alternative_classes': []
            },
            'key_features': [],
            'reasoning': f"Flow classified as {classification_result['predicted_class']} with {classification_result['confidence']:.1%} confidence based on learned traffic patterns.",
            'qos_requirements': self._infer_qos_requirements(classification_result['predicted_class']),
            'confidence_factors': {
                'confidence_level': 'unknown',
                'decision_margin': 0.0,
                'uncertainty_sources': ['Limited explanation available'],
                'confidence_drivers': []
            },
            'explanation_metadata': {
                'explanation_method': 'fallback',
                'feature_count': 0,
                'top_k_features': 0,
                'timestamp': str(np.datetime64('now'))
            }
        }