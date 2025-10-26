# xai_qos/main.py
"""
XAI-QoS Framework: Main Application Entry Point
Coordinates all framework components for autonomous network management
"""

import logging
import threading
import time
from typing import Dict, Any
import argparse
from pathlib import Path

from .traffic_classifier.attention_classifier import AttentionTrafficClassifier
from .xai_engine.attention_explainer import AttentionExplainer  
from .sdn_controller.policy_translator import PolicyTranslator
from .sdn_controller.ryu_controller import XAIQoSController
from .utils.config_utils import ConfigManager, setup_logging


class XAIQoSFramework:
    """
    Main XAI-QoS Framework orchestrator that coordinates all components
    """
    
    def __init__(self, config_path: str):
        self.config = ConfigManager.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.traffic_classifier = None
        self.xai_explainer = None
        self.policy_translator = None
        self.sdn_controller = None
        
        # Framework state
        self.running = False
        self.stats = {
            'classifications': 0,
            'explanations': 0,
            'policies_enforced': 0,
            'start_time': None
        }
        
    def initialize_components(self):
        """Initialize all framework components"""
        try:
            self.logger.info("Initializing XAI-QoS Framework components...")
            
            # Initialize traffic classifier with attention mechanism
            self.traffic_classifier = AttentionTrafficClassifier(
                model_config=self.config['classifier'],
                device=self.config.get('device', 'cpu')
            )
            
            # Initialize XAI explainer 
            self.xai_explainer = AttentionExplainer(
                classifier=self.traffic_classifier,
                config=self.config['explainer']
            )
            
            # Initialize policy translator
            self.policy_translator = PolicyTranslator(
                config=self.config['policy_translator'],
                qos_policies=self.config['qos_policies']
            )
            
            # Initialize SDN controller
            self.sdn_controller = XAIQoSController(
                config=self.config['sdn_controller'],
                policy_translator=self.policy_translator
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start_framework(self):
        """Start the XAI-QoS framework"""
        if self.running:
            self.logger.warning("Framework already running")
            return
            
        try:
            self.logger.info("Starting XAI-QoS Framework...")
            
            # Load pre-trained models
            if self.config['classifier'].get('model_path'):
                self.traffic_classifier.load_model(
                    self.config['classifier']['model_path']
                )
            
            # Start SDN controller in separate thread
            controller_thread = threading.Thread(
                target=self.sdn_controller.start,
                daemon=True
            )
            controller_thread.start()
            
            # Start main processing loop
            processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True  
            )
            processing_thread.start()
            
            self.running = True
            self.stats['start_time'] = time.time()
            
            self.logger.info("XAI-QoS Framework started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start framework: {e}")
            raise
    
    def stop_framework(self):
        """Stop the XAI-QoS framework"""
        if not self.running:
            return
            
        self.logger.info("Stopping XAI-QoS Framework...")
        self.running = False
        
        if self.sdn_controller:
            self.sdn_controller.stop()
            
        self._print_stats()
        self.logger.info("XAI-QoS Framework stopped")
        
    def _processing_loop(self):
        """Main processing loop for traffic analysis and policy enforcement"""
        self.logger.info("Starting traffic processing loop...")
        
        while self.running:
            try:
                # Get network traffic data from SDN controller
                traffic_data = self.sdn_controller.get_current_traffic()
                
                if not traffic_data:
                    time.sleep(0.1)  # Short sleep if no traffic
                    continue
                
                # Process each flow
                for flow_data in traffic_data:
                    self._process_flow(flow_data)
                    
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
                
    def _process_flow(self, flow_data: Dict[str, Any]):
        """Process individual network flow through XAI-QoS pipeline"""
        try:
            # Step 1: Classify traffic using attention-based model
            classification_result = self.traffic_classifier.classify_flow(flow_data)
            self.stats['classifications'] += 1
            
            # Step 2: Generate real-time explanation using attention weights
            explanation = self.xai_explainer.explain_classification(
                flow_data, 
                classification_result
            )
            self.stats['explanations'] += 1
            
            # Step 3: Translate explanation to QoS policies
            qos_policies = self.policy_translator.translate_explanation_to_policy(
                explanation,
                classification_result,
                flow_data
            )
            
            # Step 4: Enforce policies via SDN controller
            if qos_policies:
                self.sdn_controller.enforce_policies(qos_policies, flow_data)
                self.stats['policies_enforced'] += 1
                
                self.logger.debug(
                    f"Flow {flow_data.get('flow_id', 'unknown')} classified as "
                    f"{classification_result['predicted_class']} with confidence "
                    f"{classification_result['confidence']:.3f}, policies enforced"
                )
            
        except Exception as e:
            self.logger.error(f"Error processing flow: {e}")
            
    def _print_stats(self):
        """Print framework statistics"""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            self.logger.info("=== XAI-QoS Framework Statistics ===")
            self.logger.info(f"Runtime: {runtime:.2f} seconds")
            self.logger.info(f"Classifications: {self.stats['classifications']}")
            self.logger.info(f"Explanations: {self.stats['explanations']}")  
            self.logger.info(f"Policies Enforced: {self.stats['policies_enforced']}")
            
            if runtime > 0:
                self.logger.info(
                    f"Avg. Classifications/sec: {self.stats['classifications']/runtime:.2f}"
                )
                
    def get_status(self) -> Dict[str, Any]:
        """Get current framework status"""
        return {
            'running': self.running,
            'stats': self.stats.copy(),
            'components': {
                'classifier': self.traffic_classifier is not None,
                'explainer': self.xai_explainer is not None,
                'policy_translator': self.policy_translator is not None,
                'sdn_controller': self.sdn_controller is not None
            }
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='XAI-QoS Framework')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/xai_qos_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create and initialize framework
        framework = XAIQoSFramework(args.config)
        framework.initialize_components()
        
        # Start framework
        framework.start_framework()
        
        logger.info("XAI-QoS Framework running. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Framework error: {e}")
    finally:
        if 'framework' in locals():
            framework.stop_framework()


if __name__ == "__main__":
    main()
