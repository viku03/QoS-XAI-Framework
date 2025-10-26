# xai_qos/traffic_classifier/attention_classifier.py
"""
Attention-based Neural Network for Fine-grained Traffic Classification
Supports multi-class application-specific traffic categorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for traffic feature analysis"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()

        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(context)
        output = self.layer_norm(output + x)

        # Average across heads -> [batch, seq_len, seq_len]
        # For seq_len=1 this becomes [batch, 1, 1] → scalar after squeeze
        avg_attention = attention_weights.mean(dim=1).squeeze()

        return output, avg_attention


class TrafficFeatureExtractor(nn.Module):
    """Extract features from network flow statistics"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_layers(x)


class AttentionTrafficClassifier(nn.Module):
    """
    Attention-based traffic classifier for fine-grained application categorization
    """

    def __init__(self, model_config: Dict[str, Any], device: str = "cpu"):
        super().__init__()
        self.config = model_config
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Model architecture parameters
        self.input_dim = model_config.get("input_dim", 50)
        self.hidden_dim = model_config.get("hidden_dim", 128)
        self.n_heads = model_config.get("n_heads", 8)
        self.n_layers = model_config.get("n_layers", 3)
        self.n_classes = model_config.get("n_classes", 10)
        self.dropout = model_config.get("dropout", 0.1)

        # Traffic classes mapping
        self.class_names = model_config.get(
            "class_names",
            [
                "youtube",
                "netflix",
                "zoom",
                "teams",
                "ftp",
                "http_browsing",
                "gaming",
                "voip",
                "streaming",
                "background",
            ],
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Feature extractor
        self.feature_extractor = TrafficFeatureExtractor(self.input_dim, self.hidden_dim)

        # Attention layers over the reduced representation
        self.attention_layers = nn.ModuleList(
            [
                MultiHeadAttention(self.hidden_dim // 2, self.n_heads, self.dropout)
                for _ in range(self.n_layers)
            ]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 4, self.n_classes),
        )

        # Feature scaler for normalization
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        # Stores for explainability
        self.last_attention_weights = None           # raw attention output (may be scalar)
        self.last_features_vector = None             # last normalized input vector (np.ndarray)

        self.to(device)

    def extract_traffic_features(self, flow_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from flow data - handles both synthetic and CICIDS format"""
        
        # If flow_data contains CICIDS column names, use them directly as a list
        # Check if this is CICIDS format (has column names like 'Flow Duration')
        if 'Flow Duration' in flow_data or 'Total Fwd Packets' in flow_data:
            # CICIDS format - extract features in order matching SELECTED_FEATURES
            features = []
            
            # Map CICIDS columns to features (matching your preprocessing)
            cicids_feature_map = {
                'Flow Duration': flow_data.get('Flow Duration', 0),
                'Total Fwd Packets': flow_data.get('Total Fwd Packets', 0),
                'Total Backward Packets': flow_data.get('Total Backward Packets', 0),
                'Total Length of Fwd Packets': flow_data.get('Total Length of Fwd Packets', 0),
                'Total Length of Bwd Packets': flow_data.get('Total Length of Bwd Packets', 0),
                'Fwd Packet Length Max': flow_data.get('Fwd Packet Length Max', 0),
                'Fwd Packet Length Min': flow_data.get('Fwd Packet Length Min', 0),
                'Fwd Packet Length Mean': flow_data.get('Fwd Packet Length Mean', 0),
                'Fwd Packet Length Std': flow_data.get('Fwd Packet Length Std', 0),
                'Bwd Packet Length Max': flow_data.get('Bwd Packet Length Max', 0),
                'Bwd Packet Length Min': flow_data.get('Bwd Packet Length Min', 0),
                'Bwd Packet Length Mean': flow_data.get('Bwd Packet Length Mean', 0),
                'Bwd Packet Length Std': flow_data.get('Bwd Packet Length Std', 0),
                'Flow Bytes/s': flow_data.get('Flow Bytes/s', 0),
                'Flow Packets/s': flow_data.get('Flow Packets/s', 0),
                'Flow IAT Mean': flow_data.get('Flow IAT Mean', 0),
                'Flow IAT Std': flow_data.get('Flow IAT Std', 0),
                'Flow IAT Max': flow_data.get('Flow IAT Max', 0),
                'Flow IAT Min': flow_data.get('Flow IAT Min', 0),
                'Fwd IAT Total': flow_data.get('Fwd IAT Total', 0),
                'Fwd IAT Mean': flow_data.get('Fwd IAT Mean', 0),
                'Fwd IAT Std': flow_data.get('Fwd IAT Std', 0),
                'Fwd IAT Max': flow_data.get('Fwd IAT Max', 0),
                'Fwd IAT Min': flow_data.get('Fwd IAT Min', 0),
                'Bwd IAT Total': flow_data.get('Bwd IAT Total', 0),
                'Bwd IAT Mean': flow_data.get('Bwd IAT Mean', 0),
                'Bwd IAT Std': flow_data.get('Bwd IAT Std', 0),
                'Bwd IAT Max': flow_data.get('Bwd IAT Max', 0),
                'Bwd IAT Min': flow_data.get('Bwd IAT Min', 0),
                'Fwd PSH Flags': flow_data.get('Fwd PSH Flags', 0),
                'Bwd PSH Flags': flow_data.get('Bwd PSH Flags', 0),
                'Fwd URG Flags': flow_data.get('Fwd URG Flags', 0),
                'Bwd URG Flags': flow_data.get('Bwd URG Flags', 0),
                'Fwd Header Length': flow_data.get('Fwd Header Length', 0),
                'Bwd Header Length': flow_data.get('Bwd Header Length', 0),
                'Fwd Packets/s': flow_data.get('Fwd Packets/s', 0),
                'Bwd Packets/s': flow_data.get('Bwd Packets/s', 0),
                'Min Packet Length': flow_data.get('Min Packet Length', 0),
                'Max Packet Length': flow_data.get('Max Packet Length', 0),
                'Packet Length Mean': flow_data.get('Packet Length Mean', 0),
                'Packet Length Std': flow_data.get('Packet Length Std', 0),
                'Packet Length Variance': flow_data.get('Packet Length Variance', 0),
                'FIN Flag Count': flow_data.get('FIN Flag Count', 0),
                'SYN Flag Count': flow_data.get('SYN Flag Count', 0),
                'RST Flag Count': flow_data.get('RST Flag Count', 0),
                'PSH Flag Count': flow_data.get('PSH Flag Count', 0),
                'ACK Flag Count': flow_data.get('ACK Flag Count', 0),
                'URG Flag Count': flow_data.get('URG Flag Count', 0),
                'CWE Flag Count': flow_data.get('CWE Flag Count', 0),
                'ECE Flag Count': flow_data.get('ECE Flag Count', 0),
            }
            
            features = list(cicids_feature_map.values())
            
        else:
            # Synthetic format (your original code for test scenarios)
            features = []
            
            # Basic flow statistics
            features.extend([
                flow_data.get("packet_count", 0),
                flow_data.get("byte_count", 0),
                flow_data.get("duration", 0),
                flow_data.get("avg_packet_size", 0),
                flow_data.get("packet_rate", 0),
                flow_data.get("byte_rate", 0),
            ])
            
            # Rest of synthetic feature extraction...
            # (keep your existing code for the synthetic case)
            features.extend([0] * (self.input_dim - len(features)))  # Pad to input_dim
        
        # Pad or truncate to expected input dimension
        if len(features) < self.input_dim:
            features.extend([0] * (self.input_dim - len(features)))
        else:
            features = features[:self.input_dim]
        
        return np.array(features, dtype=np.float32)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the attention-based classifier"""
        # x: [batch, input_dim]
        feats = self.feature_extractor(x)           # [batch, hidden_dim//2]
        feats_seq = feats.unsqueeze(1)              # [batch, 1, hidden_dim//2]

        attention_weights_list = []
        for attention_layer in self.attention_layers:
            feats_seq, att_w = attention_layer(feats_seq)   # att_w may be scalar for seq_len=1
            attention_weights_list.append(att_w)

        # Remove sequence dimension
        feats_final = feats_seq.squeeze(1)          # [batch, hidden_dim//2]
        logits = self.classifier(feats_final)

        # Store attention weights (may be scalar/1x1)
        self.last_attention_weights = attention_weights_list

        # Average across layers for a combined tensor
        combined_attention = None
        try:
            stacked = torch.stack(
                [w if isinstance(w, torch.Tensor) else torch.as_tensor(w) for w in attention_weights_list],
                dim=0,
            )
            combined_attention = stacked.mean(dim=0)
        except Exception:
            # Fallback to a scalar tensor
            combined_attention = torch.tensor(1.0, device=x.device)

        return logits, combined_attention

    def classify_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single network flow"""
        self.eval()

        try:
            # Extract features
            raw_features = self.extract_traffic_features(flow_data).reshape(1, -1)
            
            # CRITICAL: Must use the scaler from training!
            if not self.scaler_fitted:
                raise RuntimeError(
                    "Scaler not fitted! Model must be loaded with load_model() first."
                )
            
            # Transform using TRAINING scaler
            features = self.scaler.transform(raw_features)
            
            # Persist BOTH raw and normalized for explainability
            self.last_features_vector_normalized = features.flatten()  # For model
            self.last_features_vector_raw = raw_features.flatten()     # For display!

            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                logits, attention_weights = self(x)
                probabilities = F.softmax(logits, dim=1)

            predicted_class_idx = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0, predicted_class_idx].item())
            predicted_class = self.class_names[predicted_class_idx]

            class_probabilities = {
                self.class_names[i]: float(probabilities[0, i].item())
                for i in range(self.n_classes)
            }

            return {
                "predicted_class": predicted_class,
                "predicted_class_idx": predicted_class_idx,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "attention_weights": attention_weights.detach().cpu().numpy()
                    if isinstance(attention_weights, torch.Tensor) else np.array(attention_weights),
                "features": self.last_features_vector_normalized.copy(),
                "features_raw": self.last_features_vector_raw.copy(),  # Add raw features!
            }

        except Exception as e:
            self.logger.error(f"Error in flow classification: {e}")
            import traceback
            traceback.print_exc()
            raise



    def load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            # Explicitly set weights_only=False to quiet future warning and keep current behavior
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint["model_state_dict"])

            if "scaler" in checkpoint and checkpoint["scaler"] is not None:
                self.scaler = checkpoint["scaler"]
                self.scaler_fitted = True

            # Restore class names if present
            if "class_names" in checkpoint and checkpoint["class_names"]:
                self.class_names = checkpoint["class_names"]

            self.logger.info(f"Model loaded successfully from {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def save_model(self, model_path: str):
        """Save trained model"""
        try:
            checkpoint = {
                "model_state_dict": self.state_dict(),
                "config": self.config,
                "scaler": self.scaler if self.scaler_fitted else None,
                "class_names": self.class_names,
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"Model saved successfully to {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to save model to {model_path}: {e}")
            raise

    def _feature_name_list(self) -> list:
        """Build feature name list of length input_dim"""
        names = [
            "packet_count",
            "byte_count",
            "duration",
            "avg_packet_size",
            "packet_rate",
            "byte_rate",
            "iat_mean",
            "iat_std",
            "iat_min",
            "iat_max",
            "iat_median",
            "pkt_size_mean",
            "pkt_size_std",
            "pkt_size_min",
            "pkt_size_max",
            "pkt_size_median",
            "protocol",
            "src_port",
            "dst_port",
            "syn_flag",
            "ack_flag",
            "fin_flag",
            "rst_flag",
            "psh_flag",
            "forward_packets",
            "backward_packets",
            "forward_bytes",
            "backward_bytes",
            "flow_active_time",
            "flow_idle_mean",
            "flow_idle_std",
            "flow_idle_min",
            "flow_idle_max",
        ]
        while len(names) < self.input_dim:
            names.append(f"feature_{len(names)}")
        return names[: self.input_dim]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get per-input feature importance.
        Strategy:
          1) Try to use combined attention if it is a proper vector of size input_dim
          2) Otherwise, fall back to a weight×input proxy using the first linear layer
        """
        names = self._feature_name_list()

        # 1) Try attention: turn last_attention_weights into a vector
        att_vec = None
        try:
            att = self.last_attention_weights
            # Average list of tensors/arrays to numpy
            if isinstance(att, list):
                arrs = []
                for w in att:
                    if isinstance(w, torch.Tensor):
                        arrs.append(w.detach().cpu().numpy())
                    else:
                        arrs.append(np.array(w))
                att = np.mean(arrs, axis=0)
            elif isinstance(att, torch.Tensor):
                att = att.detach().cpu().numpy()
            else:
                att = np.array(att)

            # If attention has extra dims, squeeze
            att = np.squeeze(att)

            # If it matches input_dim, accept it
            if att.ndim == 1 and att.size == self.input_dim:
                att_vec = att.astype(np.float64)
        except Exception:
            att_vec = None

        # 2) Fallback: weight×input importance
        if att_vec is None:
            first_linear = None
            for layer in self.feature_extractor.feature_layers:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break

            if first_linear is None:
                return {}

            W = first_linear.weight.detach().cpu().numpy()  
            w_importance = np.mean(np.abs(W), axis=0)

            # Use NORMALIZED features for importance calculation
            x = getattr(self, "last_features_vector_normalized", None)
            if x is not None:
                x = np.abs(np.array(x).reshape(-1))
                imp = w_importance * x
            else:
                imp = w_importance

            att_vec = imp

        # Normalize to distribution
        att_vec = np.maximum(att_vec, 1e-12)
        att_vec = att_vec / att_vec.sum()

        # Use RAW features for display values!
        raw_features = getattr(self, "last_features_vector_raw", np.zeros(self.input_dim))

        return {
            name: {
                'importance': float(score),
                'value': float(raw_features[i])  # Show RAW values!
            }
            for i, (name, score) in enumerate(zip(names, att_vec))
        }
