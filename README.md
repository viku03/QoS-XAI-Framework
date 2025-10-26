# XAI-QoS Framework Setup and Installation Guide

## Project Structure
```
xai-qos-framework/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── xai_qos_config.yaml
├── xai_qos/
│   ├── __init__.py
│   ├── main.py
│   ├── traffic_classifier/
│   │   ├── __init__.py
│   │   └── attention_classifier.py
│   ├── xai_engine/
│   │   ├── __init__.py
│   │   └── attention_explainer.py
│   ├── sdn_controller/
│   │   ├── __init__.py
│   │   ├── policy_translator.py
│   │   └── ryu_controller.py
│   └── utils/
│       ├── __init__.py
│       └── config_utils.py
├── scripts/
│   └── train_classifier.py
├── data/
│   └── sample_data/
└── docs/
    └── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Mininet (for SDN simulation)
- Ryu SDN Framework

### Step 1: Clone and Setup Environment
```bash
git clone <your-repo-url>
cd xai-qos-framework

# Create virtual environment
python -m venv xai_qos_env
source xai_qos_env/bin/activate  # On Windows: xai_qos_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install Mininet (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install mininet
```

### Step 3: Install Ryu Controller
```bash
pip install ryu
```

## Usage

### Training the Classifier
```bash
# Train on your dataset
python scripts/train_classifier.py --data /path/to/your/dataset.csv --epochs 100

# Use sample data (if available)
python scripts/train_classifier.py --data data/sample_data/traffic_data.csv
```

### Running the Framework
```bash
# Start the XAI-QoS framework
python xai_qos/main.py --config config/xai_qos_config.yaml

# Run with debug logging
python xai_qos/main.py --config config/xai_qos_config.yaml --log-level DEBUG
```

### Testing with Mininet
```bash
# Create a simple network topology for testing
sudo mn --topo single,3 --controller remote,ip=127.0.0.1,port=6633
```

## Configuration

Edit `config/xai_qos_config.yaml` to customize:
- Model architecture parameters
- QoS policy templates  
- SDN controller settings
- Monitoring intervals

## Dataset Format

The framework expects CSV data with the following structure:
```csv
packet_count,byte_count,duration,avg_packet_size,protocol,src_port,dst_port,label
100,50000,10.5,500,6,443,80,youtube
50,25000,5.2,500,6,80,443,http_browsing
...
```

## Key Features

1. **Real-time Traffic Classification**: Uses attention-based neural networks
2. **Explainable AI**: Provides human-readable explanations for classifications
3. **Dynamic Policy Enforcement**: Automatically translates explanations into QoS policies
4. **SDN Integration**: Works with Ryu controller and OpenFlow

## Architecture Overview

The framework consists of four main components:

1. **Traffic Classifier**: Attention-based neural network for fine-grained traffic classification
2. **XAI Explainer**: Generates real-time explanations using attention weights
3. **Policy Translator**: Converts explanations into concrete QoS policies
4. **SDN Controller**: Enforces policies through OpenFlow rules

## Performance

Expected performance metrics:
- Classification accuracy: >95% on standard datasets
- Real-time processing: <10ms per flow
- Policy enforcement latency: <50ms
- Explanation generation: <5ms per classification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{xai_qos_framework,
  title={XAI-QoS: Explainable AI for Dynamic, Fine-Grained QoS Management in Software-Defined Networks},
  author={Your Names},
  journal={Your Journal},
  year={2025}
}
```

## Contact

For questions and support:
- Email: your.email@domain.com  
- Issues: Create a GitHub issue
- Documentation: See docs/ folder