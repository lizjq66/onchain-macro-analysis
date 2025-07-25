# Contributing to OnChain Macro Analysis

Thank you for your interest in contributing to OnChain Macro Analysis! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Environment Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/yourusername/onchain-macro-analysis.git
cd onchain-macro-analysis
```

2. **Set up development environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run tests to ensure everything works**
```bash
python test_pipeline.py
python test_network_analysis.py
```

## 🎯 How to Contribute

### 🐞 Bug Reports
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide system information (OS, Python version)
- Include relevant log outputs

### 💡 Feature Requests
- Describe the feature and its use case
- Explain why it would be valuable
- Consider implementation complexity

### 🔧 Code Contributions

#### Areas for Contribution
1. **Data Sources Integration**
   - New blockchain APIs
   - Additional macro indicators
   - Alternative data feeds

2. **Analysis Algorithms**
   - New centrality metrics
   - Risk assessment improvements
   - Machine learning models

3. **Visualization Enhancements**
   - New chart types
   - Interactive features
   - Dashboard improvements

4. **Performance Optimizations**
   - Data processing efficiency
   - Caching improvements
   - Memory usage optimization

### 📝 Pull Request Process

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
```bash
python test_pipeline.py
python -m pytest tests/
```

4. **Commit with clear messages**
```bash
git commit -m "Add: New centrality metric for protocol importance"
```

5. **Push and create pull request**
```bash
git push origin feature/your-feature-name
```

## 📋 Code Style Guidelines

### Python Code Style
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Maximum line length: 100 characters

### Example Function Documentation
```python
def analyze_network_centrality(self, network: nx.Graph) -> Dict[str, Dict[str, float]]:
    """
    Calculate centrality metrics for all nodes in the network.
    
    Args:
        network: NetworkX graph representing protocol relationships
        
    Returns:
        Dictionary mapping protocol names to centrality metrics
        
    Raises:
        ValueError: If network is empty or invalid
    """
```

### Import Organization
```python
# Standard library imports
import asyncio
from datetime import datetime
from typing import Dict, List

# Third-party imports
import pandas as pd
import networkx as nx
import streamlit as st

# Local imports
from src.utils.logger import setup_logger
from src.data.collectors.base_collector import BaseCollector
```

## 🧪 Testing Guidelines

### Test Structure
- Unit tests in `tests/` directory
- Integration tests for data pipeline
- End-to-end tests for dashboards

### Writing Tests
```python
import pytest
from src.analysis.network_analysis.protocol_network import ProtocolNetworkAnalyzer

class TestProtocolNetworkAnalyzer:
    def test_centrality_calculation(self):
        analyzer = ProtocolNetworkAnalyzer()
        # Test implementation
        assert result is not None
```

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_network_analysis.py

# Run with coverage
python -m pytest --cov=src
```

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions
- Include type hints
- Explain complex algorithms
- Add usage examples

### README Updates
- Update feature list for new capabilities
- Add screenshots for UI changes
- Update installation instructions if needed

## 🏗️ Architecture Guidelines

### Project Structure
```
src/
├── data/           # Data collection and processing
├── analysis/       # Core analysis algorithms  
├── visualization/  # Dashboard and charts
└── utils/         # Shared utilities
```

### Design Principles
1. **Modularity** - Keep components loosely coupled
2. **Extensibility** - Easy to add new data sources
3. **Performance** - Optimize for large datasets
4. **Reliability** - Handle errors gracefully

## 🔍 Review Process

### What We Look For
- ✅ Code follows style guidelines
- ✅ Tests are included and passing  
- ✅ Documentation is updated
- ✅ Feature addresses a real need
- ✅ No breaking changes without discussion

### Review Timeline
- Initial review within 48 hours
- Feedback provided within 1 week
- Final decision within 2 weeks

## 🎖️ Recognition

Contributors will be:
- Added to the contributors list
- Mentioned in release notes
- Credited in documentation

## 📞 Getting Help

- **GitHub Issues** - For bugs and feature requests
- **Discussions** - For questions and general discussion
- **Discord** - Real-time chat (link in README)

## 📋 Development Roadmap

### Short Term (1-2 months)
- [ ] Additional centrality metrics
- [ ] More data source integrations
- [ ] Performance optimizations
- [ ] Mobile-responsive dashboards

### Medium Term (3-6 months)  
- [ ] Machine learning predictions
- [ ] Real-time alerts system
- [ ] API endpoints for external access
- [ ] Advanced visualization options

### Long Term (6+ months)
- [ ] Multi-language support
- [ ] Cloud deployment options
- [ ] Enterprise features
- [ ] Academic research tools

## 🤝 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Celebrate diverse perspectives

### Unacceptable Behavior
- Harassment or discrimination
- Spam or off-topic content
- Sharing others' private information
- Any illegal activities

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
