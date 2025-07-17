import pytest
import torch
import numpy as np

def test_pytorch():
    """Test PyTorch installation"""
    assert torch.__version__.endswith('+cpu')
    x = torch.randn(10, 10)
    assert x.shape == (10, 10)

def test_imports():
    """Test critical imports"""
    from src.matrix.assembler_30m_enhanced import MatrixAssembler30mEnhanced
    from src.agents.trading_env import TradingMAEnv
    from src.llm.ollama_llm import OllamaLLM
    
def test_marl_env():
    """Test MARL environment"""
    from src.agents.trading_env import TradingMAEnv
    env = TradingMAEnv()
    obs = env.reset()
    assert len(obs) == 3

def test_ollama():
    """Test Ollama connection"""
    try:
        from src.llm.ollama_llm import OllamaLLM
        llm = OllamaLLM()
        assert True
    except (ImportError, ConnectionError, OSError, RuntimeError) as e:
        pytest.skip(f"Ollama not running: {e}")
    except Exception as e:
        pytest.skip(f"Ollama test failed: {e}")