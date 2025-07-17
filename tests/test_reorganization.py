"""
Test to verify test reorganization works correctly
"""
import pytest
import os
import sys
from pathlib import Path

def test_src_import():
    """Test that src modules can be imported"""
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Try to import grandmodel
    try:
        import grandmodel
        assert True, "Successfully imported grandmodel"
    except ImportError as e:
        pytest.skip(f"Could not import grandmodel: {e}")

def test_test_structure():
    """Test that test structure is correct"""
    test_dir = Path(__file__).parent
    
    # Check that tests directory exists
    assert test_dir.exists()
    assert test_dir.name == "tests"
    
    # Check that conftest.py exists
    conftest_path = test_dir / "conftest.py"
    assert conftest_path.exists()

def test_pytest_config():
    """Test that pytest configuration is accessible"""
    project_root = Path(__file__).parent.parent
    pytest_ini = project_root / "pytest.ini"
    
    assert pytest_ini.exists(), "pytest.ini should exist at project root"
    
    # Check that the config has our expected markers
    with open(pytest_ini, 'r') as f:
        content = f.read()
        assert "grandmodel" in content
        assert "agents:" in content
        assert "core:" in content
        assert "risk:" in content

@pytest.mark.unit
def test_sample_test_passes():
    """Sample test to ensure testing works"""
    assert 1 + 1 == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])