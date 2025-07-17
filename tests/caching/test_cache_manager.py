"""
Test caching and result management system for maximum efficiency.
Agent 4 Mission: Test Data Management & Caching System
"""
import os
import json
import time
import hashlib
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pytest
from unittest.mock import Mock, patch

@dataclass
class TestResult:
    """Container for test result data."""
    test_id: str
    outcome: str
    duration: float
    timestamp: datetime
    file_hash: str
    dependencies: List[str]
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestResult':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class TestCacheManager:
    """Advanced test result caching system."""
    
    def __init__(self, cache_dir: str = ".pytest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.results_cache = self.cache_dir / "test_results.json"
        self.file_checksums = self.cache_dir / "file_checksums.json"
        self.dependency_graph = self.cache_dir / "dependency_graph.json"
        
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached results from disk."""
        try:
            if self.results_cache.exists():
                with open(self.results_cache, 'r') as f:
                    data = json.load(f)
                    self.results = {
                        k: TestResult.from_dict(v) for k, v in data.items()
                    }
            else:
                self.results = {}
                
            if self.file_checksums.exists():
                with open(self.file_checksums, 'r') as f:
                    self.checksums = json.load(f)
            else:
                self.checksums = {}
                
            if self.dependency_graph.exists():
                with open(self.dependency_graph, 'r') as f:
                    self.dependencies = json.load(f)
            else:
                self.dependencies = {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.results = {}
            self.checksums = {}
            self.dependencies = {}
    
    def _save_cache(self) -> None:
        """Save cached results to disk."""
        try:
            with open(self.results_cache, 'w') as f:
                json.dump(
                    {k: v.to_dict() for k, v in self.results.items()},
                    f, indent=2
                )
            
            with open(self.file_checksums, 'w') as f:
                json.dump(self.checksums, f, indent=2)
            
            with open(self.dependency_graph, 'w') as f:
                json.dump(self.dependencies, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def has_file_changed(self, file_path: str) -> bool:
        """Check if file has changed since last run."""
        current_hash = self.get_file_hash(file_path)
        if file_path not in self.checksums:
            self.checksums[file_path] = current_hash
            return True
        
        changed = self.checksums[file_path] != current_hash
        if changed:
            self.checksums[file_path] = current_hash
        return changed
    
    def should_run_test(self, test_id: str, test_file: str) -> bool:
        """Determine if test should be run based on cache."""
        # Always run if no cached result
        if test_id not in self.results:
            return True
        
        # Run if test file changed
        if self.has_file_changed(test_file):
            return True
        
        # Run if any dependency changed
        if test_id in self.dependencies:
            for dep in self.dependencies[test_id]:
                if self.has_file_changed(dep):
                    return True
        
        # Run if result is too old (configurable)
        cached_result = self.results[test_id]
        if datetime.now() - cached_result.timestamp > timedelta(hours=24):
            return True
        
        # Run if previous result was failure
        if cached_result.outcome != "passed":
            return True
        
        return False
    
    def store_result(self, test_id: str, outcome: str, duration: float, 
                    test_file: str, dependencies: List[str] = None) -> None:
        """Store test result in cache."""
        if dependencies is None:
            dependencies = []
        
        result = TestResult(
            test_id=test_id,
            outcome=outcome,
            duration=duration,
            timestamp=datetime.now(),
            file_hash=self.get_file_hash(test_file),
            dependencies=dependencies
        )
        
        self.results[test_id] = result
        self.dependencies[test_id] = dependencies
        self._save_cache()
    
    def get_cached_result(self, test_id: str) -> Optional[TestResult]:
        """Get cached result if available."""
        return self.results.get(test_id)
    
    def invalidate_cache(self, pattern: str = None) -> None:
        """Invalidate cache entries matching pattern."""
        if pattern is None:
            self.results.clear()
            self.checksums.clear()
            self.dependencies.clear()
        else:
            keys_to_remove = [k for k in self.results.keys() if pattern in k]
            for key in keys_to_remove:
                del self.results[key]
                if key in self.dependencies:
                    del self.dependencies[key]
        self._save_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.outcome == "passed")
        failed_tests = total_tests - passed_tests
        
        if self.results:
            avg_duration = sum(r.duration for r in self.results.values()) / total_tests
            total_duration = sum(r.duration for r in self.results.values())
        else:
            avg_duration = 0
            total_duration = 0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "avg_duration": avg_duration,
            "total_duration": total_duration,
            "cache_size": len(self.checksums)
        }

class IncrementalTestRunner:
    """Runs tests incrementally based on code changes."""
    
    def __init__(self, cache_manager: TestCacheManager):
        self.cache_manager = cache_manager
        self.src_dir = Path("src")
        self.test_dir = Path("tests")
    
    def get_test_dependencies(self, test_file: str) -> List[str]:
        """Analyze test file to determine dependencies."""
        dependencies = []
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Simple dependency analysis based on imports
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('from src.') or line.startswith('import src.'):
                    # Extract module path
                    if 'from src.' in line:
                        module = line.split('from src.')[1].split(' import')[0]
                    else:
                        module = line.split('import src.')[1].split(' ')[0]
                    
                    # Convert to file path
                    file_path = self.src_dir / f"{module.replace('.', '/')}.py"
                    if file_path.exists():
                        dependencies.append(str(file_path))
        
        except Exception as e:
            print(f"Error analyzing dependencies for {test_file}: {e}")
        
        return dependencies
    
    def get_changed_files(self) -> List[str]:
        """Get list of files that have changed."""
        changed_files = []
        
        # Check all source files
        for file_path in self.src_dir.rglob("*.py"):
            if self.cache_manager.has_file_changed(str(file_path)):
                changed_files.append(str(file_path))
        
        # Check all test files
        for file_path in self.test_dir.rglob("test_*.py"):
            if self.cache_manager.has_file_changed(str(file_path)):
                changed_files.append(str(file_path))
        
        return changed_files
    
    def get_tests_to_run(self) -> List[str]:
        """Get list of tests that need to be run."""
        tests_to_run = []
        changed_files = self.get_changed_files()
        
        # Find all test files
        for test_file in self.test_dir.rglob("test_*.py"):
            test_id = str(test_file.relative_to(self.test_dir))
            dependencies = self.get_test_dependencies(str(test_file))
            
            should_run = False
            
            # Check if test file itself changed
            if str(test_file) in changed_files:
                should_run = True
            
            # Check if any dependency changed
            for dep in dependencies:
                if dep in changed_files:
                    should_run = True
                    break
            
            # Check cache manager decision
            if self.cache_manager.should_run_test(test_id, str(test_file)):
                should_run = True
            
            if should_run:
                tests_to_run.append(str(test_file))
        
        return tests_to_run

# Global cache manager instance
cache_manager = TestCacheManager()
incremental_runner = IncrementalTestRunner(cache_manager)

# Pytest fixtures for caching
@pytest.fixture(scope="session")
def test_cache_manager():
    """Provide test cache manager."""
    return cache_manager

@pytest.fixture(scope="session") 
def incremental_test_runner():
    """Provide incremental test runner."""
    return incremental_runner

# Test class for the caching system
class TestCacheManagerTests:
    """Tests for the test cache manager."""
    
    def test_cache_initialization(self):
        """Test cache manager initialization."""
        cache_dir = Path(".pytest_cache_test")
        manager = TestCacheManager(str(cache_dir))
        
        assert manager.cache_dir.exists()
        assert isinstance(manager.results, dict)
        assert isinstance(manager.checksums, dict)
        assert isinstance(manager.dependencies, dict)
    
    def test_file_hash_calculation(self):
        """Test file hash calculation."""
        manager = TestCacheManager()
        
        # Create temporary test file
        test_file = Path("test_temp.py")
        test_file.write_text("print('hello')")
        
        hash1 = manager.get_file_hash(str(test_file))
        hash2 = manager.get_file_hash(str(test_file))
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 length
        
        # Change file content
        test_file.write_text("print('world')")
        hash3 = manager.get_file_hash(str(test_file))
        
        assert hash1 != hash3
        
        # Cleanup
        test_file.unlink()
    
    def test_result_storage_and_retrieval(self):
        """Test storing and retrieving test results."""
        manager = TestCacheManager()
        
        test_id = "test_example"
        outcome = "passed"
        duration = 0.5
        test_file = __file__
        dependencies = ["src/core/config.py"]
        
        manager.store_result(test_id, outcome, duration, test_file, dependencies)
        
        result = manager.get_cached_result(test_id)
        assert result is not None
        assert result.test_id == test_id
        assert result.outcome == outcome
        assert result.duration == duration
        assert result.dependencies == dependencies
    
    def test_should_run_test_logic(self):
        """Test the logic for determining if test should run."""
        manager = TestCacheManager()
        
        # Test should run if not cached
        assert manager.should_run_test("new_test", __file__) == True
        
        # Store a passing result
        manager.store_result("cached_test", "passed", 0.1, __file__)
        
        # Should not run if nothing changed
        assert manager.should_run_test("cached_test", __file__) == False
        
        # Store a failing result
        manager.store_result("failed_test", "failed", 0.1, __file__)
        
        # Should run if previous result was failure
        assert manager.should_run_test("failed_test", __file__) == True
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        manager = TestCacheManager()
        
        # Store some results
        manager.store_result("test1", "passed", 0.1, __file__)
        manager.store_result("test2", "passed", 0.1, __file__)
        
        assert len(manager.results) >= 2
        
        # Invalidate specific pattern
        manager.invalidate_cache("test1")
        assert "test1" not in manager.results
        
        # Invalidate all
        manager.invalidate_cache()
        assert len(manager.results) == 0
    
    def test_cache_statistics(self):
        """Test cache statistics generation."""
        manager = TestCacheManager()
        
        # Store some results
        manager.store_result("test1", "passed", 0.1, __file__)
        manager.store_result("test2", "failed", 0.2, __file__)
        manager.store_result("test3", "passed", 0.3, __file__)
        
        stats = manager.get_cache_stats()
        
        assert stats["total_tests"] >= 3
        assert stats["passed_tests"] >= 2
        assert stats["failed_tests"] >= 1
        assert stats["avg_duration"] > 0
        assert stats["total_duration"] > 0

class TestIncrementalRunner:
    """Tests for the incremental test runner."""
    
    def test_dependency_analysis(self):
        """Test dependency analysis."""
        runner = IncrementalTestRunner(cache_manager)
        
        # Create temporary test file with imports
        test_file = Path("test_temp.py")
        test_file.write_text("""
import pytest
from src.core.config import Config
from src.risk.core.var_calculator import VarCalculator
""")
        
        dependencies = runner.get_test_dependencies(str(test_file))
        
        # Should find dependencies
        assert len(dependencies) > 0
        
        # Cleanup
        test_file.unlink()
    
    def test_changed_files_detection(self):
        """Test detection of changed files."""
        runner = IncrementalTestRunner(cache_manager)
        
        # This will return files that have changed
        changed_files = runner.get_changed_files()
        
        # Should return a list
        assert isinstance(changed_files, list)
    
    def test_tests_to_run_selection(self):
        """Test selection of tests to run."""
        runner = IncrementalTestRunner(cache_manager)
        
        tests_to_run = runner.get_tests_to_run()
        
        # Should return a list
        assert isinstance(tests_to_run, list)