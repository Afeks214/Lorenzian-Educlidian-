#!/usr/bin/env python3
"""
Integration Testing Coordination Script
Runs cross-terminal integration tests and validation
"""

import json
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import argparse
import time

class IntegrationTestCoordinator:
    def __init__(self):
        self.coordination_dir = Path("/home/QuantNova/GrandModel/coordination")
        self.project_root = Path("/home/QuantNova/GrandModel")
        self.test_results_dir = self.coordination_dir / "test_data" / "integration_tests"
        
    def run_notebook_execution_tests(self):
        """Test that all notebooks can execute without errors"""
        notebooks = {
            "terminal_1": [
                "colab/notebooks/risk_management_mappo_training.ipynb",
                "colab/notebooks/execution_engine_mappo_training.ipynb",
                "colab/notebooks/xai_trading_explanations_training.ipynb"
            ],
            "terminal_2": [
                "colab/notebooks/strategic_mappo_training.ipynb",
                "colab/notebooks/tactical_mappo_training.ipynb"
            ]
        }
        
        results = {
            "test_type": "notebook_execution",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "terminal_1_results": {},
            "terminal_2_results": {},
            "overall_status": "pending"
        }
        
        # Test Terminal 1 notebooks
        for notebook in notebooks["terminal_1"]:
            notebook_path = self.project_root / notebook
            if notebook_path.exists():
                result = self._test_notebook_syntax(notebook_path)
                results["terminal_1_results"][notebook] = result
            else:
                results["terminal_1_results"][notebook] = {
                    "status": "not_found",
                    "error": f"Notebook not found: {notebook_path}"
                }
        
        # Test Terminal 2 notebooks
        for notebook in notebooks["terminal_2"]:
            notebook_path = self.project_root / notebook
            if notebook_path.exists():
                result = self._test_notebook_syntax(notebook_path)
                results["terminal_2_results"][notebook] = result
            else:
                results["terminal_2_results"][notebook] = {
                    "status": "not_found",
                    "error": f"Notebook not found: {notebook_path}"
                }
        
        # Determine overall status
        all_results = list(results["terminal_1_results"].values()) + list(results["terminal_2_results"].values())
        if all(r["status"] == "success" for r in all_results):
            results["overall_status"] = "success"
        elif any(r["status"] == "error" for r in all_results):
            results["overall_status"] = "error"
        else:
            results["overall_status"] = "partial"
        
        return results
    
    def _test_notebook_syntax(self, notebook_path):
        """Test notebook syntax and basic structure"""
        try:
            with open(notebook_path, 'r') as f:
                notebook_content = json.load(f)
            
            # Basic validation
            if "cells" not in notebook_content:
                return {"status": "error", "error": "No cells found in notebook"}
            
            if len(notebook_content["cells"]) == 0:
                return {"status": "error", "error": "Notebook is empty"}
            
            # Check for code cells
            code_cells = [cell for cell in notebook_content["cells"] if cell.get("cell_type") == "code"]
            if len(code_cells) == 0:
                return {"status": "warning", "warning": "No code cells found"}
            
            # Check for MAPPO training patterns
            has_training_code = False
            for cell in code_cells:
                source = "".join(cell.get("source", []))
                if "MAPPO" in source or "train" in source.lower() or "agent" in source.lower():
                    has_training_code = True
                    break
            
            if not has_training_code:
                return {"status": "warning", "warning": "No training code patterns detected"}
            
            return {
                "status": "success",
                "cells_count": len(notebook_content["cells"]),
                "code_cells_count": len(code_cells)
            }
            
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"JSON decode error: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error": f"Unexpected error: {str(e)}"}
    
    def run_checkpoint_sharing_tests(self):
        """Test checkpoint sharing between terminals"""
        results = {
            "test_type": "checkpoint_sharing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tests": {}
        }
        
        # Test checkpoint directories exist
        checkpoint_dirs = ["strategic_models", "tactical_models", "risk_models", "execution_models"]
        
        for dir_name in checkpoint_dirs:
            checkpoint_dir = self.coordination_dir / "shared_checkpoints" / dir_name
            
            test_result = {
                "directory_exists": checkpoint_dir.exists(),
                "is_writable": os.access(checkpoint_dir, os.W_OK) if checkpoint_dir.exists() else False,
                "file_count": len(list(checkpoint_dir.iterdir())) if checkpoint_dir.exists() else 0
            }
            
            # Test write/read operations
            if test_result["is_writable"]:
                test_file = checkpoint_dir / "test_checkpoint.txt"
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    with open(test_file, 'r') as f:
                        content = f.read()
                    test_file.unlink()  # Remove test file
                    test_result["read_write_test"] = "success"
                except Exception as e:
                    test_result["read_write_test"] = f"failed: {str(e)}"
            else:
                test_result["read_write_test"] = "skipped: not writable"
            
            results["tests"][dir_name] = test_result
        
        # Overall status
        all_tests_passed = all(
            test["directory_exists"] and test["is_writable"] and test["read_write_test"] == "success"
            for test in results["tests"].values()
        )
        
        results["overall_status"] = "success" if all_tests_passed else "failed"
        
        return results
    
    def run_configuration_consistency_tests(self):
        """Test configuration consistency across terminals"""
        results = {
            "test_type": "configuration_consistency",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tests": {}
        }
        
        config_files = [
            "shared_configs/marl_config.yaml",
            "shared_configs/colab_config.yaml",
            "shared_configs/training_params.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.coordination_dir / config_file
            
            test_result = {
                "file_exists": config_path.exists(),
                "readable": False,
                "valid_yaml": False,
                "has_required_sections": False
            }
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                    test_result["readable"] = True
                    
                    # Test YAML parsing
                    import yaml
                    config_data = yaml.safe_load(content)
                    test_result["valid_yaml"] = True
                    
                    # Check for required sections based on file type
                    if "marl_config" in config_file:
                        required_sections = ["agents", "centralized_critic", "training"]
                    elif "colab_config" in config_file:
                        required_sections = ["colab_environment", "gpu_optimization", "checkpoint_management"]
                    elif "training_params" in config_file:
                        required_sections = ["mappo_config", "agent_training_params", "centralized_critic"]
                    else:
                        required_sections = []
                    
                    has_all_sections = all(section in config_data for section in required_sections)
                    test_result["has_required_sections"] = has_all_sections
                    test_result["missing_sections"] = [s for s in required_sections if s not in config_data]
                    
                except yaml.YAMLError as e:
                    test_result["yaml_error"] = str(e)
                except Exception as e:
                    test_result["error"] = str(e)
            
            results["tests"][config_file] = test_result
        
        # Overall status
        all_tests_passed = all(
            test["file_exists"] and test["readable"] and test["valid_yaml"] and test["has_required_sections"]
            for test in results["tests"].values()
        )
        
        results["overall_status"] = "success" if all_tests_passed else "failed"
        
        return results
    
    def run_communication_tests(self):
        """Test terminal communication protocols"""
        results = {
            "test_type": "terminal_communication",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tests": {}
        }
        
        # Test status file updates
        status_files = [
            "terminal_progress/terminal1_status.json",
            "terminal_progress/terminal2_status.json",
            "terminal_progress/shared_milestones.json"
        ]
        
        for status_file in status_files:
            status_path = self.coordination_dir / status_file
            
            test_result = {
                "file_exists": status_path.exists(),
                "readable": False,
                "valid_json": False,
                "has_required_fields": False
            }
            
            if status_path.exists():
                try:
                    with open(status_path, 'r') as f:
                        content = f.read()
                    test_result["readable"] = True
                    
                    # Test JSON parsing
                    status_data = json.loads(content)
                    test_result["valid_json"] = True
                    
                    # Check for required fields based on file type
                    if "terminal1_status" in status_file:
                        required_fields = ["terminal_id", "current_status", "last_coordination_sync"]
                    elif "terminal2_status" in status_file:
                        required_fields = ["terminal_id", "current_status", "last_coordination_sync"]
                    elif "shared_milestones" in status_file:
                        required_fields = ["project_milestones"]
                    else:
                        required_fields = []
                    
                    has_all_fields = all(field in status_data for field in required_fields)
                    test_result["has_required_fields"] = has_all_fields
                    test_result["missing_fields"] = [f for f in required_fields if f not in status_data]
                    
                except json.JSONDecodeError as e:
                    test_result["json_error"] = str(e)
                except Exception as e:
                    test_result["error"] = str(e)
            
            results["tests"][status_file] = test_result
        
        # Test script executability
        script_files = [
            "scripts/update_terminal1_status.py",
            "scripts/update_terminal2_status.py",
            "scripts/sync_milestones.py",
            "scripts/check_dependencies.py"
        ]
        
        for script_file in script_files:
            script_path = self.coordination_dir / script_file
            
            test_result = {
                "file_exists": script_path.exists(),
                "executable": os.access(script_path, os.X_OK) if script_path.exists() else False,
                "syntax_valid": False
            }
            
            if script_path.exists():
                try:
                    # Test Python syntax
                    with open(script_path, 'r') as f:
                        content = f.read()
                    compile(content, script_path, 'exec')
                    test_result["syntax_valid"] = True
                except SyntaxError as e:
                    test_result["syntax_error"] = str(e)
                except Exception as e:
                    test_result["error"] = str(e)
            
            results["tests"][f"script_{script_file}"] = test_result
        
        # Overall status
        all_tests_passed = all(
            test.get("file_exists", False) and 
            (test.get("valid_json", False) or test.get("syntax_valid", False)) and
            test.get("has_required_fields", True)
            for test in results["tests"].values()
        )
        
        results["overall_status"] = "success" if all_tests_passed else "failed"
        
        return results
    
    def run_all_integration_tests(self):
        """Run all integration tests"""
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_suite": "integration_tests",
            "tests": {}
        }
        
        # Run individual test suites
        test_suites = [
            ("notebook_execution", self.run_notebook_execution_tests),
            ("checkpoint_sharing", self.run_checkpoint_sharing_tests),
            ("configuration_consistency", self.run_configuration_consistency_tests),
            ("terminal_communication", self.run_communication_tests)
        ]
        
        overall_success = True
        
        for suite_name, test_function in test_suites:
            print(f"Running {suite_name} tests...")
            try:
                suite_results = test_function()
                all_results["tests"][suite_name] = suite_results
                
                if suite_results.get("overall_status") != "success":
                    overall_success = False
                    
                print(f"  {suite_name}: {suite_results.get('overall_status', 'unknown')}")
                
            except Exception as e:
                all_results["tests"][suite_name] = {
                    "test_type": suite_name,
                    "overall_status": "error",
                    "error": str(e)
                }
                overall_success = False
                print(f"  {suite_name}: error - {str(e)}")
        
        all_results["overall_status"] = "success" if overall_success else "failed"
        all_results["success_rate"] = sum(
            1 for test in all_results["tests"].values() 
            if test.get("overall_status") == "success"
        ) / len(all_results["tests"]) if all_results["tests"] else 0
        
        # Save results
        os.makedirs(self.test_results_dir, exist_ok=True)
        results_file = self.test_results_dir / f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def validate_system_readiness(self):
        """Validate overall system readiness for coordination"""
        integration_results = self.run_all_integration_tests()
        
        # Check dependencies
        dependency_checker_script = self.coordination_dir / "scripts" / "check_dependencies.py"
        dependency_results = {}
        
        if dependency_checker_script.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(dependency_checker_script), "--all"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    dependency_results = json.loads(result.stdout)
                else:
                    dependency_results = {"error": result.stderr}
            except Exception as e:
                dependency_results = {"error": str(e)}
        
        # Check milestones
        milestone_sync_script = self.coordination_dir / "scripts" / "sync_milestones.py"
        milestone_results = {}
        
        if milestone_sync_script.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(milestone_sync_script), "--sync", "--report"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Parse the output for milestone information
                    milestone_results = {"status": "synced", "output": result.stdout}
                else:
                    milestone_results = {"error": result.stderr}
            except Exception as e:
                milestone_results = {"error": str(e)}
        
        readiness_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "integration_tests": integration_results,
            "dependency_status": dependency_results,
            "milestone_status": milestone_results,
            "overall_readiness": self._calculate_overall_readiness(
                integration_results, dependency_results, milestone_results
            )
        }
        
        # Save readiness report
        readiness_file = self.coordination_dir / "documentation" / "shared_findings" / f"system_readiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(readiness_file.parent, exist_ok=True)
        with open(readiness_file, 'w') as f:
            json.dump(readiness_report, f, indent=2)
        
        return readiness_report
    
    def _calculate_overall_readiness(self, integration_results, dependency_results, milestone_results):
        """Calculate overall system readiness score"""
        readiness_score = 0
        max_score = 100
        
        # Integration tests (40% of score)
        if integration_results.get("overall_status") == "success":
            readiness_score += 40
        elif integration_results.get("success_rate", 0) > 0:
            readiness_score += 40 * integration_results["success_rate"]
        
        # Dependencies (30% of score)
        if not dependency_results.get("error"):
            # Check if terminals can communicate
            comm_status = dependency_results.get("terminal_communication", {})
            if comm_status.get("terminal1_reachable") and comm_status.get("terminal2_reachable"):
                readiness_score += 30
            else:
                readiness_score += 15  # Partial credit
        
        # Milestones (30% of score)
        if not milestone_results.get("error"):
            readiness_score += 30
        
        return {
            "score": readiness_score,
            "max_score": max_score,
            "percentage": (readiness_score / max_score) * 100,
            "status": "ready" if readiness_score >= 80 else "partial" if readiness_score >= 50 else "not_ready"
        }

def main():
    parser = argparse.ArgumentParser(description="Run Integration Tests")
    parser.add_argument("--notebooks", action="store_true", help="Test notebook execution")
    parser.add_argument("--checkpoints", action="store_true", help="Test checkpoint sharing")
    parser.add_argument("--config", action="store_true", help="Test configuration consistency")
    parser.add_argument("--communication", action="store_true", help="Test terminal communication")
    parser.add_argument("--all", action="store_true", help="Run all integration tests")
    parser.add_argument("--readiness", action="store_true", help="Validate system readiness")
    
    args = parser.parse_args()
    
    coordinator = IntegrationTestCoordinator()
    
    if args.notebooks:
        results = coordinator.run_notebook_execution_tests()
        print(json.dumps(results, indent=2))
    
    if args.checkpoints:
        results = coordinator.run_checkpoint_sharing_tests()
        print(json.dumps(results, indent=2))
    
    if args.config:
        results = coordinator.run_configuration_consistency_tests()
        print(json.dumps(results, indent=2))
    
    if args.communication:
        results = coordinator.run_communication_tests()
        print(json.dumps(results, indent=2))
    
    if args.all:
        results = coordinator.run_all_integration_tests()
        print(f"Integration Tests Complete")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
    
    if args.readiness:
        readiness = coordinator.validate_system_readiness()
        print(f"System Readiness: {readiness['overall_readiness']['status']}")
        print(f"Readiness Score: {readiness['overall_readiness']['score']}/{readiness['overall_readiness']['max_score']} ({readiness['overall_readiness']['percentage']:.1f}%)")

if __name__ == "__main__":
    main()