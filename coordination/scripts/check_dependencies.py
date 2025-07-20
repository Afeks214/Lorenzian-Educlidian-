#!/usr/bin/env python3
"""
Dependency Checker Script
Checks and manages dependencies between terminals
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
import time

class DependencyChecker:
    def __init__(self):
        self.coordination_dir = Path("/home/QuantNova/GrandModel/coordination")
        self.terminal1_status_file = self.coordination_dir / "terminal_progress" / "terminal1_status.json"
        self.terminal2_status_file = self.coordination_dir / "terminal_progress" / "terminal2_status.json"
        self.checkpoints_dir = self.coordination_dir / "shared_checkpoints"
        
    def load_terminal_status(self, terminal):
        """Load terminal status"""
        if terminal == "terminal_1":
            status_file = self.terminal1_status_file
        elif terminal == "terminal_2":
            status_file = self.terminal2_status_file
        else:
            return None
            
        if status_file.exists():
            with open(status_file, 'r') as f:
                return json.load(f)
        return None
    
    def check_strategic_models_available(self):
        """Check if strategic models are available for Terminal 1"""
        strategic_dir = self.checkpoints_dir / "strategic_models"
        
        if not strategic_dir.exists():
            return {"available": False, "reason": "Strategic models directory does not exist"}
        
        checkpoint_info_file = strategic_dir / "strategic_checkpoint_info.json"
        if not checkpoint_info_file.exists():
            return {"available": False, "reason": "No strategic checkpoint info found"}
        
        with open(checkpoint_info_file, 'r') as f:
            checkpoint_info = json.load(f)
        
        # Check if checkpoint is recent and complete
        if checkpoint_info.get("status") == "completed":
            return {
                "available": True,
                "checkpoint_info": checkpoint_info,
                "timestamp": checkpoint_info.get("timestamp"),
                "progress": checkpoint_info.get("progress", 0)
            }
        
        return {"available": False, "reason": "Strategic models not completed"}
    
    def check_tactical_models_available(self):
        """Check if tactical models are available for Terminal 1"""
        tactical_dir = self.checkpoints_dir / "tactical_models"
        
        if not tactical_dir.exists():
            return {"available": False, "reason": "Tactical models directory does not exist"}
        
        checkpoint_info_file = tactical_dir / "tactical_checkpoint_info.json"
        if not checkpoint_info_file.exists():
            return {"available": False, "reason": "No tactical checkpoint info found"}
        
        with open(checkpoint_info_file, 'r') as f:
            checkpoint_info = json.load(f)
        
        # Check if checkpoint is recent and complete
        if checkpoint_info.get("status") == "completed":
            return {
                "available": True,
                "checkpoint_info": checkpoint_info,
                "timestamp": checkpoint_info.get("timestamp"),
                "progress": checkpoint_info.get("progress", 0)
            }
        
        return {"available": False, "reason": "Tactical models not completed"}
    
    def check_risk_models_available(self):
        """Check if risk models are available for Terminal 2"""
        risk_dir = self.checkpoints_dir / "risk_models"
        
        if not risk_dir.exists():
            return {"available": False, "reason": "Risk models directory does not exist"}
        
        checkpoint_info_file = risk_dir / "risk_checkpoint_info.json"
        if not checkpoint_info_file.exists():
            return {"available": False, "reason": "No risk checkpoint info found"}
        
        with open(checkpoint_info_file, 'r') as f:
            checkpoint_info = json.load(f)
        
        if checkpoint_info.get("status") == "completed":
            return {
                "available": True,
                "checkpoint_info": checkpoint_info,
                "timestamp": checkpoint_info.get("timestamp"),
                "progress": checkpoint_info.get("progress", 0)
            }
        
        return {"available": False, "reason": "Risk models not completed"}
    
    def check_execution_models_available(self):
        """Check if execution models are available for Terminal 2"""
        execution_dir = self.checkpoints_dir / "execution_models"
        
        if not execution_dir.exists():
            return {"available": False, "reason": "Execution models directory does not exist"}
        
        checkpoint_info_file = execution_dir / "execution_checkpoint_info.json"
        if not checkpoint_info_file.exists():
            return {"available": False, "reason": "No execution checkpoint info found"}
        
        with open(checkpoint_info_file, 'r') as f:
            checkpoint_info = json.load(f)
        
        if checkpoint_info.get("status") == "completed":
            return {
                "available": True,
                "checkpoint_info": checkpoint_info,
                "timestamp": checkpoint_info.get("timestamp"),
                "progress": checkpoint_info.get("progress", 0)
            }
        
        return {"available": False, "reason": "Execution models not completed"}
    
    def check_all_dependencies(self):
        """Check all cross-terminal dependencies"""
        terminal1_status = self.load_terminal_status("terminal_1")
        terminal2_status = self.load_terminal_status("terminal_2")
        
        dependencies = {
            "terminal1_dependencies": {
                "strategic_models": self.check_strategic_models_available(),
                "tactical_models": self.check_tactical_models_available()
            },
            "terminal2_dependencies": {
                "risk_models": self.check_risk_models_available(),
                "execution_models": self.check_execution_models_available()
            },
            "terminal_communication": {
                "terminal1_reachable": terminal1_status is not None,
                "terminal2_reachable": terminal2_status is not None,
                "last_sync": self._get_last_sync_time(terminal1_status, terminal2_status)
            }
        }
        
        return dependencies
    
    def _get_last_sync_time(self, terminal1_status, terminal2_status):
        """Get last synchronization time between terminals"""
        times = []
        
        if terminal1_status and "last_coordination_sync" in terminal1_status:
            times.append(terminal1_status["last_coordination_sync"])
        
        if terminal2_status and "last_coordination_sync" in terminal2_status:
            times.append(terminal2_status["last_coordination_sync"])
        
        if times:
            return min(times)  # Return earliest sync time
        return None
    
    def wait_for_dependency(self, dependency_type, timeout_hours=2, check_interval_minutes=5):
        """Wait for a specific dependency to become available"""
        timeout_seconds = timeout_hours * 3600
        check_interval_seconds = check_interval_minutes * 60
        start_time = time.time()
        
        print(f"Waiting for dependency: {dependency_type}")
        print(f"Timeout: {timeout_hours} hours, Check interval: {check_interval_minutes} minutes")
        
        while (time.time() - start_time) < timeout_seconds:
            if dependency_type == "strategic_models":
                result = self.check_strategic_models_available()
            elif dependency_type == "tactical_models":
                result = self.check_tactical_models_available()
            elif dependency_type == "risk_models":
                result = self.check_risk_models_available()
            elif dependency_type == "execution_models":
                result = self.check_execution_models_available()
            else:
                print(f"Unknown dependency type: {dependency_type}")
                return False
            
            if result["available"]:
                print(f"Dependency {dependency_type} is now available!")
                return True
            
            elapsed_minutes = (time.time() - start_time) / 60
            remaining_minutes = (timeout_seconds - (time.time() - start_time)) / 60
            
            print(f"Dependency {dependency_type} not ready. Elapsed: {elapsed_minutes:.1f}min, Remaining: {remaining_minutes:.1f}min")
            print(f"Reason: {result.get('reason', 'Unknown')}")
            
            time.sleep(check_interval_seconds)
        
        print(f"Timeout reached waiting for dependency: {dependency_type}")
        return False
    
    def check_terminal1_ready_for_work(self):
        """Check if Terminal 1 can start its work"""
        dependencies = self.check_all_dependencies()
        terminal1_deps = dependencies["terminal1_dependencies"]
        
        # Terminal 1 needs strategic models to start risk management
        # Terminal 1 can start execution engine after risk management
        # Terminal 1 can start XAI after strategic models
        
        readiness = {
            "risk_management": {
                "ready": terminal1_deps["strategic_models"]["available"],
                "dependencies": ["strategic_models"],
                "reason": terminal1_deps["strategic_models"].get("reason", "")
            },
            "execution_engine": {
                "ready": terminal1_deps["tactical_models"]["available"],
                "dependencies": ["tactical_models"],
                "reason": terminal1_deps["tactical_models"].get("reason", "")
            },
            "xai_explanations": {
                "ready": terminal1_deps["strategic_models"]["available"],
                "dependencies": ["strategic_models"],
                "reason": terminal1_deps["strategic_models"].get("reason", "")
            }
        }
        
        return readiness
    
    def check_terminal2_ready_for_work(self):
        """Check if Terminal 2 can start its work"""
        # Terminal 2 can start strategic training immediately
        # Terminal 2 can start tactical training after strategic is partially complete
        
        terminal2_status = self.load_terminal_status("terminal_2")
        
        readiness = {
            "strategic_training": {
                "ready": True,
                "dependencies": [],
                "reason": "No dependencies"
            },
            "tactical_training": {
                "ready": True,  # Can start in parallel or after strategic foundation
                "dependencies": [],
                "reason": "Can start independently or after strategic foundation"
            }
        }
        
        # Check if strategic training has enough progress for tactical to start
        if terminal2_status:
            strategic_progress = terminal2_status["current_status"]["strategic_training"]["progress"]
            if strategic_progress >= 50.0:  # Tactical can start when strategic is 50% done
                readiness["tactical_training"]["ready"] = True
                readiness["tactical_training"]["reason"] = f"Strategic training at {strategic_progress}%"
            elif strategic_progress < 20.0:
                readiness["tactical_training"]["ready"] = False
                readiness["tactical_training"]["reason"] = f"Wait for strategic training to reach 20% (currently {strategic_progress}%)"
        
        return readiness
    
    def get_dependency_graph(self):
        """Get the complete dependency graph"""
        return {
            "dependency_flow": {
                "terminal_2_strategic": {
                    "produces": ["strategic_models"],
                    "consumed_by": ["terminal_1_risk_management", "terminal_1_xai_explanations"],
                    "dependencies": []
                },
                "terminal_2_tactical": {
                    "produces": ["tactical_models"],
                    "consumed_by": ["terminal_1_execution_engine"],
                    "dependencies": ["strategic_foundation"]
                },
                "terminal_1_risk_management": {
                    "produces": ["risk_models", "risk_constraints"],
                    "consumed_by": ["terminal_2_strategic_planning"],
                    "dependencies": ["strategic_models"]
                },
                "terminal_1_execution_engine": {
                    "produces": ["execution_models", "execution_feedback"],
                    "consumed_by": ["terminal_2_tactical_refinement"],
                    "dependencies": ["tactical_models", "risk_constraints"]
                },
                "terminal_1_xai_explanations": {
                    "produces": ["explanation_models"],
                    "consumed_by": ["system_integration"],
                    "dependencies": ["strategic_models", "execution_signals"]
                }
            },
            "critical_path": [
                "terminal_2_strategic",
                "terminal_1_risk_management",
                "terminal_1_execution_engine",
                "system_integration"
            ],
            "parallel_paths": [
                ["terminal_2_strategic", "terminal_1_xai_explanations"],
                ["terminal_2_tactical", "terminal_1_execution_engine"]
            ]
        }
    
    def generate_dependency_report(self):
        """Generate comprehensive dependency report"""
        dependencies = self.check_all_dependencies()
        terminal1_readiness = self.check_terminal1_ready_for_work()
        terminal2_readiness = self.check_terminal2_ready_for_work()
        dependency_graph = self.get_dependency_graph()
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dependencies_status": dependencies,
            "terminal1_readiness": terminal1_readiness,
            "terminal2_readiness": terminal2_readiness,
            "dependency_graph": dependency_graph,
            "blocking_dependencies": self._get_blocking_dependencies(dependencies, terminal1_readiness, terminal2_readiness),
            "recommendations": self._get_dependency_recommendations(terminal1_readiness, terminal2_readiness)
        }
        
        # Save report
        report_path = self.coordination_dir / "documentation" / "progress_logs" / f"dependency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _get_blocking_dependencies(self, dependencies, terminal1_readiness, terminal2_readiness):
        """Get dependencies that are currently blocking progress"""
        blocking = []
        
        # Check Terminal 1 blocking dependencies
        for component, readiness in terminal1_readiness.items():
            if not readiness["ready"]:
                blocking.append({
                    "terminal": "terminal_1",
                    "component": component,
                    "blocked_by": readiness["dependencies"],
                    "reason": readiness["reason"]
                })
        
        # Check Terminal 2 blocking dependencies
        for component, readiness in terminal2_readiness.items():
            if not readiness["ready"]:
                blocking.append({
                    "terminal": "terminal_2",
                    "component": component,
                    "blocked_by": readiness["dependencies"],
                    "reason": readiness["reason"]
                })
        
        return blocking
    
    def _get_dependency_recommendations(self, terminal1_readiness, terminal2_readiness):
        """Get recommendations for resolving dependencies"""
        recommendations = []
        
        # Terminal 2 should start strategic training first
        if terminal2_readiness["strategic_training"]["ready"]:
            recommendations.append("Terminal 2: Start strategic training immediately")
        
        # Terminal 1 can start components that are ready
        for component, readiness in terminal1_readiness.items():
            if readiness["ready"]:
                recommendations.append(f"Terminal 1: {component} is ready to start")
            else:
                recommendations.append(f"Terminal 1: Wait for {readiness['dependencies']} before starting {component}")
        
        # Terminal 2 tactical training recommendations
        if terminal2_readiness["tactical_training"]["ready"]:
            recommendations.append("Terminal 2: Tactical training can start")
        else:
            recommendations.append(f"Terminal 2: {terminal2_readiness['tactical_training']['reason']}")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description="Check Dependencies")
    parser.add_argument("--all", action="store_true", help="Check all dependencies")
    parser.add_argument("--wait", help="Wait for specific dependency (strategic_models, tactical_models, risk_models, execution_models)")
    parser.add_argument("--timeout", type=int, default=2, help="Timeout in hours for waiting")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in minutes")
    parser.add_argument("--terminal1", action="store_true", help="Check Terminal 1 readiness")
    parser.add_argument("--terminal2", action="store_true", help="Check Terminal 2 readiness")
    parser.add_argument("--report", action="store_true", help="Generate dependency report")
    parser.add_argument("--graph", action="store_true", help="Show dependency graph")
    
    args = parser.parse_args()
    
    checker = DependencyChecker()
    
    if args.all:
        dependencies = checker.check_all_dependencies()
        print(json.dumps(dependencies, indent=2))
    
    if args.wait:
        success = checker.wait_for_dependency(args.wait, args.timeout, args.interval)
        if success:
            print(f"Dependency {args.wait} is now available")
            sys.exit(0)
        else:
            print(f"Timeout waiting for dependency {args.wait}")
            sys.exit(1)
    
    if args.terminal1:
        readiness = checker.check_terminal1_ready_for_work()
        print("Terminal 1 Readiness:")
        print(json.dumps(readiness, indent=2))
    
    if args.terminal2:
        readiness = checker.check_terminal2_ready_for_work()
        print("Terminal 2 Readiness:")
        print(json.dumps(readiness, indent=2))
    
    if args.graph:
        graph = checker.get_dependency_graph()
        print("Dependency Graph:")
        print(json.dumps(graph, indent=2))
    
    if args.report:
        report = checker.generate_dependency_report()
        print("Dependency Report Generated:")
        print(f"Blocking Dependencies: {len(report['blocking_dependencies'])}")
        print("Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    main()