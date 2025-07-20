#!/usr/bin/env python3
"""
Milestone Synchronization Script
Synchronizes milestone progress between terminals and updates shared status
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import argparse

class MilestoneSynchronizer:
    def __init__(self):
        self.coordination_dir = Path("/home/QuantNova/GrandModel/coordination")
        self.milestones_file = self.coordination_dir / "terminal_progress" / "shared_milestones.json"
        self.terminal1_status_file = self.coordination_dir / "terminal_progress" / "terminal1_status.json"
        self.terminal2_status_file = self.coordination_dir / "terminal_progress" / "terminal2_status.json"
        
    def load_milestones(self):
        """Load current milestone data"""
        if self.milestones_file.exists():
            with open(self.milestones_file, 'r') as f:
                return json.load(f)
        return self._get_default_milestones()
    
    def _get_default_milestones(self):
        """Get default milestone structure"""
        return {
            "project_milestones": {
                "checkpoint_1": {
                    "name": "All notebooks execute without errors",
                    "description": "All 5 MAPPO training notebooks run successfully without compilation or runtime errors",
                    "status": "pending",
                    "priority": "critical",
                    "target_date": "2025-07-20T12:00:00Z",
                    "terminal1_tasks": [
                        "risk_management_notebook_execution",
                        "execution_engine_notebook_execution", 
                        "xai_explanations_notebook_execution"
                    ],
                    "terminal2_tasks": [
                        "strategic_notebook_execution",
                        "tactical_notebook_execution"
                    ],
                    "success_criteria": [
                        "All notebooks complete without errors",
                        "All model architectures initialize properly",
                        "Training loops execute for at least 10 iterations",
                        "Checkpoint saving/loading functional"
                    ],
                    "terminal1_progress": 0.0,
                    "terminal2_progress": 0.0,
                    "overall_progress": 0.0
                },
                "checkpoint_2": {
                    "name": "MARL integration functional", 
                    "description": "Multi-agent coordination between all 5 trained models working properly",
                    "status": "pending",
                    "priority": "critical",
                    "target_date": "2025-07-20T18:00:00Z",
                    "terminal1_tasks": [
                        "risk_execution_integration_testing",
                        "xai_integration_validation"
                    ],
                    "terminal2_tasks": [
                        "strategic_tactical_coordination_testing",
                        "multi_agent_communication_validation"
                    ],
                    "success_criteria": [
                        "Agents can communicate via shared observation space",
                        "Centralized critic properly aggregates agent observations",
                        "Action coordination protocols functional",
                        "Reward sharing mechanisms operational"
                    ],
                    "terminal1_progress": 0.0,
                    "terminal2_progress": 0.0,
                    "overall_progress": 0.0
                },
                "checkpoint_3": {
                    "name": "Colab Pro optimization complete",
                    "description": "All notebooks optimized for Google Colab Pro environment with GPU acceleration",
                    "status": "pending", 
                    "priority": "high",
                    "target_date": "2025-07-21T00:00:00Z",
                    "terminal1_tasks": [
                        "risk_colab_gpu_optimization",
                        "execution_colab_memory_optimization",
                        "xai_colab_visualization_optimization"
                    ],
                    "terminal2_tasks": [
                        "strategic_colab_batch_optimization",
                        "tactical_colab_real_time_optimization"
                    ],
                    "success_criteria": [
                        "GPU memory usage optimized below 15GB per notebook",
                        "Training time reduced by at least 50%",
                        "Automatic checkpoint management implemented",
                        "Google Drive integration functional"
                    ],
                    "terminal1_progress": 0.0,
                    "terminal2_progress": 0.0,
                    "overall_progress": 0.0
                },
                "checkpoint_4": {
                    "name": "Full system performance validated",
                    "description": "End-to-end system performance meets production requirements",
                    "status": "pending",
                    "priority": "critical",
                    "target_date": "2025-07-21T12:00:00Z",
                    "terminal1_tasks": [
                        "risk_performance_validation",
                        "execution_latency_validation", 
                        "xai_explanation_quality_validation"
                    ],
                    "terminal2_tasks": [
                        "strategic_decision_quality_validation",
                        "tactical_signal_accuracy_validation"
                    ],
                    "success_criteria": [
                        "System latency under 100ms per decision",
                        "Risk management accuracy above 95%",
                        "Strategic decisions show positive alpha",
                        "Tactical signals achieve target Sharpe ratio",
                        "XAI explanations are coherent and actionable"
                    ],
                    "terminal1_progress": 0.0,
                    "terminal2_progress": 0.0,
                    "overall_progress": 0.0
                }
            }
        }
    
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
    
    def calculate_milestone_progress(self, milestone_name):
        """Calculate milestone progress based on terminal status"""
        terminal1_status = self.load_terminal_status("terminal_1")
        terminal2_status = self.load_terminal_status("terminal_2")
        
        if not terminal1_status or not terminal2_status:
            return {"terminal1_progress": 0.0, "terminal2_progress": 0.0, "overall_progress": 0.0}
        
        if milestone_name == "checkpoint_1":
            # All notebooks execute without errors
            t1_progress = self._calculate_notebook_execution_progress(terminal1_status)
            t2_progress = self._calculate_notebook_execution_progress(terminal2_status)
        elif milestone_name == "checkpoint_2":
            # MARL integration functional
            t1_progress = self._calculate_integration_progress(terminal1_status)
            t2_progress = self._calculate_integration_progress(terminal2_status)
        elif milestone_name == "checkpoint_3":
            # Colab Pro optimization complete
            t1_progress = self._calculate_colab_optimization_progress(terminal1_status)
            t2_progress = self._calculate_colab_optimization_progress(terminal2_status)
        elif milestone_name == "checkpoint_4":
            # Full system performance validated
            t1_progress = self._calculate_performance_validation_progress(terminal1_status)
            t2_progress = self._calculate_performance_validation_progress(terminal2_status)
        else:
            t1_progress = 0.0
            t2_progress = 0.0
        
        overall_progress = (t1_progress + t2_progress) / 2.0
        
        return {
            "terminal1_progress": t1_progress,
            "terminal2_progress": t2_progress,
            "overall_progress": overall_progress
        }
    
    def _calculate_notebook_execution_progress(self, terminal_status):
        """Calculate notebook execution progress for checkpoint 1"""
        components = terminal_status["current_status"]
        total_components = len(components)
        completed_components = 0
        
        for component in components.values():
            if component["status"] == "completed" or component["progress"] >= 100.0:
                completed_components += 1
            elif component["status"] == "in_progress":
                completed_components += component["progress"] / 100.0
                
        return (completed_components / total_components) * 100.0 if total_components > 0 else 0.0
    
    def _calculate_integration_progress(self, terminal_status):
        """Calculate integration progress for checkpoint 2"""
        # Integration depends on all models being trained
        notebook_progress = self._calculate_notebook_execution_progress(terminal_status)
        # Integration is 50% complete when notebooks are done, 100% when integration tests pass
        return min(notebook_progress * 0.5, 50.0)
    
    def _calculate_colab_optimization_progress(self, terminal_status):
        """Calculate Colab optimization progress for checkpoint 3"""
        # Assume optimization happens after training
        notebook_progress = self._calculate_notebook_execution_progress(terminal_status)
        return min(notebook_progress * 0.3, 30.0)  # Optimization is 30% of effort after training
    
    def _calculate_performance_validation_progress(self, terminal_status):
        """Calculate performance validation progress for checkpoint 4"""
        # Performance validation is final step
        notebook_progress = self._calculate_notebook_execution_progress(terminal_status)
        return min(notebook_progress * 0.2, 20.0)  # Validation is 20% of effort after training
    
    def sync_all_milestones(self):
        """Sync all milestone progress"""
        milestones = self.load_milestones()
        updated = False
        
        for milestone_name in milestones["project_milestones"].keys():
            progress = self.calculate_milestone_progress(milestone_name)
            milestone = milestones["project_milestones"][milestone_name]
            
            # Update progress
            milestone["terminal1_progress"] = progress["terminal1_progress"]
            milestone["terminal2_progress"] = progress["terminal2_progress"]
            milestone["overall_progress"] = progress["overall_progress"]
            milestone["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Update status based on progress
            if progress["overall_progress"] >= 100.0:
                milestone["status"] = "completed"
            elif progress["overall_progress"] > 0.0:
                milestone["status"] = "in_progress"
            else:
                milestone["status"] = "pending"
                
            updated = True
        
        # Update coordination metrics
        milestones["coordination_metrics"] = self._calculate_coordination_metrics(milestones)
        milestones["coordination_metrics"]["last_sync"] = datetime.now(timezone.utc).isoformat()
        
        if updated:
            self._save_milestones(milestones)
            
        return milestones
    
    def _calculate_coordination_metrics(self, milestones):
        """Calculate overall coordination metrics"""
        project_milestones = milestones["project_milestones"]
        
        total_progress = 0.0
        terminal1_total = 0.0
        terminal2_total = 0.0
        completed_milestones = 0
        
        for milestone in project_milestones.values():
            total_progress += milestone["overall_progress"]
            terminal1_total += milestone["terminal1_progress"]
            terminal2_total += milestone["terminal2_progress"]
            
            if milestone["status"] == "completed":
                completed_milestones += 1
        
        num_milestones = len(project_milestones)
        
        return {
            "overall_progress": total_progress / num_milestones if num_milestones > 0 else 0.0,
            "terminal1_progress": terminal1_total / num_milestones if num_milestones > 0 else 0.0,
            "terminal2_progress": terminal2_total / num_milestones if num_milestones > 0 else 0.0,
            "completed_milestones": completed_milestones,
            "total_milestones": num_milestones,
            "completion_percentage": (completed_milestones / num_milestones) * 100.0 if num_milestones > 0 else 0.0
        }
    
    def _save_milestones(self, milestones):
        """Save milestones to file"""
        os.makedirs(self.milestones_file.parent, exist_ok=True)
        with open(self.milestones_file, 'w') as f:
            json.dump(milestones, f, indent=2)
    
    def get_blocking_milestones(self):
        """Get milestones that are blocking progress"""
        milestones = self.load_milestones()
        blocking = []
        
        for milestone_name, milestone in milestones["project_milestones"].items():
            if milestone["status"] == "pending" and milestone["priority"] == "critical":
                # Check if target date is past
                target_date = datetime.fromisoformat(milestone["target_date"].replace('Z', '+00:00'))
                if datetime.now(timezone.utc) > target_date:
                    blocking.append({
                        "milestone": milestone_name,
                        "name": milestone["name"],
                        "target_date": milestone["target_date"],
                        "progress": milestone["overall_progress"],
                        "days_overdue": (datetime.now(timezone.utc) - target_date).days
                    })
        
        return blocking
    
    def check_milestone_dependencies(self, milestone_name):
        """Check if milestone dependencies are satisfied"""
        milestones = self.load_milestones()
        
        if milestone_name not in milestones["project_milestones"]:
            return {"satisfied": False, "reason": "Milestone not found"}
        
        # Define dependency chain
        dependency_chain = {
            "checkpoint_2": ["checkpoint_1"],
            "checkpoint_3": ["checkpoint_2"],
            "checkpoint_4": ["checkpoint_3"]
        }
        
        if milestone_name not in dependency_chain:
            return {"satisfied": True, "reason": "No dependencies"}
        
        for dependency in dependency_chain[milestone_name]:
            dep_milestone = milestones["project_milestones"][dependency]
            if dep_milestone["status"] != "completed":
                return {
                    "satisfied": False,
                    "reason": f"Dependency {dependency} not completed",
                    "dependency_progress": dep_milestone["overall_progress"]
                }
        
        return {"satisfied": True, "reason": "All dependencies satisfied"}
    
    def generate_milestone_report(self):
        """Generate comprehensive milestone report"""
        milestones = self.sync_all_milestones()
        blocking = self.get_blocking_milestones()
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coordination_metrics": milestones["coordination_metrics"],
            "milestone_status": {},
            "blocking_milestones": blocking,
            "next_actions": [],
            "critical_path": []
        }
        
        # Analyze each milestone
        for milestone_name, milestone in milestones["project_milestones"].items():
            dependencies = self.check_milestone_dependencies(milestone_name)
            
            report["milestone_status"][milestone_name] = {
                "name": milestone["name"],
                "status": milestone["status"],
                "overall_progress": milestone["overall_progress"],
                "terminal1_progress": milestone["terminal1_progress"],
                "terminal2_progress": milestone["terminal2_progress"],
                "target_date": milestone["target_date"],
                "dependencies_satisfied": dependencies["satisfied"],
                "priority": milestone["priority"]
            }
            
            # Determine next actions
            if milestone["status"] == "pending" and dependencies["satisfied"]:
                report["next_actions"].append(f"Start {milestone_name}: {milestone['name']}")
            elif milestone["status"] == "in_progress":
                report["next_actions"].append(f"Continue {milestone_name}: {milestone['overall_progress']:.1f}% complete")
        
        # Determine critical path
        critical_milestones = [m for m in milestones["project_milestones"].keys() 
                             if milestones["project_milestones"][m]["priority"] == "critical"]
        report["critical_path"] = critical_milestones
        
        # Save report
        report_path = self.coordination_dir / "documentation" / "progress_logs" / f"milestone_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Synchronize Milestones")
    parser.add_argument("--sync", action="store_true", help="Sync all milestones")
    parser.add_argument("--report", action="store_true", help="Generate milestone report")
    parser.add_argument("--blocking", action="store_true", help="Show blocking milestones")
    parser.add_argument("--milestone", help="Check specific milestone")
    
    args = parser.parse_args()
    
    synchronizer = MilestoneSynchronizer()
    
    if args.sync:
        milestones = synchronizer.sync_all_milestones()
        print("Milestones synchronized successfully")
        print(f"Overall progress: {milestones['coordination_metrics']['overall_progress']:.1f}%")
    
    if args.blocking:
        blocking = synchronizer.get_blocking_milestones()
        if blocking:
            print("Blocking milestones:")
            for milestone in blocking:
                print(f"  - {milestone['milestone']}: {milestone['days_overdue']} days overdue")
        else:
            print("No blocking milestones")
    
    if args.milestone:
        dependencies = synchronizer.check_milestone_dependencies(args.milestone)
        print(f"Milestone {args.milestone} dependencies: {dependencies}")
    
    if args.report:
        report = synchronizer.generate_milestone_report()
        print("Milestone Report Generated:")
        print(f"Overall Progress: {report['coordination_metrics']['overall_progress']:.1f}%")
        print(f"Completed Milestones: {report['coordination_metrics']['completed_milestones']}/{report['coordination_metrics']['total_milestones']}")

if __name__ == "__main__":
    main()