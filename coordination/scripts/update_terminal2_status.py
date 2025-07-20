#!/usr/bin/env python3
"""
Terminal 2 Status Update Script
Updates Terminal 2 progress and synchronizes with coordination system
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class Terminal2StatusUpdater:
    def __init__(self):
        self.coordination_dir = Path("/home/QuantNova/GrandModel/coordination")
        self.status_file = self.coordination_dir / "terminal_progress" / "terminal2_status.json"
        self.milestones_file = self.coordination_dir / "terminal_progress" / "shared_milestones.json"
        
    def load_current_status(self):
        """Load current Terminal 2 status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return self._get_default_status()
    
    def _get_default_status(self):
        """Get default status structure"""
        return {
            "terminal_id": "terminal_2",
            "primary_responsibilities": [
                "Strategic MAPPO Training",
                "Tactical MAPPO Training"
            ],
            "current_status": {
                "strategic_training": {
                    "progress": 83.3,
                    "status": "in_progress",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "completion_target": 100.0,
                    "dependencies": [],
                    "issues": []
                },
                "tactical_training": {
                    "progress": 36.8,
                    "status": "pending",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "completion_target": 100.0,
                    "dependencies": ["strategic_foundation"],
                    "issues": []
                }
            },
            "last_coordination_sync": datetime.now(timezone.utc).isoformat()
        }
    
    def update_component_status(self, component, progress=None, status=None, issues=None):
        """Update status of a specific component"""
        current_status = self.load_current_status()
        
        if component not in current_status["current_status"]:
            print(f"Warning: Component {component} not found in status")
            return False
            
        component_status = current_status["current_status"][component]
        
        if progress is not None:
            component_status["progress"] = float(progress)
            
        if status is not None:
            component_status["status"] = status
            
        if issues is not None:
            if isinstance(issues, str):
                issues = [issues]
            component_status["issues"] = issues
            
        component_status["last_updated"] = datetime.now(timezone.utc).isoformat()
        current_status["last_coordination_sync"] = datetime.now(timezone.utc).isoformat()
        
        # If strategic training is completed, create checkpoint
        if component == "strategic_training" and progress == 100.0:
            self._create_strategic_checkpoint()
            
        # If tactical training is completed, create checkpoint
        if component == "tactical_training" and progress == 100.0:
            self._create_tactical_checkpoint()
        
        self._save_status(current_status)
        print(f"Updated {component}: progress={component_status['progress']}%, status={component_status['status']}")
        return True
    
    def _save_status(self, status):
        """Save status to file"""
        os.makedirs(self.status_file.parent, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def _create_strategic_checkpoint(self):
        """Create strategic model checkpoint for Terminal 1"""
        checkpoint_dir = self.coordination_dir / "shared_checkpoints" / "strategic_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "terminal": "terminal_2",
            "model_type": "strategic_mappo",
            "progress": 100.0,
            "status": "completed",
            "checkpoint_path": str(checkpoint_dir / "strategic_model_latest.pth"),
            "metadata": {
                "architecture": "MAPPO",
                "observation_space": 512,
                "action_space": 64,
                "training_episodes": 10000,
                "final_reward": "TBD"
            }
        }
        
        with open(checkpoint_dir / "strategic_checkpoint_info.json", 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print("Strategic model checkpoint created for Terminal 1")
    
    def _create_tactical_checkpoint(self):
        """Create tactical model checkpoint for Terminal 1"""
        checkpoint_dir = self.coordination_dir / "shared_checkpoints" / "tactical_models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "terminal": "terminal_2",
            "model_type": "tactical_mappo",
            "progress": 100.0,
            "status": "completed",
            "checkpoint_path": str(checkpoint_dir / "tactical_model_latest.pth"),
            "metadata": {
                "architecture": "MAPPO",
                "observation_space": 256,
                "action_space": 32,
                "training_episodes": 5000,
                "final_reward": "TBD"
            }
        }
        
        with open(checkpoint_dir / "tactical_checkpoint_info.json", 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print("Tactical model checkpoint created for Terminal 1")
    
    def check_terminal1_status(self):
        """Check Terminal 1 status and dependencies"""
        terminal1_status_file = self.coordination_dir / "terminal_progress" / "terminal1_status.json"
        if terminal1_status_file.exists():
            with open(terminal1_status_file, 'r') as f:
                return json.load(f)
        return None
    
    def update_milestone_contribution(self, milestone, contribution_status):
        """Update Terminal 2's contribution to a shared milestone"""
        if self.milestones_file.exists():
            with open(self.milestones_file, 'r') as f:
                milestones = json.load(f)
        else:
            return False
            
        if milestone in milestones["project_milestones"]:
            milestone_data = milestones["project_milestones"][milestone]
            milestone_data[f"terminal2_contribution_status"] = contribution_status
            milestone_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            with open(self.milestones_file, 'w') as f:
                json.dump(milestones, f, indent=2)
            return True
        return False
    
    def generate_status_report(self):
        """Generate a comprehensive status report"""
        status = self.load_current_status()
        terminal1_status = self.check_terminal1_status()
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "terminal": "terminal_2",
            "overall_progress": self._calculate_overall_progress(status),
            "component_status": status["current_status"],
            "terminal1_dependencies": self._check_terminal1_dependencies(terminal1_status),
            "blocking_issues": self._get_blocking_issues(status),
            "next_actions": self._get_next_actions(status),
            "coordination_status": self._get_coordination_status(terminal1_status)
        }
        
        # Save report
        report_path = self.coordination_dir / "documentation" / "progress_logs" / f"terminal2_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
    
    def _calculate_overall_progress(self, status):
        """Calculate overall progress across all components"""
        total_progress = 0
        components = status["current_status"]
        for component in components.values():
            total_progress += component["progress"]
        return total_progress / len(components) if components else 0
    
    def _check_terminal1_dependencies(self, terminal1_status):
        """Check what Terminal 1 is waiting for from Terminal 2"""
        if not terminal1_status:
            return {"status": "unknown", "dependencies": []}
            
        dependencies = []
        for component_name, component in terminal1_status["current_status"].items():
            for dep in component.get("dependencies", []):
                if "strategic" in dep or "tactical" in dep:
                    dependencies.append({
                        "component": component_name,
                        "dependency": dep,
                        "status": component["status"]
                    })
        
        return {"status": "available", "dependencies": dependencies}
    
    def _get_blocking_issues(self, status):
        """Get all blocking issues"""
        blocking_issues = []
        for component_name, component in status["current_status"].items():
            for issue in component.get("issues", []):
                if "blocking" in issue.lower() or "critical" in issue.lower():
                    blocking_issues.append({
                        "component": component_name,
                        "issue": issue,
                        "status": component["status"]
                    })
        return blocking_issues
    
    def _get_next_actions(self, status):
        """Get recommended next actions"""
        actions = []
        
        for component_name, component in status["current_status"].items():
            if component["status"] == "pending":
                actions.append(f"Start {component_name}")
            elif component["status"] == "in_progress" and component["progress"] < 100:
                actions.append(f"Continue {component_name} (progress: {component['progress']}%)")
            elif component["progress"] == 100.0:
                actions.append(f"Validate and checkpoint {component_name}")
                
        return actions
    
    def _get_coordination_status(self, terminal1_status):
        """Get coordination status with Terminal 1"""
        if not terminal1_status:
            return {"status": "no_communication", "message": "Cannot reach Terminal 1"}
            
        # Check if Terminal 1 is waiting for our models
        waiting_for_us = []
        for component_name, component in terminal1_status["current_status"].items():
            for dep in component.get("dependencies", []):
                if "strategic" in dep or "tactical" in dep:
                    waiting_for_us.append(component_name)
        
        return {
            "status": "active" if waiting_for_us else "independent",
            "terminal1_waiting_for": waiting_for_us,
            "last_sync": terminal1_status.get("last_coordination_sync", "unknown")
        }

def main():
    parser = argparse.ArgumentParser(description="Update Terminal 2 Status")
    parser.add_argument("--component", help="Component to update (strategic_training, tactical_training)")
    parser.add_argument("--progress", type=float, help="Progress percentage (0-100)")
    parser.add_argument("--status", help="Status (pending, in_progress, completed)")
    parser.add_argument("--issue", help="Issue to report")
    parser.add_argument("--milestone", help="Milestone to update")
    parser.add_argument("--milestone-status", help="Milestone contribution status")
    parser.add_argument("--report", action="store_true", help="Generate status report")
    parser.add_argument("--check-terminal1", action="store_true", help="Check Terminal 1 status")
    
    args = parser.parse_args()
    
    updater = Terminal2StatusUpdater()
    
    if args.component:
        updater.update_component_status(
            component=args.component,
            progress=args.progress,
            status=args.status,
            issues=[args.issue] if args.issue else None
        )
    
    if args.milestone and args.milestone_status:
        updater.update_milestone_contribution(args.milestone, args.milestone_status)
    
    if args.check_terminal1:
        terminal1_status = updater.check_terminal1_status()
        if terminal1_status:
            print("Terminal 1 Status:")
            print(json.dumps(terminal1_status, indent=2))
        else:
            print("Cannot reach Terminal 1")
    
    if args.report:
        report = updater.generate_status_report()
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()