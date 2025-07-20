#!/usr/bin/env python3
"""
Terminal 1 Status Update Script
Updates Terminal 1 progress and synchronizes with coordination system
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

class Terminal1StatusUpdater:
    def __init__(self):
        self.coordination_dir = Path("/home/QuantNova/GrandModel/coordination")
        self.status_file = self.coordination_dir / "terminal_progress" / "terminal1_status.json"
        self.milestones_file = self.coordination_dir / "terminal_progress" / "shared_milestones.json"
        
    def load_current_status(self):
        """Load current Terminal 1 status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return self._get_default_status()
    
    def _get_default_status(self):
        """Get default status structure"""
        return {
            "terminal_id": "terminal_1",
            "primary_responsibilities": [
                "Risk Management MAPPO Training",
                "Execution Engine MAPPO Training", 
                "XAI Explanations MAPPO Training"
            ],
            "current_status": {
                "risk_management": {
                    "progress": 7.1,
                    "status": "pending",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "completion_target": 100.0,
                    "dependencies": ["strategic_models", "tactical_signals"],
                    "issues": []
                },
                "execution_engine": {
                    "progress": 9.1,
                    "status": "pending",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "completion_target": 100.0,
                    "dependencies": ["risk_constraints", "tactical_signals"],
                    "issues": []
                },
                "xai_explanations": {
                    "progress": 16.7,
                    "status": "pending",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "completion_target": 100.0,
                    "dependencies": ["strategic_models", "execution_signals"],
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
        
        self._save_status(current_status)
        print(f"Updated {component}: progress={component_status['progress']}%, status={component_status['status']}")
        return True
    
    def _save_status(self, status):
        """Save status to file"""
        os.makedirs(self.status_file.parent, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def check_dependencies(self):
        """Check if dependencies are satisfied"""
        dependencies_status = {}
        
        # Check for strategic models
        strategic_models_path = self.coordination_dir / "shared_checkpoints" / "strategic_models"
        dependencies_status["strategic_models"] = strategic_models_path.exists() and any(strategic_models_path.iterdir())
        
        # Check for tactical models
        tactical_models_path = self.coordination_dir / "shared_checkpoints" / "tactical_models"
        dependencies_status["tactical_models"] = tactical_models_path.exists() and any(tactical_models_path.iterdir())
        
        # Check Terminal 2 status
        terminal2_status_file = self.coordination_dir / "terminal_progress" / "terminal2_status.json"
        if terminal2_status_file.exists():
            with open(terminal2_status_file, 'r') as f:
                terminal2_status = json.load(f)
            dependencies_status["strategic_progress"] = terminal2_status["current_status"]["strategic_training"]["progress"]
            dependencies_status["tactical_progress"] = terminal2_status["current_status"]["tactical_training"]["progress"]
        
        return dependencies_status
    
    def update_milestone_contribution(self, milestone, contribution_status):
        """Update Terminal 1's contribution to a shared milestone"""
        if self.milestones_file.exists():
            with open(self.milestones_file, 'r') as f:
                milestones = json.load(f)
        else:
            return False
            
        if milestone in milestones["project_milestones"]:
            milestone_data = milestones["project_milestones"][milestone]
            milestone_data[f"terminal1_contribution_status"] = contribution_status
            milestone_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            with open(self.milestones_file, 'w') as f:
                json.dump(milestones, f, indent=2)
            return True
        return False
    
    def generate_status_report(self):
        """Generate a comprehensive status report"""
        status = self.load_current_status()
        dependencies = self.check_dependencies()
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "terminal": "terminal_1",
            "overall_progress": self._calculate_overall_progress(status),
            "component_status": status["current_status"],
            "dependencies_status": dependencies,
            "blocking_issues": self._get_blocking_issues(status),
            "next_actions": self._get_next_actions(status, dependencies)
        }
        
        # Save report
        report_path = self.coordination_dir / "documentation" / "progress_logs" / f"terminal1_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    
    def _get_next_actions(self, status, dependencies):
        """Get recommended next actions"""
        actions = []
        
        for component_name, component in status["current_status"].items():
            if component["status"] == "pending":
                # Check if dependencies are satisfied
                deps_satisfied = True
                for dep in component.get("dependencies", []):
                    if dep in dependencies and not dependencies[dep]:
                        deps_satisfied = False
                        break
                        
                if deps_satisfied:
                    actions.append(f"Start {component_name} training")
                else:
                    actions.append(f"Wait for dependencies for {component_name}: {component['dependencies']}")
            elif component["status"] == "in_progress" and component["progress"] < 100:
                actions.append(f"Continue {component_name} training (progress: {component['progress']}%)")
                
        return actions

def main():
    parser = argparse.ArgumentParser(description="Update Terminal 1 Status")
    parser.add_argument("--component", help="Component to update (risk_management, execution_engine, xai_explanations)")
    parser.add_argument("--progress", type=float, help="Progress percentage (0-100)")
    parser.add_argument("--status", help="Status (pending, in_progress, completed)")
    parser.add_argument("--issue", help="Issue to report")
    parser.add_argument("--milestone", help="Milestone to update")
    parser.add_argument("--milestone-status", help="Milestone contribution status")
    parser.add_argument("--report", action="store_true", help="Generate status report")
    
    args = parser.parse_args()
    
    updater = Terminal1StatusUpdater()
    
    if args.component:
        updater.update_component_status(
            component=args.component,
            progress=args.progress,
            status=args.status,
            issues=[args.issue] if args.issue else None
        )
    
    if args.milestone and args.milestone_status:
        updater.update_milestone_contribution(args.milestone, args.milestone_status)
    
    if args.report:
        report = updater.generate_status_report()
        print(json.dumps(report, indent=2))
    
    # Always check dependencies
    dependencies = updater.check_dependencies()
    print(f"Dependencies Status: {dependencies}")

if __name__ == "__main__":
    main()