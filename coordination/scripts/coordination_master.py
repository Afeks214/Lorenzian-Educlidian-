#!/usr/bin/env python3
"""
Coordination Master Script
Main automation script for terminal coordination and system management
"""

import json
import os
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import argparse
import threading
import signal

class CoordinationMaster:
    def __init__(self):
        self.coordination_dir = Path("/home/QuantNova/GrandModel/coordination")
        self.scripts_dir = self.coordination_dir / "scripts"
        self.project_root = Path("/home/QuantNova/GrandModel")
        self.running = True
        self.monitoring_threads = []
        
    def start_coordination_system(self):
        """Start the complete coordination system"""
        print("🚀 Starting Terminal Coordination System")
        print("=" * 50)
        
        # Initialize system
        self._initialize_coordination_system()
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        print("✅ Coordination system started successfully")
        print("\nMonitoring threads active:")
        print("  - Status synchronization (every 30 minutes)")
        print("  - Dependency checking (every 10 minutes)")
        print("  - Milestone tracking (every 15 minutes)")
        print("  - Integration testing (every hour)")
        
        # Keep the main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping coordination system...")
            self.stop_coordination_system()
    
    def _initialize_coordination_system(self):
        """Initialize the coordination system"""
        print("📋 Initializing coordination system...")
        
        # Create necessary directories
        os.makedirs(self.coordination_dir / "terminal_progress", exist_ok=True)
        os.makedirs(self.coordination_dir / "shared_checkpoints" / "strategic_models", exist_ok=True)
        os.makedirs(self.coordination_dir / "shared_checkpoints" / "tactical_models", exist_ok=True)
        os.makedirs(self.coordination_dir / "shared_checkpoints" / "risk_models", exist_ok=True)
        os.makedirs(self.coordination_dir / "shared_checkpoints" / "execution_models", exist_ok=True)
        os.makedirs(self.coordination_dir / "test_data" / "integration_tests", exist_ok=True)
        os.makedirs(self.coordination_dir / "documentation" / "progress_logs", exist_ok=True)
        os.makedirs(self.coordination_dir / "documentation" / "issue_tracking", exist_ok=True)
        os.makedirs(self.coordination_dir / "documentation" / "shared_findings", exist_ok=True)
        
        # Initialize status files if they don't exist
        self._initialize_status_files()
        
        # Run initial system validation
        self._run_initial_validation()
        
        print("✅ System initialization complete")
    
    def _initialize_status_files(self):
        """Initialize status files if they don't exist"""
        # Initialize Terminal 1 status
        terminal1_status_file = self.coordination_dir / "terminal_progress" / "terminal1_status.json"
        if not terminal1_status_file.exists():
            self._run_script("update_terminal1_status.py", ["--report"])
        
        # Initialize Terminal 2 status
        terminal2_status_file = self.coordination_dir / "terminal_progress" / "terminal2_status.json"
        if not terminal2_status_file.exists():
            self._run_script("update_terminal2_status.py", ["--report"])
        
        # Initialize shared milestones
        milestones_file = self.coordination_dir / "terminal_progress" / "shared_milestones.json"
        if not milestones_file.exists():
            self._run_script("sync_milestones.py", ["--sync"])
    
    def _run_initial_validation(self):
        """Run initial system validation"""
        print("🔍 Running initial system validation...")
        
        # Run integration tests
        integration_result = self._run_script("run_integration_tests.py", ["--all"])
        
        # Check dependencies
        dependency_result = self._run_script("check_dependencies.py", ["--all"])
        
        # Sync milestones
        milestone_result = self._run_script("sync_milestones.py", ["--sync", "--report"])
        
        print("✅ Initial validation complete")
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads"""
        # Status synchronization thread
        status_thread = threading.Thread(
            target=self._status_monitor_thread,
            name="StatusMonitor",
            daemon=True
        )
        status_thread.start()
        self.monitoring_threads.append(status_thread)
        
        # Dependency checking thread
        dependency_thread = threading.Thread(
            target=self._dependency_monitor_thread,
            name="DependencyMonitor",
            daemon=True
        )
        dependency_thread.start()
        self.monitoring_threads.append(dependency_thread)
        
        # Milestone tracking thread
        milestone_thread = threading.Thread(
            target=self._milestone_monitor_thread,
            name="MilestoneMonitor",
            daemon=True
        )
        milestone_thread.start()
        self.monitoring_threads.append(milestone_thread)
        
        # Integration testing thread
        integration_thread = threading.Thread(
            target=self._integration_monitor_thread,
            name="IntegrationMonitor",
            daemon=True
        )
        integration_thread.start()
        self.monitoring_threads.append(integration_thread)
    
    def _status_monitor_thread(self):
        """Monitor and synchronize terminal status"""
        while self.running:
            try:
                print("🔄 Syncing terminal status...")
                
                # Check Terminal 1 status
                self._run_script("update_terminal1_status.py", ["--report"])
                
                # Check Terminal 2 status
                self._run_script("update_terminal2_status.py", ["--report", "--check-terminal1"])
                
                # Wait 30 minutes
                time.sleep(30 * 60)
                
            except Exception as e:
                print(f"❌ Error in status monitor: {e}")
                time.sleep(5 * 60)  # Wait 5 minutes before retry
    
    def _dependency_monitor_thread(self):
        """Monitor dependencies between terminals"""
        while self.running:
            try:
                print("🔗 Checking dependencies...")
                
                # Check all dependencies
                self._run_script("check_dependencies.py", ["--report"])
                
                # Wait 10 minutes
                time.sleep(10 * 60)
                
            except Exception as e:
                print(f"❌ Error in dependency monitor: {e}")
                time.sleep(2 * 60)  # Wait 2 minutes before retry
    
    def _milestone_monitor_thread(self):
        """Monitor milestone progress"""
        while self.running:
            try:
                print("🎯 Syncing milestones...")
                
                # Sync all milestones
                self._run_script("sync_milestones.py", ["--sync", "--report"])
                
                # Check for blocking milestones
                self._run_script("sync_milestones.py", ["--blocking"])
                
                # Wait 15 minutes
                time.sleep(15 * 60)
                
            except Exception as e:
                print(f"❌ Error in milestone monitor: {e}")
                time.sleep(3 * 60)  # Wait 3 minutes before retry
    
    def _integration_monitor_thread(self):
        """Monitor integration testing"""
        while self.running:
            try:
                print("🧪 Running integration tests...")
                
                # Run integration tests
                self._run_script("run_integration_tests.py", ["--all"])
                
                # Check system readiness
                self._run_script("run_integration_tests.py", ["--readiness"])
                
                # Wait 1 hour
                time.sleep(60 * 60)
                
            except Exception as e:
                print(f"❌ Error in integration monitor: {e}")
                time.sleep(10 * 60)  # Wait 10 minutes before retry
    
    def _run_script(self, script_name, args=None):
        """Run a coordination script"""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return None
        
        command = [sys.executable, str(script_path)]
        if args:
            command.extend(args)
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.project_root)
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                print(f"⚠️ Script {script_name} failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Script {script_name} timed out")
            return None
        except Exception as e:
            print(f"❌ Error running script {script_name}: {e}")
            return None
    
    def stop_coordination_system(self):
        """Stop the coordination system"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.monitoring_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        print("✅ Coordination system stopped")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        print("📊 Generating system status report...")
        
        status_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_status": "running" if self.running else "stopped",
            "monitoring_threads": len([t for t in self.monitoring_threads if t.is_alive()]),
            "terminal_status": {},
            "dependencies": {},
            "milestones": {},
            "integration_tests": {}
        }
        
        # Get terminal status
        try:
            terminal1_result = self._run_script("update_terminal1_status.py", ["--report"])
            if terminal1_result:
                # Parse the JSON output
                lines = terminal1_result.strip().split('\n')
                for line in lines:
                    if line.startswith('{'):
                        status_report["terminal_status"]["terminal1"] = json.loads(line)
                        break
        except Exception as e:
            status_report["terminal_status"]["terminal1"] = {"error": str(e)}
        
        try:
            terminal2_result = self._run_script("update_terminal2_status.py", ["--report"])
            if terminal2_result:
                # Parse the JSON output
                lines = terminal2_result.strip().split('\n')
                for line in lines:
                    if line.startswith('{'):
                        status_report["terminal_status"]["terminal2"] = json.loads(line)
                        break
        except Exception as e:
            status_report["terminal_status"]["terminal2"] = {"error": str(e)}
        
        # Get dependencies
        try:
            dependency_result = self._run_script("check_dependencies.py", ["--all"])
            if dependency_result:
                lines = dependency_result.strip().split('\n')
                for line in lines:
                    if line.startswith('{'):
                        status_report["dependencies"] = json.loads(line)
                        break
        except Exception as e:
            status_report["dependencies"] = {"error": str(e)}
        
        # Get milestones
        try:
            milestone_result = self._run_script("sync_milestones.py", ["--report"])
            if milestone_result:
                # Extract milestone information from output
                status_report["milestones"] = {"status": "synced"}
        except Exception as e:
            status_report["milestones"] = {"error": str(e)}
        
        # Save status report
        report_file = self.coordination_dir / "documentation" / "shared_findings" / f"system_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(report_file.parent, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(status_report, f, indent=2)
        
        return status_report
    
    def emergency_stop(self):
        """Emergency stop of all coordination activities"""
        print("🚨 EMERGENCY STOP INITIATED")
        self.running = False
        
        # Stop all monitoring threads immediately
        for thread in self.monitoring_threads:
            if thread.is_alive():
                # Force thread termination (not ideal but necessary for emergency)
                pass
        
        # Log emergency stop
        emergency_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "emergency_stop",
            "reason": "Manual emergency stop initiated"
        }
        
        emergency_file = self.coordination_dir / "documentation" / "issue_tracking" / f"emergency_stop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(emergency_file.parent, exist_ok=True)
        with open(emergency_file, 'w') as f:
            json.dump(emergency_log, f, indent=2)
        
        print("🛑 Emergency stop complete")
    
    def restart_coordination_system(self):
        """Restart the coordination system"""
        print("🔄 Restarting coordination system...")
        
        # Stop current system
        self.stop_coordination_system()
        
        # Reset state
        self.running = True
        self.monitoring_threads = []
        
        # Start again
        self.start_coordination_system()

def signal_handler(signum, frame):
    """Handle system signals"""
    print(f"\n🔔 Received signal {signum}")
    if hasattr(signal_handler, 'coordination_master'):
        signal_handler.coordination_master.emergency_stop()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Coordination Master")
    parser.add_argument("--start", action="store_true", help="Start coordination system")
    parser.add_argument("--stop", action="store_true", help="Stop coordination system")
    parser.add_argument("--status", action="store_true", help="Get system status")
    parser.add_argument("--restart", action="store_true", help="Restart coordination system")
    parser.add_argument("--emergency-stop", action="store_true", help="Emergency stop")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    args = parser.parse_args()
    
    coordination_master = CoordinationMaster()
    
    # Set up signal handling
    signal_handler.coordination_master = coordination_master
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.start or args.daemon:
        coordination_master.start_coordination_system()
    
    if args.status:
        status = coordination_master.get_system_status()
        print(json.dumps(status, indent=2))
    
    if args.emergency_stop:
        coordination_master.emergency_stop()
    
    if args.restart:
        coordination_master.restart_coordination_system()
    
    if not any([args.start, args.stop, args.status, args.restart, args.emergency_stop, args.daemon]):
        print("Terminal Coordination Master")
        print("Usage: coordination_master.py --start|--stop|--status|--restart|--emergency-stop")
        print("\nAvailable commands:")
        print("  --start          Start coordination system")
        print("  --daemon         Run as daemon (same as --start)")
        print("  --status         Get system status")
        print("  --restart        Restart coordination system")
        print("  --emergency-stop Emergency stop all activities")

if __name__ == "__main__":
    main()