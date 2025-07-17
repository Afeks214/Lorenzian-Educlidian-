#!/usr/bin/env python3
"""
Memory Recovery Script
=====================

Automated recovery script for memory-related issues.
Handles memory leaks, out-of-memory conditions, and high memory usage.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryStatus:
    """Current memory status of a service."""
    service_name: str
    current_usage_mb: float
    current_percentage: float
    limit_mb: float
    available_mb: float
    swap_usage_mb: float
    cache_usage_mb: float
    buffer_usage_mb: float


class MemoryRecoveryHandler:
    """Handles memory-related recovery actions."""
    
    def __init__(self):
        self.recovery_history = []
        self.memory_thresholds = {
            'critical': 95.0,
            'high': 85.0,
            'warning': 75.0
        }
    
    async def assess_memory_situation(self, service_name: str) -> MemoryStatus:
        """Assess current memory situation for a service."""
        try:
            # Get memory stats from Kubernetes
            cmd = f"""
            kubectl top pod -l app={service_name} --no-headers | awk '{{
                gsub(/Mi/, "", $3); 
                gsub(/Gi/, "", $3); 
                if ($3 ~ /Gi/) $3 = $3 * 1024; 
                print $1, $3
            }}'
            """
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                lines = stdout.decode().strip().split('\n')
                if lines and lines[0]:
                    # Parse memory usage
                    parts = lines[0].split()
                    if len(parts) >= 2:
                        current_usage_mb = float(parts[1])
                        
                        # Get memory limit
                        limit_mb = await self._get_memory_limit(service_name)
                        
                        return MemoryStatus(
                            service_name=service_name,
                            current_usage_mb=current_usage_mb,
                            current_percentage=(current_usage_mb / limit_mb) * 100 if limit_mb > 0 else 0,
                            limit_mb=limit_mb,
                            available_mb=limit_mb - current_usage_mb,
                            swap_usage_mb=0,  # K8s doesn't typically use swap
                            cache_usage_mb=0,
                            buffer_usage_mb=0
                        )
            
            # Default status if unable to get metrics
            return MemoryStatus(
                service_name=service_name,
                current_usage_mb=0,
                current_percentage=0,
                limit_mb=512,  # Default 512MB
                available_mb=512,
                swap_usage_mb=0,
                cache_usage_mb=0,
                buffer_usage_mb=0
            )
            
        except Exception as e:
            logger.error(f"Failed to assess memory situation for {service_name}: {e}")
            raise
    
    async def _get_memory_limit(self, service_name: str) -> float:
        """Get memory limit for a service."""
        try:
            cmd = f"kubectl get deployment {service_name} -o jsonpath='{{.spec.template.spec.containers[0].resources.limits.memory}}'"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                limit_str = stdout.decode().strip()
                
                # Parse memory limit
                if limit_str.endswith('Mi'):
                    return float(limit_str[:-2])
                elif limit_str.endswith('Gi'):
                    return float(limit_str[:-2]) * 1024
                elif limit_str.endswith('Ki'):
                    return float(limit_str[:-2]) / 1024
                else:
                    return float(limit_str) / (1024 * 1024)  # Assume bytes
            
            return 512  # Default 512MB
            
        except Exception as e:
            logger.error(f"Failed to get memory limit for {service_name}: {e}")
            return 512
    
    async def clear_memory_cache(self, service_name: str) -> Dict[str, Any]:
        """Clear memory cache for a service."""
        try:
            logger.info(f"Clearing memory cache for {service_name}")
            
            # Clear page cache on the nodes (requires privileged access)
            cmd = f"""
            kubectl get pods -l app={service_name} -o jsonpath='{{.items[0].spec.nodeName}}' | xargs -I {{}} kubectl debug node/{{}} -it --image=busybox -- sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
            """
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            return {
                'success': success,
                'action': 'clear_memory_cache',
                'output': stdout.decode() if success else stderr.decode(),
                'service_name': service_name
            }
            
        except Exception as e:
            logger.error(f"Failed to clear memory cache for {service_name}: {e}")
            return {
                'success': False,
                'action': 'clear_memory_cache',
                'error': str(e),
                'service_name': service_name
            }
    
    async def force_garbage_collection(self, service_name: str) -> Dict[str, Any]:
        """Force garbage collection in application."""
        try:
            logger.info(f"Forcing garbage collection for {service_name}")
            
            # Send signal to application to trigger GC
            # This assumes the application handles SIGUSR1 for GC
            cmd = f"kubectl exec -l app={service_name} -- kill -USR1 1"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            return {
                'success': success,
                'action': 'force_garbage_collection',
                'output': stdout.decode() if success else stderr.decode(),
                'service_name': service_name
            }
            
        except Exception as e:
            logger.error(f"Failed to force garbage collection for {service_name}: {e}")
            return {
                'success': False,
                'action': 'force_garbage_collection',
                'error': str(e),
                'service_name': service_name
            }
    
    async def increase_memory_limit(self, service_name: str, increase_percentage: int = 50) -> Dict[str, Any]:
        """Increase memory limit for a service."""
        try:
            logger.info(f"Increasing memory limit for {service_name} by {increase_percentage}%")
            
            # Get current memory limit
            current_limit = await self._get_memory_limit(service_name)
            new_limit = int(current_limit * (1 + increase_percentage / 100))
            
            # Update deployment
            patch = {
                'spec': {
                    'template': {
                        'spec': {
                            'containers': [{
                                'name': service_name,
                                'resources': {
                                    'limits': {
                                        'memory': f"{new_limit}Mi"
                                    },
                                    'requests': {
                                        'memory': f"{int(new_limit * 0.8)}Mi"
                                    }
                                }
                            }]
                        }
                    }
                }
            }
            
            patch_json = json.dumps(patch)
            cmd = f"kubectl patch deployment {service_name} -p '{patch_json}'"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            if success:
                # Wait for rollout to complete
                await self._wait_for_rollout(service_name)
            
            return {
                'success': success,
                'action': 'increase_memory_limit',
                'output': stdout.decode() if success else stderr.decode(),
                'service_name': service_name,
                'old_limit_mb': current_limit,
                'new_limit_mb': new_limit,
                'increase_percentage': increase_percentage
            }
            
        except Exception as e:
            logger.error(f"Failed to increase memory limit for {service_name}: {e}")
            return {
                'success': False,
                'action': 'increase_memory_limit',
                'error': str(e),
                'service_name': service_name
            }
    
    async def restart_with_memory_optimization(self, service_name: str) -> Dict[str, Any]:
        """Restart service with memory optimization flags."""
        try:
            logger.info(f"Restarting {service_name} with memory optimization")
            
            # Add memory optimization environment variables
            env_vars = {
                'PYTHONOPTIMIZE': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'MALLOC_ARENA_MAX': '2',
                'MALLOC_MMAP_THRESHOLD_': '131072',
                'MALLOC_TRIM_THRESHOLD_': '131072',
                'MALLOC_TOP_PAD_': '131072',
                'MALLOC_MMAP_MAX_': '65536'
            }
            
            # Create patch with environment variables
            env_patch = []
            for key, value in env_vars.items():
                env_patch.append({'name': key, 'value': value})
            
            patch = {
                'spec': {
                    'template': {
                        'spec': {
                            'containers': [{
                                'name': service_name,
                                'env': env_patch
                            }]
                        }
                    }
                }
            }
            
            patch_json = json.dumps(patch)
            cmd = f"kubectl patch deployment {service_name} -p '{patch_json}'"
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            if success:
                # Trigger rolling restart
                restart_cmd = f"kubectl rollout restart deployment/{service_name}"
                restart_process = await asyncio.create_subprocess_shell(
                    restart_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await restart_process.communicate()
                
                # Wait for rollout to complete
                await self._wait_for_rollout(service_name)
            
            return {
                'success': success,
                'action': 'restart_with_memory_optimization',
                'output': stdout.decode() if success else stderr.decode(),
                'service_name': service_name,
                'optimization_flags': env_vars
            }
            
        except Exception as e:
            logger.error(f"Failed to restart {service_name} with memory optimization: {e}")
            return {
                'success': False,
                'action': 'restart_with_memory_optimization',
                'error': str(e),
                'service_name': service_name
            }
    
    async def _wait_for_rollout(self, service_name: str, timeout: int = 300):
        """Wait for deployment rollout to complete."""
        cmd = f"kubectl rollout status deployment/{service_name} --timeout={timeout}s"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
    
    async def emergency_memory_recovery(self, service_name: str) -> Dict[str, Any]:
        """Emergency memory recovery procedure."""
        try:
            logger.warning(f"Starting emergency memory recovery for {service_name}")
            
            # Get current memory status
            memory_status = await self.assess_memory_situation(service_name)
            
            recovery_actions = []
            
            # Step 1: Clear cache
            cache_result = await self.clear_memory_cache(service_name)
            recovery_actions.append(cache_result)
            
            # Step 2: Force garbage collection
            gc_result = await self.force_garbage_collection(service_name)
            recovery_actions.append(gc_result)
            
            # Wait and reassess
            await asyncio.sleep(10)
            new_memory_status = await self.assess_memory_situation(service_name)
            
            # Step 3: If still critical, increase memory limit
            if new_memory_status.current_percentage > self.memory_thresholds['critical']:
                increase_result = await self.increase_memory_limit(service_name, 100)  # Double memory
                recovery_actions.append(increase_result)
            
            # Step 4: If still critical, restart with optimization
            if new_memory_status.current_percentage > self.memory_thresholds['critical']:
                restart_result = await self.restart_with_memory_optimization(service_name)
                recovery_actions.append(restart_result)
            
            # Final assessment
            final_memory_status = await self.assess_memory_situation(service_name)
            
            overall_success = final_memory_status.current_percentage < self.memory_thresholds['critical']
            
            recovery_record = {
                'service_name': service_name,
                'timestamp': time.time(),
                'initial_memory_status': memory_status,
                'final_memory_status': final_memory_status,
                'recovery_actions': recovery_actions,
                'overall_success': overall_success,
                'recovery_type': 'emergency'
            }
            
            self.recovery_history.append(recovery_record)
            
            return {
                'success': overall_success,
                'action': 'emergency_memory_recovery',
                'service_name': service_name,
                'initial_memory_percentage': memory_status.current_percentage,
                'final_memory_percentage': final_memory_status.current_percentage,
                'actions_taken': len(recovery_actions),
                'successful_actions': len([a for a in recovery_actions if a.get('success', False)]),
                'recovery_record': recovery_record
            }
            
        except Exception as e:
            logger.error(f"Emergency memory recovery failed for {service_name}: {e}")
            return {
                'success': False,
                'action': 'emergency_memory_recovery',
                'error': str(e),
                'service_name': service_name
            }
    
    async def proactive_memory_management(self, service_name: str) -> Dict[str, Any]:
        """Proactive memory management to prevent issues."""
        try:
            logger.info(f"Starting proactive memory management for {service_name}")
            
            memory_status = await self.assess_memory_situation(service_name)
            
            actions_taken = []
            
            # If memory usage is high but not critical, take proactive measures
            if memory_status.current_percentage > self.memory_thresholds['high']:
                # Clear cache
                cache_result = await self.clear_memory_cache(service_name)
                actions_taken.append(cache_result)
                
                # Force garbage collection
                gc_result = await self.force_garbage_collection(service_name)
                actions_taken.append(gc_result)
                
                # Wait and reassess
                await asyncio.sleep(5)
                new_memory_status = await self.assess_memory_situation(service_name)
                
                # If still high, consider increasing memory limit
                if new_memory_status.current_percentage > self.memory_thresholds['high']:
                    increase_result = await self.increase_memory_limit(service_name, 25)  # 25% increase
                    actions_taken.append(increase_result)
                
                final_memory_status = await self.assess_memory_situation(service_name)
            else:
                final_memory_status = memory_status
            
            recovery_record = {
                'service_name': service_name,
                'timestamp': time.time(),
                'initial_memory_status': memory_status,
                'final_memory_status': final_memory_status,
                'recovery_actions': actions_taken,
                'overall_success': final_memory_status.current_percentage < self.memory_thresholds['high'],
                'recovery_type': 'proactive'
            }
            
            self.recovery_history.append(recovery_record)
            
            return {
                'success': True,
                'action': 'proactive_memory_management',
                'service_name': service_name,
                'initial_memory_percentage': memory_status.current_percentage,
                'final_memory_percentage': final_memory_status.current_percentage,
                'actions_taken': len(actions_taken),
                'successful_actions': len([a for a in actions_taken if a.get('success', False)]),
                'recovery_record': recovery_record
            }
            
        except Exception as e:
            logger.error(f"Proactive memory management failed for {service_name}: {e}")
            return {
                'success': False,
                'action': 'proactive_memory_management',
                'error': str(e),
                'service_name': service_name
            }
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Get memory recovery history."""
        return self.recovery_history
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get memory recovery statistics."""
        if not self.recovery_history:
            return {
                'total_recoveries': 0,
                'successful_recoveries': 0,
                'success_rate': 0.0,
                'average_recovery_time': 0.0,
                'most_common_actions': []
            }
        
        total_recoveries = len(self.recovery_history)
        successful_recoveries = len([r for r in self.recovery_history if r['overall_success']])
        
        # Calculate action frequency
        action_counts = {}
        for record in self.recovery_history:
            for action in record['recovery_actions']:
                action_name = action.get('action', 'unknown')
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        most_common_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_recoveries': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'success_rate': successful_recoveries / total_recoveries if total_recoveries > 0 else 0.0,
            'emergency_recoveries': len([r for r in self.recovery_history if r['recovery_type'] == 'emergency']),
            'proactive_recoveries': len([r for r in self.recovery_history if r['recovery_type'] == 'proactive']),
            'most_common_actions': most_common_actions
        }


# Global memory recovery handler
memory_recovery_handler = MemoryRecoveryHandler()


async def main():
    """Main function for standalone execution."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python memory_recovery.py <service_name> <action>")
        print("Actions: assess, emergency, proactive, clear_cache, force_gc, increase_limit")
        sys.exit(1)
    
    service_name = sys.argv[1]
    action = sys.argv[2]
    
    handler = MemoryRecoveryHandler()
    
    if action == 'assess':
        status = await handler.assess_memory_situation(service_name)
        print(f"Memory Status for {service_name}:")
        print(f"  Current Usage: {status.current_usage_mb:.1f}MB ({status.current_percentage:.1f}%)")
        print(f"  Memory Limit: {status.limit_mb:.1f}MB")
        print(f"  Available: {status.available_mb:.1f}MB")
    
    elif action == 'emergency':
        result = await handler.emergency_memory_recovery(service_name)
        print(f"Emergency Recovery Result: {result}")
    
    elif action == 'proactive':
        result = await handler.proactive_memory_management(service_name)
        print(f"Proactive Management Result: {result}")
    
    elif action == 'clear_cache':
        result = await handler.clear_memory_cache(service_name)
        print(f"Clear Cache Result: {result}")
    
    elif action == 'force_gc':
        result = await handler.force_garbage_collection(service_name)
        print(f"Force GC Result: {result}")
    
    elif action == 'increase_limit':
        increase_pct = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        result = await handler.increase_memory_limit(service_name, increase_pct)
        print(f"Increase Limit Result: {result}")
    
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())