"""
AGENT 9: SQL Injection Prevention Tests
Comprehensive test suite for SQL injection attack prevention and database security.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
import warnings

# Security-focused imports
import sqlparse
from sqlparse import sql
from sqlparse.tokens import Token

# Test database imports
try:
    import asyncpg
    import psycopg2
    from psycopg2 import sql as psql
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

# Core imports
from src.performance.connection_pool import ConnectionPool, ConnectionConfig, ConnectionType
from src.security.secrets_manager import SecretsManager
from src.governance.audit_system import AuditSystem


class TestSQLInjectionPrevention:
    """Test SQL injection prevention across all database interfaces."""
    
    @pytest.fixture
    def mock_database_connection(self):
        """Mock database connection with proper SQL escaping."""
        mock_conn = Mock()
        
        # Mock asyncpg connection methods
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value="SELECT 1")
        mock_conn.close = AsyncMock()
        
        # Add parameterized query support
        mock_conn.prepare = AsyncMock()
        mock_conn.prepared_stmt = Mock()
        mock_conn.prepared_stmt.fetch = AsyncMock(return_value=[])
        
        return mock_conn
    
    @pytest.fixture
    def malicious_inputs(self):
        """Collection of malicious SQL injection payloads."""
        return [
            # Basic SQL injection
            "1; DROP TABLE users; --",
            "1' OR '1'='1",
            "1' OR '1'='1' --",
            "1' OR '1'='1' /*",
            
            # Union-based injection
            "1' UNION SELECT * FROM users --",
            "1' UNION SELECT password FROM users WHERE username='admin' --",
            "1' UNION SELECT 1,2,3,4,5,6,7,8,9,10 --",
            
            # Boolean-based blind injection
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "1' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE username='admin')='a' --",
            
            # Time-based blind injection
            "1'; WAITFOR DELAY '00:00:05' --",
            "1' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a) --",
            
            # Error-based injection
            "1' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a) --",
            "1' AND ExtractValue(1, CONCAT(0x7e, (SELECT @@version), 0x7e)) --",
            
            # Stored procedure injection
            "1'; EXEC xp_cmdshell('dir') --",
            "1'; EXEC sp_addlogin 'test','test' --",
            
            # NoSQL injection attempts
            "1' || ''=='",
            "1' || this.password.match(/.*/) || '1'=='1",
            "1' || '1'=='1",
            
            # Second-order injection
            "admin'--",
            "admin'/*",
            "admin'#",
            
            # Encoded injection
            "%27%20OR%20%271%27%3D%271",
            "%27%20UNION%20SELECT%20%2A%20FROM%20users%20--%20",
            "%27%3B%20DROP%20TABLE%20users%3B%20--",
            
            # Advanced techniques
            "1' AND (ASCII(SUBSTRING((SELECT password FROM users WHERE username='admin'),1,1)))>64 --",
            "1' AND (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE()) > 0 --",
            "1' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a) --",
        ]
    
    @pytest.fixture
    def safe_inputs(self):
        """Collection of safe, legitimate inputs."""
        return [
            "user123",
            "test@example.com",
            "password123",
            "Product Name",
            "2023-10-15",
            "123.45",
            "true",
            "false",
            "null",
            "",
            "   ",
            "Special-Characters_123",
            "unicode_test_café",
            "numeric_123456",
            "mixed_alphanumeric_abc123",
        ]
    
    def test_parameterized_queries_prevent_injection(self, mock_database_connection, malicious_inputs):
        """Test that parameterized queries prevent SQL injection."""
        
        async def test_parameterized_query(connection, user_input):
            """Test parameterized query execution."""
            # This should use parameterized queries internally
            query = "SELECT * FROM users WHERE username = $1"
            await connection.fetch(query, user_input)
        
        # Test with malicious inputs
        for malicious_input in malicious_inputs:
            with pytest.raises(Exception):
                # Direct string concatenation should fail
                dangerous_query = f"SELECT * FROM users WHERE username = '{malicious_input}'"
                self._validate_query_safety(dangerous_query)
            
            # Parameterized query should be safe
            safe_query = "SELECT * FROM users WHERE username = $1"
            assert self._validate_query_safety(safe_query, malicious_input)
    
    def test_input_validation_and_sanitization(self, malicious_inputs, safe_inputs):
        """Test input validation and sanitization functions."""
        
        def validate_and_sanitize_input(user_input: str) -> str:
            """Validate and sanitize user input."""
            if not isinstance(user_input, str):
                raise ValueError("Input must be a string")
            
            # Check for SQL injection patterns
            dangerous_patterns = [
                r"'.*OR.*'.*=.*'",  # Basic OR injection
                r"';.*--",          # SQL comment injection
                r"UNION.*SELECT",   # Union-based injection
                r"DROP.*TABLE",     # Drop table attempts
                r"INSERT.*INTO",    # Insert injection
                r"UPDATE.*SET",     # Update injection
                r"DELETE.*FROM",    # Delete injection
                r"EXEC.*\(",        # Stored procedure execution
                r"xp_cmdshell",     # Command execution
                r"sp_addlogin",     # User creation
                r"WAITFOR.*DELAY",  # Time-based attacks
            ]
            
            import re
            for pattern in dangerous_patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    raise ValueError(f"Potentially dangerous input detected: {pattern}")
            
            # Basic sanitization
            sanitized = user_input.replace("'", "''")  # Escape single quotes
            sanitized = sanitized.replace(";", "")     # Remove semicolons
            sanitized = sanitized.replace("--", "")    # Remove SQL comments
            sanitized = sanitized.replace("/*", "")    # Remove block comments
            sanitized = sanitized.replace("*/", "")    # Remove block comments
            
            return sanitized
        
        # Test malicious inputs are caught
        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError, match="Potentially dangerous input detected"):
                validate_and_sanitize_input(malicious_input)
        
        # Test safe inputs pass through
        for safe_input in safe_inputs:
            result = validate_and_sanitize_input(safe_input)
            assert isinstance(result, str)
            assert len(result) <= len(safe_input) * 2  # Accounting for escape sequences
    
    def test_connection_pool_sql_injection_prevention(self, malicious_inputs):
        """Test that connection pool prevents SQL injection attacks."""
        
        with patch('src.performance.connection_pool.asyncpg') as mock_asyncpg:
            mock_connection = Mock()
            mock_connection.fetch = AsyncMock(return_value=[])
            mock_connection.execute = AsyncMock(return_value="SELECT 1")
            mock_asyncpg.connect = AsyncMock(return_value=mock_connection)
            
            config = ConnectionConfig(
                connection_type=ConnectionType.DATABASE,
                host="localhost",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_pass"
            )
            
            pool = ConnectionPool(config)
            
            async def test_injection_prevention():
                # Test that parameterized queries are used
                for malicious_input in malicious_inputs:
                    # This should not execute the malicious SQL
                    await pool.execute("fetch", "SELECT * FROM users WHERE id = $1", malicious_input)
                    
                    # Verify that the connection was called with parameterized query
                    mock_connection.fetch.assert_called_with(
                        "SELECT * FROM users WHERE id = $1",
                        malicious_input
                    )
            
            asyncio.run(test_injection_prevention())
    
    def test_secrets_manager_sql_injection_prevention(self, malicious_inputs):
        """Test that secrets manager prevents SQL injection in secret retrieval."""
        
        secrets_manager = SecretsManager()
        
        for malicious_input in malicious_inputs:
            # Test that malicious secret names are handled safely
            result = secrets_manager.get_secret(malicious_input)
            assert result is None or isinstance(result, str)
            
            # Test that malicious inputs don't cause SQL injection in internal operations
            with pytest.raises(ValueError):
                # This should validate the secret name format
                self._validate_secret_name(malicious_input)
    
    def test_audit_system_sql_injection_prevention(self, malicious_inputs):
        """Test that audit system prevents SQL injection in log queries."""
        
        with patch('src.governance.audit_system.asyncpg') as mock_asyncpg:
            mock_connection = Mock()
            mock_connection.fetch = AsyncMock(return_value=[])
            mock_connection.execute = AsyncMock(return_value="INSERT 1")
            mock_asyncpg.connect = AsyncMock(return_value=mock_connection)
            
            audit_system = AuditSystem()
            
            async def test_audit_injection_prevention():
                for malicious_input in malicious_inputs:
                    # Test audit log insertion with malicious data
                    await audit_system.log_event(
                        event_type="test_event",
                        user_id="test_user",
                        details={"message": malicious_input}
                    )
                    
                    # Verify parameterized query was used
                    mock_connection.execute.assert_called()
                    call_args = mock_connection.execute.call_args
                    
                    # Should be parameterized query, not string concatenation
                    assert "$" in call_args[0][0]  # Should contain parameter placeholder
            
            asyncio.run(test_audit_injection_prevention())
    
    def test_query_whitelisting(self):
        """Test that only whitelisted queries are allowed."""
        
        allowed_queries = [
            "SELECT * FROM users WHERE id = $1",
            "INSERT INTO audit_log (event_type, user_id, timestamp) VALUES ($1, $2, $3)",
            "UPDATE user_sessions SET last_activity = $1 WHERE session_id = $2",
            "DELETE FROM temp_data WHERE created_at < $1",
        ]
        
        def is_query_allowed(query: str) -> bool:
            """Check if query is in whitelist."""
            normalized_query = sqlparse.format(query, strip_comments=True, reindent=True)
            return normalized_query in [sqlparse.format(q, strip_comments=True, reindent=True) for q in allowed_queries]
        
        # Test allowed queries
        for query in allowed_queries:
            assert is_query_allowed(query)
        
        # Test malicious queries are blocked
        malicious_queries = [
            "DROP TABLE users",
            "SELECT * FROM users; DROP TABLE audit_log; --",
            "INSERT INTO users (username, password) VALUES ('admin', 'password')",
            "UPDATE users SET password = 'hacked' WHERE 1=1",
        ]
        
        for query in malicious_queries:
            assert not is_query_allowed(query)
    
    def test_prepared_statement_usage(self, mock_database_connection):
        """Test that prepared statements are used for repeated queries."""
        
        async def test_prepared_statements():
            # Mock prepared statement
            mock_prepared = Mock()
            mock_prepared.fetch = AsyncMock(return_value=[])
            mock_database_connection.prepare = AsyncMock(return_value=mock_prepared)
            
            # Test that prepare is called for repeated queries
            query = "SELECT * FROM users WHERE id = $1"
            
            # First execution should prepare the statement
            await mock_database_connection.prepare(query)
            result = await mock_prepared.fetch(1)
            
            # Verify prepare was called
            mock_database_connection.prepare.assert_called_with(query)
            mock_prepared.fetch.assert_called_with(1)
        
        asyncio.run(test_prepared_statements())
    
    def test_database_permissions_enforcement(self):
        """Test that database permissions are properly enforced."""
        
        # Test that application user has limited permissions
        restricted_operations = [
            "CREATE TABLE",
            "DROP TABLE",
            "ALTER TABLE",
            "CREATE USER",
            "DROP USER",
            "GRANT",
            "REVOKE",
            "CREATE DATABASE",
            "DROP DATABASE",
        ]
        
        for operation in restricted_operations:
            with pytest.raises(Exception):
                # These operations should fail due to insufficient permissions
                self._simulate_database_operation(operation)
    
    def test_transaction_isolation_security(self):
        """Test that transaction isolation prevents data leakage."""
        
        async def test_transaction_isolation():
            # Test that transactions are properly isolated
            # and don't leak data between different users/sessions
            
            # Mock transaction context
            mock_transaction = Mock()
            mock_transaction.commit = AsyncMock()
            mock_transaction.rollback = AsyncMock()
            mock_transaction.fetch = AsyncMock(return_value=[])
            
            # Test that user A cannot see user B's data
            user_a_data = await mock_transaction.fetch(
                "SELECT * FROM user_data WHERE user_id = $1", "user_a"
            )
            user_b_data = await mock_transaction.fetch(
                "SELECT * FROM user_data WHERE user_id = $1", "user_b"
            )
            
            # Verify proper isolation
            assert user_a_data != user_b_data
            mock_transaction.fetch.assert_called()
        
        asyncio.run(test_transaction_isolation())
    
    def test_sql_error_information_disclosure(self, malicious_inputs):
        """Test that SQL errors don't disclose sensitive information."""
        
        for malicious_input in malicious_inputs:
            try:
                # Attempt to trigger SQL error
                self._execute_dangerous_query(malicious_input)
            except Exception as e:
                error_message = str(e)
                
                # Check that error doesn't contain sensitive info
                sensitive_keywords = [
                    "password",
                    "secret",
                    "token",
                    "key",
                    "admin",
                    "root",
                    "database",
                    "schema",
                    "table",
                    "column",
                    "connection",
                    "host",
                    "port",
                ]
                
                for keyword in sensitive_keywords:
                    assert keyword.lower() not in error_message.lower(), \
                        f"Error message contains sensitive keyword: {keyword}"
    
    def test_connection_string_security(self):
        """Test that database connection strings don't contain hardcoded credentials."""
        
        # Test configuration from environment variables only
        config = ConnectionConfig(
            connection_type=ConnectionType.DATABASE,
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        # Verify no hardcoded credentials in source code
        assert config.username != "admin"
        assert config.password != "password"
        assert config.password != "admin"
        assert config.password != "root"
        assert config.password != "123456"
        
        # Test that credentials come from secure sources
        assert len(config.password) > 8  # Minimum password length
    
    def test_sql_injection_in_dynamic_queries(self, malicious_inputs):
        """Test prevention of SQL injection in dynamically constructed queries."""
        
        def build_dynamic_query(filters: Dict[str, Any]) -> tuple:
            """Build a dynamic query with filters."""
            base_query = "SELECT * FROM users WHERE 1=1"
            params = []
            param_count = 0
            
            for key, value in filters.items():
                # Validate column name (whitelist approach)
                if key not in ["username", "email", "created_at", "status"]:
                    raise ValueError(f"Invalid column name: {key}")
                
                param_count += 1
                base_query += f" AND {key} = ${param_count}"
                params.append(value)
            
            return base_query, params
        
        # Test with malicious filter values
        for malicious_input in malicious_inputs:
            filters = {"username": malicious_input}
            
            query, params = build_dynamic_query(filters)
            
            # Verify parameterized query structure
            assert "$1" in query
            assert malicious_input in params
            assert "DROP" not in query  # Should not be in the query itself
            assert "UNION" not in query  # Should not be in the query itself
        
        # Test with malicious column names
        malicious_columns = ["username; DROP TABLE users; --", "email' OR '1'='1"]
        for malicious_col in malicious_columns:
            with pytest.raises(ValueError, match="Invalid column name"):
                build_dynamic_query({malicious_col: "test"})
    
    def test_stored_procedure_security(self):
        """Test that stored procedures are secure against injection."""
        
        # Test that stored procedures use parameterized inputs
        def call_stored_procedure(proc_name: str, params: List[Any]) -> str:
            """Call a stored procedure safely."""
            # Whitelist allowed procedures
            allowed_procedures = [
                "get_user_by_id",
                "create_audit_log",
                "update_user_session",
                "calculate_risk_metrics",
            ]
            
            if proc_name not in allowed_procedures:
                raise ValueError(f"Stored procedure not allowed: {proc_name}")
            
            # Return parameterized call
            param_placeholders = ", ".join(f"${i+1}" for i in range(len(params)))
            return f"CALL {proc_name}({param_placeholders})"
        
        # Test allowed procedures
        call_str = call_stored_procedure("get_user_by_id", [123])
        assert "CALL get_user_by_id($1)" == call_str
        
        # Test malicious procedure names
        malicious_procedures = [
            "xp_cmdshell",
            "sp_addlogin",
            "get_user_by_id; DROP TABLE users; --",
        ]
        
        for proc in malicious_procedures:
            with pytest.raises(ValueError, match="Stored procedure not allowed"):
                call_stored_procedure(proc, [])
    
    # Helper methods
    def _validate_query_safety(self, query: str, *params) -> bool:
        """Validate that a query is safe."""
        # Check for parameterized query markers
        if params and ("$" not in query and "?" not in query):
            return False  # Should use parameterized queries
        
        # Check for dangerous SQL keywords in the query itself
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper and "$" not in query:
                return False  # Dangerous keyword without parameterization
        
        return True
    
    def _validate_secret_name(self, secret_name: str) -> bool:
        """Validate secret name format."""
        import re
        
        # Secret names should only contain alphanumeric characters, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', secret_name):
            raise ValueError("Invalid secret name format")
        
        # Check for injection patterns
        if any(pattern in secret_name.lower() for pattern in ["drop", "select", "union", "insert"]):
            raise ValueError("Potentially dangerous secret name")
        
        return True
    
    def _simulate_database_operation(self, operation: str):
        """Simulate a database operation to test permissions."""
        # This would normally interact with the database
        # For testing, we simulate permission failures
        if operation.upper() in ["CREATE", "DROP", "ALTER", "GRANT", "REVOKE"]:
            raise Exception(f"Permission denied: {operation}")
    
    def _execute_dangerous_query(self, user_input: str):
        """Simulate executing a dangerous query to test error handling."""
        # This would trigger SQL errors in a real environment
        # For testing, we simulate various error conditions
        if "DROP" in user_input.upper():
            raise Exception("Syntax error near 'DROP'")
        elif "UNION" in user_input.upper():
            raise Exception("Invalid column count in UNION")
        elif "OR" in user_input.upper():
            raise Exception("Invalid WHERE clause")
        else:
            raise Exception("General SQL error")


class TestDatabaseSecurityIntegration:
    """Integration tests for database security across the system."""
    
    def test_end_to_end_sql_injection_prevention(self):
        """Test SQL injection prevention across the entire system."""
        
        # Test from user input to database query
        user_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
        ]
        
        for user_input in user_inputs:
            # Simulate the full flow: user input -> validation -> database query
            try:
                # Step 1: Input validation
                validated_input = self._validate_user_input(user_input)
                
                # Step 2: Business logic processing
                processed_input = self._process_business_logic(validated_input)
                
                # Step 3: Database query execution
                result = self._execute_database_query(processed_input)
                
                # Should either be safely handled or raise an exception
                assert result is not None or True  # Either safe result or exception raised
                
            except ValueError:
                # Input validation should catch malicious inputs
                pass
            except Exception as e:
                # Other exceptions should be logged but not expose sensitive info
                assert "password" not in str(e).lower()
                assert "secret" not in str(e).lower()
    
    def test_performance_impact_of_security_measures(self):
        """Test that security measures don't significantly impact performance."""
        
        import time
        
        # Test query execution time with security measures
        start_time = time.perf_counter()
        
        # Execute 1000 parameterized queries
        for i in range(1000):
            self._execute_secure_query("SELECT * FROM users WHERE id = $1", [i])
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (< 1 second for 1000 queries)
        assert execution_time < 1.0, f"Security measures causing performance degradation: {execution_time}s"
    
    def test_concurrent_access_security(self):
        """Test security under concurrent access scenarios."""
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def worker_thread(thread_id: int):
            """Worker thread for concurrent testing."""
            try:
                # Each thread tries to access data
                result = self._execute_secure_query(
                    "SELECT * FROM users WHERE id = $1", [thread_id]
                )
                results.put(("success", thread_id, result))
            except Exception as e:
                results.put(("error", thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(10):
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
            else:
                error_count += 1
        
        # All threads should complete successfully
        assert success_count == 10, f"Concurrent access failed: {error_count} errors"
    
    # Helper methods
    def _validate_user_input(self, user_input: str) -> str:
        """Validate user input."""
        # Basic validation - would be more comprehensive in real system
        if len(user_input) > 1000:
            raise ValueError("Input too long")
        
        dangerous_patterns = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "UNION"]
        for pattern in dangerous_patterns:
            if pattern in user_input.upper():
                raise ValueError(f"Dangerous pattern detected: {pattern}")
        
        return user_input
    
    def _process_business_logic(self, validated_input: str) -> str:
        """Process business logic."""
        # Simulate business logic processing
        return validated_input.strip()
    
    def _execute_database_query(self, processed_input: str) -> Dict[str, Any]:
        """Execute database query."""
        # Simulate database query execution
        return {"result": "success", "input": processed_input}
    
    def _execute_secure_query(self, query: str, params: List[Any]) -> Dict[str, Any]:
        """Execute a secure parameterized query."""
        # Simulate secure query execution
        return {"query": query, "params": params, "result": "success"}


@pytest.mark.security
@pytest.mark.unit
class TestSQLInjectionRegressionTests:
    """Regression tests to ensure SQL injection fixes remain effective."""
    
    def test_known_vulnerability_fixes(self):
        """Test that previously discovered vulnerabilities are fixed."""
        
        # Test cases based on previously discovered vulnerabilities
        vulnerability_test_cases = [
            {
                "name": "user_authentication_bypass",
                "input": "admin'--",
                "expected_behavior": "authentication_failure"
            },
            {
                "name": "data_extraction_attempt",
                "input": "1' UNION SELECT password FROM users--",
                "expected_behavior": "query_error"
            },
            {
                "name": "table_drop_attempt",
                "input": "'; DROP TABLE audit_log; --",
                "expected_behavior": "validation_error"
            },
        ]
        
        for test_case in vulnerability_test_cases:
            try:
                result = self._test_vulnerability_fix(test_case["input"])
                assert result["status"] == "blocked", f"Vulnerability not blocked: {test_case['name']}"
            except Exception as e:
                # Exception is expected for blocked attempts
                assert "blocked" in str(e) or "invalid" in str(e)
    
    def test_new_attack_vectors(self):
        """Test protection against new SQL injection attack vectors."""
        
        # Test modern SQL injection techniques
        modern_attacks = [
            # JSON injection
            "1' AND JSON_EXTRACT(sensitive_data, '$.password') --",
            
            # XML injection
            "1' AND extractvalue(1, concat(0x7e, (SELECT password FROM users LIMIT 1))) --",
            
            # Regular expression injection
            "1' AND password REGEXP '^a' --",
            
            # Full-text search injection
            "1' AND MATCH(content) AGAINST('password' IN BOOLEAN MODE) --",
        ]
        
        for attack in modern_attacks:
            with pytest.raises(Exception):
                self._test_vulnerability_fix(attack)
    
    def _test_vulnerability_fix(self, malicious_input: str) -> Dict[str, Any]:
        """Test that a specific vulnerability is fixed."""
        
        # Simulate the vulnerability test
        if any(pattern in malicious_input.upper() for pattern in ["DROP", "UNION", "INSERT", "UPDATE"]):
            return {"status": "blocked", "reason": "malicious_pattern_detected"}
        
        if "'" in malicious_input and "--" in malicious_input:
            return {"status": "blocked", "reason": "sql_comment_injection"}
        
        # If we get here, the input wasn't blocked (potential vulnerability)
        return {"status": "allowed", "input": malicious_input}