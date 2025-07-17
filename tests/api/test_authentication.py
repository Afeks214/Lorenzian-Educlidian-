"""
Comprehensive authentication system testing with security focus.
Tests JWT validation, session management, security edge cases, and penetration testing.
"""

import pytest
import asyncio
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import redis.asyncio as redis
from freezegun import freeze_time

from src.api.authentication import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_EXPIRATION_HOURS,
    UserRole,
    RolePermission,
    ROLE_PERMISSIONS,
    UserInfo,
    LoginRequest,
    LoginResponse,
    TokenPayload,
    USERS_DB,
    verify_password,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    verify_token_payload,
    verify_token,
    login,
    logout,
    require_permission,
    require_role,
    audit_log,
    check_failed_login_attempts,
    record_failed_login,
    clear_failed_logins,
    init_redis,
    redis_client
)

# Test fixtures
@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    return redis_mock

@pytest.fixture
def mock_request():
    """Mock FastAPI request object."""
    request = Mock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-agent"}
    return request

@pytest.fixture
def valid_login_request():
    """Valid login request for testing."""
    return LoginRequest(
        username="admin",
        password="admin123!"
    )

@pytest.fixture
def mfa_login_request():
    """MFA login request for testing."""
    return LoginRequest(
        username="risk_manager",
        password="risk123!",
        mfa_token="123456"
    )

@pytest.fixture
def invalid_login_request():
    """Invalid login request for testing."""
    return LoginRequest(
        username="invalid_user",
        password="invalid_pass"
    )

@pytest.fixture
def valid_token():
    """Generate valid JWT token for testing."""
    user_data = USERS_DB["admin"]
    session_id = secrets.token_urlsafe(32)
    return create_access_token(user_data, session_id)

@pytest.fixture
def expired_token():
    """Generate expired JWT token for testing."""
    now = datetime.utcnow()
    expire = now - timedelta(hours=1)  # Expired 1 hour ago
    
    payload = {
        "user_id": "user_001",
        "username": "admin",
        "role": "system_admin",
        "session_id": "test_session",
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": "test_jti"
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

@pytest.fixture
def invalid_token():
    """Generate invalid JWT token for testing."""
    payload = {"invalid": "payload"}
    return jwt.encode(payload, "wrong_secret", algorithm=JWT_ALGORITHM)

class TestPasswordVerification:
    """Test password verification functionality."""
    
    def test_verify_password_success(self):
        """Test successful password verification."""
        password = "test_password123!"
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        result = asyncio.run(verify_password(password, hashed))
        assert result is True
    
    def test_verify_password_failure(self):
        """Test failed password verification."""
        password = "test_password123!"
        wrong_password = "wrong_password"
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        result = asyncio.run(verify_password(wrong_password, hashed))
        assert result is False
    
    def test_verify_password_empty_strings(self):
        """Test password verification with empty strings."""
        with pytest.raises(ValueError):
            asyncio.run(verify_password("", ""))

class TestUserAuthentication:
    """Test user authentication functionality."""
    
    def test_authenticate_user_success(self):
        """Test successful user authentication."""
        result = asyncio.run(authenticate_user("admin", "admin123!"))
        assert result is not None
        assert result["username"] == "admin"
        assert result["role"] == UserRole.SYSTEM_ADMIN
    
    def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password."""
        result = asyncio.run(authenticate_user("admin", "wrong_password"))
        assert result is None
    
    def test_authenticate_user_nonexistent_user(self):
        """Test authentication with non-existent user."""
        result = asyncio.run(authenticate_user("nonexistent", "any_password"))
        assert result is None
    
    def test_authenticate_user_inactive_user(self):
        """Test authentication with inactive user."""
        # Temporarily make user inactive
        original_active = USERS_DB["admin"]["active"]
        USERS_DB["admin"]["active"] = False
        
        try:
            result = asyncio.run(authenticate_user("admin", "admin123!"))
            assert result is None
        finally:
            USERS_DB["admin"]["active"] = original_active

class TestTokenCreation:
    """Test JWT token creation functionality."""
    
    def test_create_access_token(self):
        """Test access token creation."""
        user_data = USERS_DB["admin"]
        session_id = "test_session"
        
        token = create_access_token(user_data, session_id)
        
        # Verify token can be decoded
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["user_id"] == user_data["user_id"]
        assert payload["username"] == user_data["username"]
        assert payload["role"] == user_data["role"].value
        assert payload["session_id"] == session_id
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload
    
    def test_create_refresh_token(self):
        """Test refresh token creation."""
        user_id = "user_001"
        session_id = "test_session"
        
        token = create_refresh_token(user_id, session_id)
        
        # Verify token can be decoded
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["user_id"] == user_id
        assert payload["session_id"] == session_id
        assert payload["type"] == "refresh"
        assert "exp" in payload
    
    def test_token_expiration(self):
        """Test token expiration time."""
        user_data = USERS_DB["admin"]
        session_id = "test_session"
        
        with freeze_time("2023-01-01 12:00:00"):
            token = create_access_token(user_data, session_id)
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            expected_exp = datetime(2023, 1, 1, 12 + JWT_EXPIRATION_HOURS, 0, 0)
            actual_exp = datetime.fromtimestamp(payload["exp"])
            
            assert actual_exp == expected_exp

class TestTokenVerification:
    """Test JWT token verification functionality."""
    
    def test_verify_token_payload_success(self, valid_token):
        """Test successful token payload verification."""
        payload = asyncio.run(verify_token_payload(valid_token))
        assert isinstance(payload, TokenPayload)
        assert payload.username == "admin"
        assert payload.role == "system_admin"
    
    def test_verify_token_payload_expired(self, expired_token):
        """Test token payload verification with expired token."""
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(verify_token_payload(expired_token))
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()
    
    def test_verify_token_payload_invalid(self, invalid_token):
        """Test token payload verification with invalid token."""
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(verify_token_payload(invalid_token))
        assert exc_info.value.status_code == 401
        assert "invalid" in exc_info.value.detail.lower()
    
    def test_verify_token_payload_malformed(self):
        """Test token payload verification with malformed token."""
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(verify_token_payload("malformed.token"))
        assert exc_info.value.status_code == 401
    
    @patch('src.api.authentication.redis_client')
    def test_verify_token_success(self, mock_redis_client, valid_token):
        """Test successful token verification."""
        mock_redis_client.get.return_value = b'{"valid": "session"}'
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=valid_token
        )
        
        user_info = asyncio.run(verify_token(credentials))
        assert isinstance(user_info, UserInfo)
        assert user_info.username == "admin"
        assert user_info.role == UserRole.SYSTEM_ADMIN
    
    @patch('src.api.authentication.redis_client')
    def test_verify_token_session_expired(self, mock_redis_client, valid_token):
        """Test token verification with expired session."""
        mock_redis_client.get.return_value = None
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=valid_token
        )
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(verify_token(credentials))
        assert exc_info.value.status_code == 401
        assert "session expired" in exc_info.value.detail.lower()

class TestLoginProcess:
    """Test login process functionality."""
    
    @patch('src.api.authentication.redis_client')
    def test_login_success(self, mock_redis_client, valid_login_request):
        """Test successful login."""
        mock_redis_client.setex.return_value = True
        
        response = asyncio.run(login(valid_login_request))
        
        assert isinstance(response, LoginResponse)
        assert response.access_token is not None
        assert response.refresh_token is not None
        assert response.token_type == "bearer"
        assert response.expires_in == JWT_EXPIRATION_HOURS * 3600
        assert response.user_info.username == "admin"
        assert response.user_info.role == UserRole.SYSTEM_ADMIN
    
    def test_login_invalid_credentials(self, invalid_login_request):
        """Test login with invalid credentials."""
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(login(invalid_login_request))
        assert exc_info.value.status_code == 401
        assert "invalid credentials" in exc_info.value.detail.lower()
    
    @patch('src.api.authentication.redis_client')
    def test_login_mfa_required(self, mock_redis_client):
        """Test login with MFA required."""
        mock_redis_client.setex.return_value = True
        
        # Login request without MFA token
        request = LoginRequest(
            username="risk_manager",
            password="risk123!"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(login(request))
        assert exc_info.value.status_code == 401
        assert "mfa token required" in exc_info.value.detail.lower()
    
    @patch('src.api.authentication.redis_client')
    def test_login_mfa_success(self, mock_redis_client, mfa_login_request):
        """Test successful MFA login."""
        mock_redis_client.setex.return_value = True
        
        response = asyncio.run(login(mfa_login_request))
        
        assert isinstance(response, LoginResponse)
        assert response.user_info.username == "risk_manager"
        assert response.user_info.mfa_enabled is True
    
    @patch('src.api.authentication.redis_client')
    def test_login_mfa_invalid_token(self, mock_redis_client):
        """Test login with invalid MFA token."""
        mock_redis_client.setex.return_value = True
        
        request = LoginRequest(
            username="risk_manager",
            password="risk123!",
            mfa_token="invalid_token"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(login(request))
        assert exc_info.value.status_code == 401
        assert "invalid mfa token" in exc_info.value.detail.lower()

class TestLogout:
    """Test logout functionality."""
    
    @patch('src.api.authentication.redis_client')
    def test_logout_success(self, mock_redis_client):
        """Test successful logout."""
        mock_redis_client.delete.return_value = True
        
        result = asyncio.run(logout("test_session"))
        assert result is True
        mock_redis_client.delete.assert_called_once_with("session:test_session")
    
    @patch('src.api.authentication.redis_client')
    def test_logout_redis_error(self, mock_redis_client):
        """Test logout with Redis error."""
        mock_redis_client.delete.side_effect = Exception("Redis error")
        
        result = asyncio.run(logout("test_session"))
        assert result is False

class TestPermissionAndRoleChecking:
    """Test permission and role checking functionality."""
    
    def test_require_permission_success(self):
        """Test successful permission check."""
        user_info = UserInfo(
            user_id="user_001",
            username="admin",
            email="admin@test.com",
            role=UserRole.SYSTEM_ADMIN,
            permissions=ROLE_PERMISSIONS[UserRole.SYSTEM_ADMIN],
            session_id="test_session",
            login_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            mfa_enabled=False
        )
        
        permission_checker = require_permission(RolePermission.DASHBOARD_READ)
        result = permission_checker(user_info)
        assert result == user_info
    
    def test_require_permission_failure(self):
        """Test failed permission check."""
        user_info = UserInfo(
            user_id="user_001",
            username="viewer",
            email="viewer@test.com",
            role=UserRole.VIEWER,
            permissions=ROLE_PERMISSIONS[UserRole.VIEWER],
            session_id="test_session",
            login_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            mfa_enabled=False
        )
        
        permission_checker = require_permission(RolePermission.USER_MANAGEMENT)
        
        with pytest.raises(HTTPException) as exc_info:
            permission_checker(user_info)
        assert exc_info.value.status_code == 403
        assert "permission" in exc_info.value.detail.lower()
    
    def test_require_role_success(self):
        """Test successful role check."""
        user_info = UserInfo(
            user_id="user_001",
            username="admin",
            email="admin@test.com",
            role=UserRole.SYSTEM_ADMIN,
            permissions=ROLE_PERMISSIONS[UserRole.SYSTEM_ADMIN],
            session_id="test_session",
            login_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            mfa_enabled=False
        )
        
        role_checker = require_role(UserRole.SYSTEM_ADMIN)
        result = role_checker(user_info)
        assert result == user_info
    
    def test_require_role_failure(self):
        """Test failed role check."""
        user_info = UserInfo(
            user_id="user_001",
            username="viewer",
            email="viewer@test.com",
            role=UserRole.VIEWER,
            permissions=ROLE_PERMISSIONS[UserRole.VIEWER],
            session_id="test_session",
            login_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            mfa_enabled=False
        )
        
        role_checker = require_role(UserRole.SYSTEM_ADMIN)
        
        with pytest.raises(HTTPException) as exc_info:
            role_checker(user_info)
        assert exc_info.value.status_code == 403
        assert "role" in exc_info.value.detail.lower()

class TestAuditLogging:
    """Test audit logging functionality."""
    
    @patch('src.api.authentication.redis_client')
    def test_audit_log_success(self, mock_redis_client, mock_request):
        """Test successful audit logging."""
        mock_redis_client.setex.return_value = True
        
        result = asyncio.run(audit_log(
            user_id="user_001",
            action="login",
            resource="authentication",
            details={"method": "password"},
            request=mock_request
        ))
        
        assert result is True
        mock_redis_client.setex.assert_called_once()
    
    @patch('src.api.authentication.redis_client')
    def test_audit_log_redis_error(self, mock_redis_client, mock_request):
        """Test audit logging with Redis error."""
        mock_redis_client.setex.side_effect = Exception("Redis error")
        
        result = asyncio.run(audit_log(
            user_id="user_001",
            action="login",
            resource="authentication",
            request=mock_request
        ))
        
        assert result is False

class TestFailedLoginTracking:
    """Test failed login tracking functionality."""
    
    @patch('src.api.authentication.redis_client')
    def test_check_failed_login_attempts(self, mock_redis_client):
        """Test checking failed login attempts."""
        mock_redis_client.get.return_value = b'3'
        
        result = asyncio.run(check_failed_login_attempts("test_user"))
        assert result == 3
    
    @patch('src.api.authentication.redis_client')
    def test_check_failed_login_attempts_no_record(self, mock_redis_client):
        """Test checking failed login attempts with no record."""
        mock_redis_client.get.return_value = None
        
        result = asyncio.run(check_failed_login_attempts("test_user"))
        assert result == 0
    
    @patch('src.api.authentication.redis_client')
    def test_record_failed_login(self, mock_redis_client):
        """Test recording failed login attempt."""
        mock_redis_client.incr.return_value = 1
        mock_redis_client.expire.return_value = True
        
        result = asyncio.run(record_failed_login("test_user"))
        assert result == 1
        mock_redis_client.incr.assert_called_once_with("failed_login:test_user")
        mock_redis_client.expire.assert_called_once_with("failed_login:test_user", 900)
    
    @patch('src.api.authentication.redis_client')
    def test_clear_failed_logins(self, mock_redis_client):
        """Test clearing failed login attempts."""
        mock_redis_client.delete.return_value = True
        
        result = asyncio.run(clear_failed_logins("test_user"))
        assert result is True
        mock_redis_client.delete.assert_called_once_with("failed_login:test_user")

class TestSecurityEdgeCases:
    """Test security edge cases and attack scenarios."""
    
    def test_jwt_secret_key_exposure(self):
        """Test that JWT secret key is properly protected."""
        # Ensure secret key is not easily guessable
        assert len(JWT_SECRET_KEY) >= 32
        assert JWT_SECRET_KEY != "secret"
        assert JWT_SECRET_KEY != "your-secret-key"
    
    def test_password_timing_attack_resistance(self):
        """Test resistance to password timing attacks."""
        import time
        
        # Test with valid user, wrong password
        start_time = time.time()
        asyncio.run(authenticate_user("admin", "wrong_password"))
        valid_user_time = time.time() - start_time
        
        # Test with invalid user
        start_time = time.time()
        asyncio.run(authenticate_user("nonexistent", "any_password"))
        invalid_user_time = time.time() - start_time
        
        # Times should be reasonably similar (within 50ms)
        time_diff = abs(valid_user_time - invalid_user_time)
        assert time_diff < 0.05, f"Timing difference too large: {time_diff}s"
    
    def test_session_fixation_protection(self):
        """Test protection against session fixation attacks."""
        # Each login should generate a new session ID
        user_data = USERS_DB["admin"]
        
        token1 = create_access_token(user_data, "session1")
        token2 = create_access_token(user_data, "session2")
        
        payload1 = jwt.decode(token1, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        payload2 = jwt.decode(token2, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        assert payload1["session_id"] != payload2["session_id"]
        assert payload1["jti"] != payload2["jti"]
    
    def test_jwt_algorithm_confusion(self):
        """Test protection against JWT algorithm confusion attacks."""
        # Create a token with 'none' algorithm
        payload = {
            "user_id": "user_001",
            "username": "admin",
            "role": "system_admin"
        }
        
        malicious_token = jwt.encode(payload, "", algorithm="none")
        
        with pytest.raises(HTTPException):
            asyncio.run(verify_token_payload(malicious_token))
    
    def test_token_reuse_protection(self):
        """Test that tokens have unique JTI to prevent reuse."""
        user_data = USERS_DB["admin"]
        session_id = "test_session"
        
        token1 = create_access_token(user_data, session_id)
        token2 = create_access_token(user_data, session_id)
        
        payload1 = jwt.decode(token1, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        payload2 = jwt.decode(token2, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        assert payload1["jti"] != payload2["jti"]
    
    def test_privilege_escalation_protection(self):
        """Test protection against privilege escalation."""
        # Create token with viewer role
        user_data = USERS_DB["operator"]
        session_id = "test_session"
        
        token = create_access_token(user_data, session_id)
        
        # Try to modify token to escalate privileges
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        payload["role"] = "system_admin"
        
        malicious_token = jwt.encode(payload, "wrong_secret", algorithm=JWT_ALGORITHM)
        
        with pytest.raises(HTTPException):
            asyncio.run(verify_token_payload(malicious_token))

class TestPenetrationTesting:
    """Test penetration testing scenarios for authentication bypass."""
    
    def test_brute_force_protection(self):
        """Test protection against brute force attacks."""
        # This would be implemented with rate limiting in production
        # For now, we test the failed login tracking
        username = "admin"
        
        # Simulate multiple failed attempts
        for i in range(5):
            with patch('src.api.authentication.redis_client') as mock_redis:
                mock_redis.incr.return_value = i + 1
                mock_redis.expire.return_value = True
                
                count = asyncio.run(record_failed_login(username))
                assert count == i + 1
    
    def test_sql_injection_in_username(self):
        """Test SQL injection protection in username field."""
        # Since we're using a dict-based user store, SQL injection 
        # isn't directly applicable, but we test malicious input handling
        malicious_usernames = [
            "admin'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "admin' UNION SELECT * FROM users --",
            "admin\"; DROP TABLE users; --"
        ]
        
        for username in malicious_usernames:
            result = asyncio.run(authenticate_user(username, "any_password"))
            assert result is None, f"Malicious username should not authenticate: {username}"
    
    def test_password_injection_attempts(self):
        """Test injection attempts in password field."""
        malicious_passwords = [
            "' OR '1'='1",
            "admin'; DROP TABLE users; --",
            "password' UNION SELECT * FROM users --",
            "' OR 1=1 --"
        ]
        
        for password in malicious_passwords:
            result = asyncio.run(authenticate_user("admin", password))
            assert result is None, f"Malicious password should not authenticate: {password}"
    
    def test_jwt_manipulation_attempts(self):
        """Test JWT token manipulation attempts."""
        user_data = USERS_DB["admin"]
        session_id = "test_session"
        valid_token = create_access_token(user_data, session_id)
        
        # Test various manipulation attempts
        manipulation_attempts = [
            valid_token + "extra_data",
            valid_token[:-10] + "manipulated",
            valid_token.replace("admin", "hacker"),
            "Bearer " + valid_token,  # With scheme prefix
            valid_token.replace(".", ""),  # Remove dots
        ]
        
        for token in manipulation_attempts:
            with pytest.raises(HTTPException):
                asyncio.run(verify_token_payload(token))
    
    def test_session_hijacking_protection(self):
        """Test protection against session hijacking."""
        # Sessions should be invalidated on logout
        with patch('src.api.authentication.redis_client') as mock_redis:
            mock_redis.delete.return_value = True
            
            result = asyncio.run(logout("test_session"))
            assert result is True
            mock_redis.delete.assert_called_once_with("session:test_session")
    
    def test_mfa_bypass_attempts(self):
        """Test MFA bypass attempts."""
        # Test various MFA bypass attempts
        bypass_attempts = [
            "",  # Empty token
            "000000",  # Common default
            "111111",  # Sequential numbers
            "123456",  # Common pattern (should work as it's our mock)
            "999999",  # Invalid token
            None,  # None token
        ]
        
        for mfa_token in bypass_attempts:
            if mfa_token == "123456":
                continue  # This is our valid mock token
                
            request = LoginRequest(
                username="risk_manager",
                password="risk123!",
                mfa_token=mfa_token
            )
            
            with pytest.raises(HTTPException):
                asyncio.run(login(request))
    
    def test_role_confusion_attacks(self):
        """Test role confusion attacks."""
        # Create tokens for different roles and ensure they can't be confused
        roles = [UserRole.VIEWER, UserRole.RISK_OPERATOR, UserRole.SYSTEM_ADMIN]
        
        for role in roles:
            # Find a user with this role
            user_data = None
            for user in USERS_DB.values():
                if user["role"] == role:
                    user_data = user
                    break
            
            if user_data:
                token = create_access_token(user_data, "test_session")
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                
                # Ensure role is correctly encoded
                assert payload["role"] == role.value
                
                # Ensure token can't be used for higher privileges
                with patch('src.api.authentication.redis_client') as mock_redis:
                    mock_redis.get.return_value = b'{"valid": "session"}'
                    
                    credentials = HTTPAuthorizationCredentials(
                        scheme="Bearer",
                        credentials=token
                    )
                    
                    user_info = asyncio.run(verify_token(credentials))
                    assert user_info.role == role
                    assert len(user_info.permissions) == len(ROLE_PERMISSIONS[role])

class TestRedisIntegration:
    """Test Redis integration for session management."""
    
    @patch('src.api.authentication.redis_client')
    def test_redis_connection_failure(self, mock_redis_client):
        """Test graceful handling of Redis connection failures."""
        mock_redis_client.get.side_effect = Exception("Redis connection failed")
        
        # Should continue without Redis check
        user_data = USERS_DB["admin"]
        token = create_access_token(user_data, "test_session")
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        # Should not raise exception due to Redis failure
        user_info = asyncio.run(verify_token(credentials))
        assert user_info.username == "admin"
    
    @patch('src.api.authentication.redis_client')
    def test_session_cleanup_on_logout(self, mock_redis_client):
        """Test session cleanup on logout."""
        mock_redis_client.delete.return_value = True
        
        result = asyncio.run(logout("test_session"))
        assert result is True
        mock_redis_client.delete.assert_called_once_with("session:test_session")
    
    @patch('src.api.authentication.redis_client')
    def test_session_data_storage(self, mock_redis_client):
        """Test session data storage format."""
        mock_redis_client.setex.return_value = True
        
        login_request = LoginRequest(username="admin", password="admin123!")
        response = asyncio.run(login(login_request))
        
        # Verify session was stored with correct format
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        
        # Check session key format
        session_key = call_args[0][0]
        assert session_key.startswith("session:")
        
        # Check expiration time
        expiration = call_args[0][1]
        assert expiration == JWT_EXPIRATION_HOURS * 3600

class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
    def test_token_generation_performance(self):
        """Test token generation performance."""
        user_data = USERS_DB["admin"]
        session_id = "test_session"
        
        import time
        start_time = time.time()
        
        # Generate 1000 tokens
        for _ in range(1000):
            create_access_token(user_data, session_id)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should generate 1000 tokens in less than 1 second
        assert duration < 1.0, f"Token generation too slow: {duration}s for 1000 tokens"
    
    def test_password_verification_performance(self):
        """Test password verification performance."""
        password = "test_password123!"
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        import time
        start_time = time.time()
        
        # Verify 100 passwords
        for _ in range(100):
            asyncio.run(verify_password(password, hashed))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should verify 100 passwords in less than 2 seconds
        assert duration < 2.0, f"Password verification too slow: {duration}s for 100 verifications"
    
    def test_concurrent_authentication(self):
        """Test concurrent authentication requests."""
        import asyncio
        import aiohttp
        
        async def authenticate_user_concurrent(username, password):
            return await authenticate_user(username, password)
        
        async def run_concurrent_auth():
            tasks = []
            for _ in range(100):
                task = authenticate_user_concurrent("admin", "admin123!")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        start_time = asyncio.get_event_loop().time()
        results = asyncio.run(run_concurrent_auth())
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # All should succeed
        assert all(result is not None for result in results)
        
        # Should complete in reasonable time
        assert duration < 5.0, f"Concurrent authentication too slow: {duration}s"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])