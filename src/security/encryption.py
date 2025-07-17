"""
Enterprise Encryption & Data Protection
Comprehensive encryption for data at rest and in transit
"""

import os
import base64
import json
import secrets
import hashlib
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import ssl
from dataclasses import dataclass
from enum import Enum

from src.monitoring.logger_config import get_logger
from src.security.secrets_manager import get_secret

logger = get_logger(__name__)

class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms"""
    AES_GCM = "aes_gcm"
    FERNET = "fernet"
    RSA_OAEP = "rsa_oaep"
    CHACHA20_POLY1305 = "chacha20_poly1305"

class KeyDerivationFunction(str, Enum):
    """Key derivation functions"""
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    HKDF = "hkdf"

@dataclass
class EncryptionKey:
    """Encryption key metadata"""
    key_id: str
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def is_expired(self) -> bool:
        """Check if key is expired"""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return True
        if self.max_usage and self.usage_count >= self.max_usage:
            return True
        return False

@dataclass
class EncryptedData:
    """Encrypted data container"""
    ciphertext: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "nonce": base64.b64encode(self.nonce).decode() if self.nonce else None,
            "tag": base64.b64encode(self.tag).decode() if self.tag else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedData':
        """Create from dictionary"""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            key_id=data["key_id"],
            nonce=base64.b64decode(data["nonce"]) if data["nonce"] else None,
            tag=base64.b64decode(data["tag"]) if data["tag"] else None,
            metadata=data.get("metadata")
        )

class EncryptionManager:
    """Enterprise encryption manager"""
    
    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key = self._get_master_key()
        self.key_rotation_interval = timedelta(days=90)
        
        # Initialize default encryption keys
        self._initialize_default_keys()
        
        logger.info("Encryption manager initialized")
    
    def _get_master_key(self) -> bytes:
        """Get master encryption key"""
        master_key = get_secret("master_encryption_key")
        if not master_key:
            # Generate secure master key
            master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            logger.warning("Master encryption key generated - store securely in production")
        
        if isinstance(master_key, str):
            master_key = master_key.encode()
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'grandmodel-encryption-salt',
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(master_key)
    
    def _initialize_default_keys(self):
        """Initialize default encryption keys"""
        # Data encryption key
        data_key = self._generate_encryption_key(
            "data_encryption",
            EncryptionAlgorithm.AES_GCM
        )
        
        # Database encryption key
        db_key = self._generate_encryption_key(
            "database_encryption",
            EncryptionAlgorithm.FERNET
        )
        
        # API key encryption key
        api_key = self._generate_encryption_key(
            "api_key_encryption",
            EncryptionAlgorithm.FERNET
        )
        
        # Session encryption key
        session_key = self._generate_encryption_key(
            "session_encryption",
            EncryptionAlgorithm.AES_GCM
        )
        
        # Store keys
        self.keys["data_encryption"] = data_key
        self.keys["database_encryption"] = db_key
        self.keys["api_key_encryption"] = api_key
        self.keys["session_encryption"] = session_key
    
    def _generate_encryption_key(self, 
                               key_id: str,
                               algorithm: EncryptionAlgorithm,
                               expires_in_days: Optional[int] = None) -> EncryptionKey:
        """Generate new encryption key"""
        # Generate key based on algorithm
        if algorithm == EncryptionAlgorithm.AES_GCM:
            key_data = secrets.token_bytes(32)  # 256-bit key
        elif algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm == EncryptionAlgorithm.RSA_OAEP:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)  # 256-bit key
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        return EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            key_data=key_data,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            metadata={"auto_generated": True}
        )
    
    def encrypt_data(self, 
                    data: Union[str, bytes, Dict[str, Any]],
                    key_id: str = "data_encryption",
                    additional_data: Optional[bytes] = None) -> EncryptedData:
        """Encrypt data using specified key"""
        try:
            # Get encryption key
            if key_id not in self.keys:
                raise ValueError(f"Encryption key not found: {key_id}")
            
            key = self.keys[key_id]
            
            # Check if key is expired
            if key.is_expired():
                raise ValueError(f"Encryption key expired: {key_id}")
            
            # Convert data to bytes
            if isinstance(data, str):
                plaintext = data.encode('utf-8')
            elif isinstance(data, dict):
                plaintext = json.dumps(data).encode('utf-8')
            else:
                plaintext = data
            
            # Encrypt based on algorithm
            if key.algorithm == EncryptionAlgorithm.AES_GCM:
                return self._encrypt_aes_gcm(plaintext, key, additional_data)
            elif key.algorithm == EncryptionAlgorithm.FERNET:
                return self._encrypt_fernet(plaintext, key)
            elif key.algorithm == EncryptionAlgorithm.RSA_OAEP:
                return self._encrypt_rsa_oaep(plaintext, key)
            elif key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return self._encrypt_chacha20_poly1305(plaintext, key, additional_data)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {key.algorithm}")
                
        except Exception as e:
            logger.error("Encryption failed", key_id=key_id, error=str(e))
            raise
    
    def _encrypt_aes_gcm(self, 
                        plaintext: bytes,
                        key: EncryptionKey,
                        additional_data: Optional[bytes] = None) -> EncryptedData:
        """Encrypt using AES-GCM"""
        aesgcm = AESGCM(key.key_data)
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        ciphertext = aesgcm.encrypt(nonce, plaintext, additional_data)
        
        # AES-GCM returns ciphertext + tag combined
        tag = ciphertext[-16:]  # Last 16 bytes are the tag
        ciphertext = ciphertext[:-16]  # Remove tag from ciphertext
        
        key.usage_count += 1
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=key.algorithm,
            key_id=key.key_id,
            nonce=nonce,
            tag=tag
        )
    
    def _encrypt_fernet(self, plaintext: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt using Fernet"""
        fernet = Fernet(key.key_data)
        ciphertext = fernet.encrypt(plaintext)
        
        key.usage_count += 1
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=key.algorithm,
            key_id=key.key_id
        )
    
    def _encrypt_rsa_oaep(self, plaintext: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt using RSA-OAEP"""
        # Load private key to get public key
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        # RSA can only encrypt small amounts of data
        max_length = (public_key.key_size // 8) - 2 * (hashes.SHA256().digest_size) - 2
        
        if len(plaintext) > max_length:
            raise ValueError(f"Data too large for RSA encryption. Max: {max_length} bytes")
        
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        key.usage_count += 1
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=key.algorithm,
            key_id=key.key_id
        )
    
    def _encrypt_chacha20_poly1305(self, 
                                  plaintext: bytes,
                                  key: EncryptionKey,
                                  additional_data: Optional[bytes] = None) -> EncryptedData:
        """Encrypt using ChaCha20-Poly1305"""
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        
        cipher = Cipher(
            algorithms.ChaCha20(key.key_data, nonce),
            mode=None,
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # For ChaCha20-Poly1305, we need to use AEAD interface
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        
        chacha = ChaCha20Poly1305(key.key_data)
        ciphertext = chacha.encrypt(nonce, plaintext, additional_data)
        
        key.usage_count += 1
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=key.algorithm,
            key_id=key.key_id,
            nonce=nonce
        )
    
    def decrypt_data(self, 
                    encrypted_data: EncryptedData,
                    additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt data"""
        try:
            # Get decryption key
            if encrypted_data.key_id not in self.keys:
                raise ValueError(f"Decryption key not found: {encrypted_data.key_id}")
            
            key = self.keys[encrypted_data.key_id]
            
            # Decrypt based on algorithm
            if encrypted_data.algorithm == EncryptionAlgorithm.AES_GCM:
                return self._decrypt_aes_gcm(encrypted_data, key, additional_data)
            elif encrypted_data.algorithm == EncryptionAlgorithm.FERNET:
                return self._decrypt_fernet(encrypted_data, key)
            elif encrypted_data.algorithm == EncryptionAlgorithm.RSA_OAEP:
                return self._decrypt_rsa_oaep(encrypted_data, key)
            elif encrypted_data.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return self._decrypt_chacha20_poly1305(encrypted_data, key, additional_data)
            else:
                raise ValueError(f"Unsupported decryption algorithm: {encrypted_data.algorithm}")
                
        except Exception as e:
            logger.error("Decryption failed", key_id=encrypted_data.key_id, error=str(e))
            raise
    
    def _decrypt_aes_gcm(self, 
                        encrypted_data: EncryptedData,
                        key: EncryptionKey,
                        additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt using AES-GCM"""
        aesgcm = AESGCM(key.key_data)
        
        # Combine ciphertext and tag
        ciphertext_with_tag = encrypted_data.ciphertext + encrypted_data.tag
        
        plaintext = aesgcm.decrypt(encrypted_data.nonce, ciphertext_with_tag, additional_data)
        
        return plaintext
    
    def _decrypt_fernet(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using Fernet"""
        fernet = Fernet(key.key_data)
        plaintext = fernet.decrypt(encrypted_data.ciphertext)
        
        return plaintext
    
    def _decrypt_rsa_oaep(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using RSA-OAEP"""
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=default_backend()
        )
        
        plaintext = private_key.decrypt(
            encrypted_data.ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext
    
    def _decrypt_chacha20_poly1305(self, 
                                  encrypted_data: EncryptedData,
                                  key: EncryptionKey,
                                  additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        
        chacha = ChaCha20Poly1305(key.key_data)
        plaintext = chacha.decrypt(encrypted_data.nonce, encrypted_data.ciphertext, additional_data)
        
        return plaintext
    
    def rotate_key(self, key_id: str) -> str:
        """Rotate encryption key"""
        try:
            if key_id not in self.keys:
                raise ValueError(f"Key not found: {key_id}")
            
            old_key = self.keys[key_id]
            
            # Generate new key with same algorithm
            new_key_id = f"{key_id}_rotated_{int(datetime.utcnow().timestamp())}"
            new_key = self._generate_encryption_key(new_key_id, old_key.algorithm)
            
            # Store new key
            self.keys[new_key_id] = new_key
            
            # Mark old key as expired
            old_key.expires_at = datetime.utcnow()
            
            logger.info("Key rotated", old_key_id=key_id, new_key_id=new_key_id)
            
            return new_key_id
            
        except Exception as e:
            logger.error("Key rotation failed", key_id=key_id, error=str(e))
            raise
    
    def get_key_info(self, key_id: str) -> Dict[str, Any]:
        """Get key information"""
        if key_id not in self.keys:
            return {}
        
        key = self.keys[key_id]
        
        return {
            "key_id": key.key_id,
            "algorithm": key.algorithm.value,
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "usage_count": key.usage_count,
            "max_usage": key.max_usage,
            "is_expired": key.is_expired(),
            "metadata": key.metadata
        }
    
    def encrypt_sensitive_fields(self, 
                               data: Dict[str, Any],
                               sensitive_fields: List[str],
                               key_id: str = "data_encryption") -> Dict[str, Any]:
        """Encrypt sensitive fields in a dictionary"""
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                # Encrypt the field
                encrypted_field = self.encrypt_data(encrypted_data[field], key_id)
                encrypted_data[field] = encrypted_field.to_dict()
                encrypted_data[f"{field}_encrypted"] = True
        
        return encrypted_data
    
    def decrypt_sensitive_fields(self, 
                               data: Dict[str, Any],
                               sensitive_fields: List[str]) -> Dict[str, Any]:
        """Decrypt sensitive fields in a dictionary"""
        decrypted_data = data.copy()
        
        for field in sensitive_fields:
            if f"{field}_encrypted" in decrypted_data and decrypted_data[f"{field}_encrypted"]:
                # Decrypt the field
                encrypted_field = EncryptedData.from_dict(decrypted_data[field])
                decrypted_value = self.decrypt_data(encrypted_field)
                
                # Try to decode as JSON first, then as string
                try:
                    decrypted_data[field] = json.loads(decrypted_value.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    decrypted_data[field] = decrypted_value.decode('utf-8')
                
                # Remove encryption metadata
                del decrypted_data[f"{field}_encrypted"]
        
        return decrypted_data

class TLSManager:
    """TLS/SSL certificate and configuration management"""
    
    def __init__(self):
        self.cert_path = "/etc/ssl/certs/grandmodel"
        self.key_path = "/etc/ssl/private/grandmodel"
        
        logger.info("TLS Manager initialized")
    
    def create_ssl_context(self, 
                          cert_file: Optional[str] = None,
                          key_file: Optional[str] = None) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        try:
            # Create SSL context
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            
            # Load certificates
            cert_file = cert_file or os.path.join(self.cert_path, "server.crt")
            key_file = key_file or os.path.join(self.key_path, "server.key")
            
            if os.path.exists(cert_file) and os.path.exists(key_file):
                context.load_cert_chain(cert_file, key_file)
                logger.info("SSL certificates loaded")
            else:
                logger.warning("SSL certificates not found - using default context")
            
            # Security settings
            context.check_hostname = False  # Disable for internal services
            context.verify_mode = ssl.CERT_NONE  # Adjust based on requirements
            
            # Set secure protocols
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_3
            
            # Set secure ciphers
            context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS")
            
            return context
            
        except Exception as e:
            logger.error("Failed to create SSL context", error=str(e))
            raise
    
    def get_tls_config(self) -> Dict[str, Any]:
        """Get TLS configuration for applications"""
        return {
            "ssl_cert_file": os.path.join(self.cert_path, "server.crt"),
            "ssl_key_file": os.path.join(self.key_path, "server.key"),
            "ssl_ca_file": os.path.join(self.cert_path, "ca.crt"),
            "ssl_protocols": ["TLSv1.2", "TLSv1.3"],
            "ssl_ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS",
            "ssl_verify_mode": "none",  # Adjust based on requirements
            "ssl_check_hostname": False
        }

# Global instances
encryption_manager = EncryptionManager()
tls_manager = TLSManager()

# Utility functions
def encrypt_data(data: Union[str, bytes, Dict[str, Any]], key_id: str = "data_encryption") -> EncryptedData:
    """Convenience function to encrypt data"""
    return encryption_manager.encrypt_data(data, key_id)

def decrypt_data(encrypted_data: EncryptedData) -> bytes:
    """Convenience function to decrypt data"""
    return encryption_manager.decrypt_data(encrypted_data)

def encrypt_sensitive_fields(data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
    """Convenience function to encrypt sensitive fields"""
    return encryption_manager.encrypt_sensitive_fields(data, sensitive_fields)

def decrypt_sensitive_fields(data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
    """Convenience function to decrypt sensitive fields"""
    return encryption_manager.decrypt_sensitive_fields(data, sensitive_fields)

def get_ssl_context() -> ssl.SSLContext:
    """Convenience function to get SSL context"""
    return tls_manager.create_ssl_context()

# Sensitive field definitions for common data types
SENSITIVE_USER_FIELDS = ["email", "password_hash", "api_key", "personal_info"]
SENSITIVE_TRADE_FIELDS = ["account_id", "position_size", "pnl", "trade_details"]
SENSITIVE_RISK_FIELDS = ["portfolio_value", "risk_metrics", "var_calculation"]
SENSITIVE_COMPLIANCE_FIELDS = ["audit_trail", "regulatory_data", "client_info"]
