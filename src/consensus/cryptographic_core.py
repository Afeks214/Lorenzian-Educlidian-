"""
Cryptographic Core for Byzantine Fault Tolerant Consensus

Implements HMAC-SHA256 based cryptographic validation system to secure
PBFT consensus messages and prevent tampering/forgery attacks.

Features:
- HMAC-SHA256 message authentication
- Secure key generation and management
- Message integrity validation
- Replay attack prevention with nonces
- Key rotation support

Author: Agent 2 - Consensus Security Engineer
Version: 1.0 - Production Ready
"""

import hmac
import hashlib
import secrets
import time
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger(__name__)


@dataclass
class CryptoKey:
    """Cryptographic key container"""
    key_id: str
    key_data: bytes
    key_type: str  # 'hmac', 'rsa_private', 'rsa_public'
    created_at: float
    expires_at: Optional[float] = None
    is_active: bool = True


@dataclass
class MessageSignature:
    """Message signature container"""
    signature: str
    signer_id: str
    timestamp: float
    nonce: str
    algorithm: str = "HMAC-SHA256"


class CryptographicCore:
    """
    Cryptographic Core for Byzantine Fault Tolerant Consensus
    
    Provides secure message authentication and validation using HMAC-SHA256
    with additional RSA signatures for non-repudiation. Includes key management,
    nonce generation, and replay attack prevention.
    """
    
    def __init__(self, master_secret: Optional[bytes] = None):
        """
        Initialize cryptographic core
        
        Args:
            master_secret: Optional master secret for key derivation
        """
        # Master secret for key derivation
        self.master_secret = master_secret or secrets.token_bytes(32)
        
        # Agent keys storage
        self.agent_keys: Dict[str, CryptoKey] = {}
        self.hmac_keys: Dict[str, bytes] = {}
        self.rsa_keys: Dict[str, Tuple[Any, Any]] = {}  # (private_key, public_key)
        
        # Nonce tracking for replay attack prevention
        self.used_nonces: Dict[str, float] = {}
        self.nonce_expiry_time = 300.0  # 5 minutes
        
        # Key rotation settings
        self.key_rotation_interval = 3600.0  # 1 hour
        self.last_key_rotation = time.time()
        
        # Security metrics
        self.security_metrics = {
            'signatures_created': 0,
            'signatures_validated': 0,
            'signature_failures': 0,
            'replay_attacks_blocked': 0,
            'key_rotations': 0,
            'nonce_collisions': 0
        }
        
        logger.info("Cryptographic core initialized with HMAC-SHA256 + RSA")
    
    def initialize_agent_keys(self, agent_ids: List[str]) -> Dict[str, str]:
        """
        Initialize cryptographic keys for all agents
        
        Args:
            agent_ids: List of agent identifiers
            
        Returns:
            Dictionary mapping agent IDs to their public key fingerprints
        """
        public_key_fingerprints = {}
        
        for agent_id in agent_ids:
            # Generate HMAC key
            hmac_key = self._derive_hmac_key(agent_id)
            self.hmac_keys[agent_id] = hmac_key
            
            # Generate RSA key pair
            private_key, public_key = self._generate_rsa_key_pair()
            self.rsa_keys[agent_id] = (private_key, public_key)
            
            # Create key objects
            hmac_key_obj = CryptoKey(
                key_id=f"{agent_id}_hmac",
                key_data=hmac_key,
                key_type="hmac",
                created_at=time.time(),
                expires_at=time.time() + self.key_rotation_interval
            )
            
            self.agent_keys[f"{agent_id}_hmac"] = hmac_key_obj
            
            # Get public key fingerprint
            public_key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            fingerprint = hashlib.sha256(public_key_bytes).hexdigest()[:16]
            public_key_fingerprints[agent_id] = fingerprint
            
            logger.debug(f"Generated keys for agent {agent_id}, fingerprint: {fingerprint}")
        
        return public_key_fingerprints
    
    def _derive_hmac_key(self, agent_id: str) -> bytes:
        """Derive HMAC key for specific agent using PBKDF2"""
        salt = f"consensus_agent_{agent_id}".encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
            backend=default_backend()
        )
        
        return kdf.derive(self.master_secret)
    
    def _generate_rsa_key_pair(self) -> Tuple[Any, Any]:
        """Generate RSA key pair for digital signatures"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,  # Secure for next 10+ years
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        return private_key, public_key
    
    def sign_message(self, message_hash: str, signer_id: str) -> str:
        """
        Create cryptographic signature for message
        
        Args:
            message_hash: SHA256 hash of message content
            signer_id: ID of the signing agent
            
        Returns:
            Base64-encoded signature
        """
        try:
            # Generate nonce for replay attack prevention
            nonce = secrets.token_hex(16)
            timestamp = time.time()
            
            # Create signature payload
            signature_payload = {
                'message_hash': message_hash,
                'signer_id': signer_id,
                'timestamp': timestamp,
                'nonce': nonce
            }
            
            payload_json = json.dumps(signature_payload, sort_keys=True)
            payload_bytes = payload_json.encode('utf-8')
            
            # Create HMAC signature
            hmac_key = self.hmac_keys.get(signer_id)
            if not hmac_key:
                raise ValueError(f"No HMAC key found for agent {signer_id}")
            
            hmac_signature = hmac.new(
                hmac_key,
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            
            # Create RSA signature for non-repudiation
            rsa_private_key, _ = self.rsa_keys.get(signer_id, (None, None))
            if rsa_private_key:
                rsa_signature = rsa_private_key.sign(
                    payload_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                rsa_signature_b64 = base64.b64encode(rsa_signature).decode('utf-8')
            else:
                rsa_signature_b64 = None
            
            # Combine signatures
            combined_signature = {
                'hmac': hmac_signature,
                'rsa': rsa_signature_b64,
                'timestamp': timestamp,
                'nonce': nonce,
                'signer_id': signer_id
            }
            
            signature_json = json.dumps(combined_signature, sort_keys=True)
            signature_b64 = base64.b64encode(signature_json.encode()).decode('utf-8')
            
            # Store nonce
            self.used_nonces[nonce] = timestamp
            
            # Update metrics
            self.security_metrics['signatures_created'] += 1
            
            logger.debug(f"Created signature for {signer_id}, nonce: {nonce[:8]}...")
            return signature_b64
            
        except Exception as e:
            logger.error(f"Failed to sign message for {signer_id}: {e}")
            raise
    
    def validate_signature(self, message_hash: str, signature: str, expected_signer: str) -> bool:
        """
        Validate cryptographic signature
        
        Args:
            message_hash: SHA256 hash of message content
            signature: Base64-encoded signature to validate
            expected_signer: Expected signer agent ID
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            self.security_metrics['signatures_validated'] += 1
            
            # Decode signature
            signature_json = base64.b64decode(signature).decode('utf-8')
            signature_data = json.loads(signature_json)
            
            # Extract signature components
            hmac_signature = signature_data.get('hmac')
            rsa_signature_b64 = signature_data.get('rsa')
            timestamp = signature_data.get('timestamp')
            nonce = signature_data.get('nonce')
            signer_id = signature_data.get('signer_id')
            
            # Validate signer ID
            if signer_id != expected_signer:
                logger.warning(f"Signer ID mismatch: expected {expected_signer}, got {signer_id}")
                self.security_metrics['signature_failures'] += 1
                return False
            
            # Check for replay attack
            if self._is_replay_attack(nonce, timestamp):
                logger.warning(f"Replay attack detected: nonce {nonce[:8]}...")
                self.security_metrics['replay_attacks_blocked'] += 1
                return False
            
            # Recreate signature payload
            signature_payload = {
                'message_hash': message_hash,
                'signer_id': signer_id,
                'timestamp': timestamp,
                'nonce': nonce
            }
            
            payload_json = json.dumps(signature_payload, sort_keys=True)
            payload_bytes = payload_json.encode('utf-8')
            
            # Validate HMAC signature
            hmac_key = self.hmac_keys.get(signer_id)
            if not hmac_key:
                logger.error(f"No HMAC key found for agent {signer_id}")
                self.security_metrics['signature_failures'] += 1
                return False
            
            expected_hmac = hmac.new(
                hmac_key,
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(hmac_signature, expected_hmac):
                logger.warning(f"HMAC validation failed for {signer_id}")
                self.security_metrics['signature_failures'] += 1
                return False
            
            # Validate RSA signature if present
            if rsa_signature_b64:
                _, rsa_public_key = self.rsa_keys.get(signer_id, (None, None))
                if rsa_public_key:
                    try:
                        rsa_signature = base64.b64decode(rsa_signature_b64)
                        rsa_public_key.verify(
                            rsa_signature,
                            payload_bytes,
                            padding.PSS(
                                mgf=padding.MGF1(hashes.SHA256()),
                                salt_length=padding.PSS.MAX_LENGTH
                            ),
                            hashes.SHA256()
                        )
                    except Exception as e:
                        logger.warning(f"RSA signature validation failed for {signer_id}: {e}")
                        self.security_metrics['signature_failures'] += 1
                        return False
            
            # Mark nonce as used
            self.used_nonces[nonce] = timestamp
            
            logger.debug(f"Signature validated for {signer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Signature validation error for {expected_signer}: {e}")
            self.security_metrics['signature_failures'] += 1
            return False
    
    def _is_replay_attack(self, nonce: str, timestamp: float) -> bool:
        """Check if message is a replay attack"""
        current_time = time.time()
        
        # Check if nonce was already used
        if nonce in self.used_nonces:
            return True
        
        # Check if timestamp is too old
        if current_time - timestamp > self.nonce_expiry_time:
            return True
        
        # Check if timestamp is in the future (clock skew tolerance: 30 seconds)
        if timestamp > current_time + 30:
            return True
        
        return False
    
    def cleanup_expired_nonces(self):
        """Clean up expired nonces to prevent memory buildup"""
        current_time = time.time()
        expired_nonces = [
            nonce for nonce, timestamp in self.used_nonces.items()
            if current_time - timestamp > self.nonce_expiry_time
        ]
        
        for nonce in expired_nonces:
            del self.used_nonces[nonce]
        
        if expired_nonces:
            logger.debug(f"Cleaned up {len(expired_nonces)} expired nonces")
    
    def rotate_keys(self, agent_ids: List[str]) -> Dict[str, str]:
        """
        Rotate cryptographic keys for security
        
        Args:
            agent_ids: List of agent identifiers
            
        Returns:
            New public key fingerprints
        """
        logger.info("Starting key rotation for all agents")
        
        # Archive old keys
        for agent_id in agent_ids:
            old_key = self.agent_keys.get(f"{agent_id}_hmac")
            if old_key:
                old_key.is_active = False
                old_key.expires_at = time.time()
        
        # Generate new keys
        new_fingerprints = self.initialize_agent_keys(agent_ids)
        
        # Update rotation timestamp
        self.last_key_rotation = time.time()
        self.security_metrics['key_rotations'] += 1
        
        logger.info(f"Key rotation completed for {len(agent_ids)} agents")
        return new_fingerprints
    
    def should_rotate_keys(self) -> bool:
        """Check if keys should be rotated based on time or security events"""
        current_time = time.time()
        
        # Time-based rotation
        if current_time - self.last_key_rotation > self.key_rotation_interval:
            return True
        
        # Security event-based rotation
        failure_rate = self.security_metrics['signature_failures'] / max(1, self.security_metrics['signatures_validated'])
        if failure_rate > 0.1:  # More than 10% signature failures
            logger.warning(f"High signature failure rate: {failure_rate:.2%}, considering key rotation")
            return True
        
        return False
    
    def get_public_key_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get public key information for agent"""
        if agent_id not in self.rsa_keys:
            return None
        
        _, public_key = self.rsa_keys[agent_id]
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        fingerprint = hashlib.sha256(public_key_bytes).hexdigest()
        
        return {
            'agent_id': agent_id,
            'public_key_pem': public_key_bytes.decode('utf-8'),
            'fingerprint': fingerprint,
            'key_size': public_key.key_size,
            'algorithm': 'RSA'
        }
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get cryptographic security metrics"""
        metrics = self.security_metrics.copy()
        
        # Add derived metrics
        total_validations = metrics['signatures_validated']
        if total_validations > 0:
            metrics['signature_success_rate'] = 1.0 - (metrics['signature_failures'] / total_validations)
            metrics['failure_rate'] = metrics['signature_failures'] / total_validations
        else:
            metrics['signature_success_rate'] = 1.0
            metrics['failure_rate'] = 0.0
        
        metrics['nonce_count'] = len(self.used_nonces)
        metrics['active_keys'] = len([k for k in self.agent_keys.values() if k.is_active])
        metrics['last_key_rotation'] = self.last_key_rotation
        
        return metrics
    
    def export_public_keys(self) -> Dict[str, str]:
        """Export all public keys for sharing with other nodes"""
        public_keys = {}
        
        for agent_id, (_, public_key) in self.rsa_keys.items():
            public_key_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
            
            public_keys[agent_id] = public_key_pem
        
        return public_keys
    
    def import_public_keys(self, public_keys: Dict[str, str]):
        """Import public keys from other nodes"""
        for agent_id, public_key_pem in public_keys.items():
            try:
                public_key = serialization.load_pem_public_key(
                    public_key_pem.encode('utf-8'),
                    backend=default_backend()
                )
                
                # Update our public key store
                if agent_id in self.rsa_keys:
                    private_key, _ = self.rsa_keys[agent_id]
                    self.rsa_keys[agent_id] = (private_key, public_key)
                else:
                    self.rsa_keys[agent_id] = (None, public_key)
                
                logger.debug(f"Imported public key for {agent_id}")
                
            except Exception as e:
                logger.error(f"Failed to import public key for {agent_id}: {e}")


class MessageValidator:
    """High-level message validation interface"""
    
    def __init__(self, crypto_core: CryptographicCore):
        """
        Initialize message validator
        
        Args:
            crypto_core: Cryptographic core instance
        """
        self.crypto_core = crypto_core
        
    def validate_message(self, message) -> bool:
        """
        Validate a PBFT message with cryptographic checks
        
        Args:
            message: PBFTMessage instance to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        try:
            # Check if message has signature
            if not message.signature:
                logger.warning(f"Message from {message.sender_id} has no signature")
                return False
            
            # Get message hash
            message_hash = message.get_hash()
            
            # Validate signature
            is_valid = self.crypto_core.validate_signature(
                message_hash, message.signature, message.sender_id
            )
            
            if not is_valid:
                logger.warning(f"Invalid signature on message from {message.sender_id}")
                return False
            
            # Additional message-specific validation
            return self._validate_message_structure(message)
            
        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return False
    
    def _validate_message_structure(self, message) -> bool:
        """Validate message structure and content"""
        try:
            # Check required fields
            if not message.sender_id or not message.timestamp:
                return False
            
            # Check timestamp is reasonable (not too old or in future)
            current_time = time.time()
            if current_time - message.timestamp > 60:  # 1 minute old
                logger.warning(f"Message timestamp too old: {current_time - message.timestamp:.1f}s")
                return False
            
            if message.timestamp > current_time + 10:  # 10 seconds in future
                logger.warning(f"Message timestamp in future: {message.timestamp - current_time:.1f}s")
                return False
            
            # Check view number is non-negative
            if message.view_number < 0:
                return False
            
            # Check sequence number is valid
            if message.sequence_number < -1:  # -1 is allowed for special messages
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Message structure validation error: {e}")
            return False