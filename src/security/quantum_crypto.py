"""
Post-Quantum Cryptography Implementation
Future-proof cryptographic framework for quantum-resistant security.

This module implements NIST-approved post-quantum cryptographic algorithms:
- CRYSTALS-Kyber for key encapsulation
- CRYSTALS-Dilithium for digital signatures
- SPHINCS+ for hash-based signatures
- Quantum-safe random number generation
- Hybrid classical/quantum cryptography
- Cryptographic agility framework
"""

import os
import json
import secrets
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Protocol
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Third-party quantum-resistant libraries
try:
    import pqcrypto
    from pqcrypto.kem import kyber1024, kyber768, kyber512
    from pqcrypto.sign import dilithium5, dilithium3, dilithium2
    from pqcrypto.sign import sphincsharaka256ssimple, sphincssha256256ssimple
    PQCRYPTO_AVAILABLE = True
except ImportError:
    PQCRYPTO_AVAILABLE = False

# Classical crypto for hybrid mode
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from src.monitoring.logger_config import get_logger
from src.core.errors.base_exceptions import CryptographicError, QuantumCryptoError

logger = get_logger(__name__)

# =============================================================================
# QUANTUM CRYPTOGRAPHY CONFIGURATION
# =============================================================================

class QuantumAlgorithm(str, Enum):
    """Post-quantum cryptographic algorithms"""
    # Key Encapsulation Mechanisms (KEMs)
    KYBER_512 = "kyber512"
    KYBER_768 = "kyber768"
    KYBER_1024 = "kyber1024"
    
    # Digital Signature Schemes
    DILITHIUM_2 = "dilithium2"
    DILITHIUM_3 = "dilithium3"
    DILITHIUM_5 = "dilithium5"
    
    # Hash-based Signatures
    SPHINCS_SHA256_256S = "sphincs_sha256_256s"
    SPHINCS_HARAKA_256S = "sphincs_haraka_256s"

class SecurityLevel(int, Enum):
    """NIST security levels"""
    LEVEL_1 = 1  # Equivalent to AES-128
    LEVEL_2 = 2  # Equivalent to AES-128
    LEVEL_3 = 3  # Equivalent to AES-192
    LEVEL_4 = 4  # Equivalent to AES-256
    LEVEL_5 = 5  # Equivalent to AES-256

class CryptoMode(str, Enum):
    """Cryptographic modes"""
    QUANTUM_ONLY = "quantum_only"
    CLASSICAL_ONLY = "classical_only"
    HYBRID = "hybrid"

@dataclass
class QuantumConfig:
    """Quantum cryptography configuration"""
    default_kem: QuantumAlgorithm = QuantumAlgorithm.KYBER_1024
    default_signature: QuantumAlgorithm = QuantumAlgorithm.DILITHIUM_5
    security_level: SecurityLevel = SecurityLevel.LEVEL_5
    crypto_mode: CryptoMode = CryptoMode.HYBRID
    
    # Hybrid configuration
    enable_classical_backup: bool = True
    classical_key_size: int = 4096  # RSA key size
    classical_curve: str = "secp384r1"  # ECC curve
    
    # Performance settings
    cache_keys: bool = True
    key_cache_ttl: int = 3600  # 1 hour
    enable_hardware_acceleration: bool = True
    
    # Quantum threat assessment
    quantum_threat_timeline: int = 2030  # Expected quantum threat year
    migration_deadline: datetime = field(default_factory=lambda: datetime(2028, 1, 1))

# =============================================================================
# QUANTUM-SAFE RANDOM NUMBER GENERATION
# =============================================================================

class QuantumRandomGenerator:
    """Quantum-safe random number generator"""
    
    def __init__(self):
        self.entropy_sources = []
        self._init_entropy_sources()
    
    def _init_entropy_sources(self):
        """Initialize multiple entropy sources"""
        # System random
        self.entropy_sources.append(('system', secrets.SystemRandom()))
        
        # Hardware random if available
        try:
            if os.path.exists('/dev/hwrng'):
                self.entropy_sources.append(('hardware', self._hw_random))
        except Exception:
            pass
        
        # CPU timestamp counter
        self.entropy_sources.append(('rdtsc', self._rdtsc_random))
        
        # Environmental entropy
        self.entropy_sources.append(('env', self._env_random))
    
    def _hw_random(self, size: int) -> bytes:
        """Get hardware random bytes"""
        try:
            with open('/dev/hwrng', 'rb') as f:
                return f.read(size)
        except Exception:
            return secrets.token_bytes(size)
    
    def _rdtsc_random(self, size: int) -> bytes:
        """Get timestamp counter entropy"""
        import time
        entropy = []
        for _ in range(size):
            entropy.append(int(time.time_ns()) & 0xFF)
        return bytes(entropy)
    
    def _env_random(self, size: int) -> bytes:
        """Get environmental entropy"""
        import os
        import psutil
        
        # Gather system metrics
        entropy_data = []
        entropy_data.append(str(os.getpid()))
        entropy_data.append(str(time.time_ns()))
        
        try:
            entropy_data.append(str(psutil.cpu_percent()))
            entropy_data.append(str(psutil.virtual_memory().percent))
        except Exception:
            pass
        
        # Hash collected entropy
        entropy_str = ''.join(entropy_data)
        return hashlib.blake2b(entropy_str.encode(), digest_size=size).digest()
    
    def generate_bytes(self, size: int) -> bytes:
        """Generate quantum-safe random bytes"""
        if size <= 0:
            raise ValueError("Size must be positive")
        
        # Collect entropy from all sources
        entropy_pool = bytearray()
        
        for source_name, source in self.entropy_sources:
            try:
                if callable(source):
                    entropy_data = source(size)
                else:
                    entropy_data = source.randbytes(size)
                
                entropy_pool.extend(entropy_data)
            except Exception as e:
                logger.warning(f"Entropy source {source_name} failed: {e}")
        
        # XOR all entropy sources
        result = bytearray(size)
        for i in range(size):
            result[i] = entropy_pool[i % len(entropy_pool)]
            for j in range(1, len(entropy_pool) // size):
                result[i] ^= entropy_pool[i + j * size]
        
        # Final conditioning with BLAKE2b
        return hashlib.blake2b(bytes(result), digest_size=size).digest()
    
    def generate_int(self, min_val: int, max_val: int) -> int:
        """Generate random integer in range"""
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        
        range_size = max_val - min_val
        byte_size = (range_size.bit_length() + 7) // 8
        
        while True:
            random_bytes = self.generate_bytes(byte_size)
            random_int = int.from_bytes(random_bytes, 'big')
            
            if random_int < range_size:
                return min_val + random_int

# =============================================================================
# ABSTRACT CRYPTOGRAPHIC INTERFACES
# =============================================================================

class KeyEncapsulationMechanism(Protocol):
    """Key Encapsulation Mechanism interface"""
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate public/private key pair"""
        ...
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate shared secret"""
        ...
    
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate shared secret"""
        ...

class DigitalSignature(Protocol):
    """Digital signature interface"""
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate signing key pair"""
        ...
    
    def sign(self, private_key: bytes, message: bytes) -> bytes:
        """Sign message"""
        ...
    
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify signature"""
        ...

# =============================================================================
# QUANTUM-RESISTANT ALGORITHMS
# =============================================================================

class KyberKEM:
    """CRYSTALS-Kyber Key Encapsulation Mechanism"""
    
    def __init__(self, variant: QuantumAlgorithm = QuantumAlgorithm.KYBER_1024):
        self.variant = variant
        self.algorithm = self._get_algorithm(variant)
        self.random_gen = QuantumRandomGenerator()
    
    def _get_algorithm(self, variant: QuantumAlgorithm):
        """Get Kyber algorithm implementation"""
        if not PQCRYPTO_AVAILABLE:
            raise QuantumCryptoError("PQCrypto library not available")
        
        algorithms = {
            QuantumAlgorithm.KYBER_512: kyber512,
            QuantumAlgorithm.KYBER_768: kyber768,
            QuantumAlgorithm.KYBER_1024: kyber1024,
        }
        
        if variant not in algorithms:
            raise QuantumCryptoError(f"Unsupported Kyber variant: {variant}")
        
        return algorithms[variant]
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Kyber key pair"""
        try:
            public_key, private_key = self.algorithm.generate_keypair()
            logger.debug(f"Generated Kyber keypair: {self.variant}")
            return public_key, private_key
        except Exception as e:
            raise QuantumCryptoError(f"Kyber key generation failed: {e}")
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate shared secret with Kyber"""
        try:
            ciphertext, shared_secret = self.algorithm.encapsulate(public_key)
            logger.debug(f"Kyber encapsulation successful: {self.variant}")
            return ciphertext, shared_secret
        except Exception as e:
            raise QuantumCryptoError(f"Kyber encapsulation failed: {e}")
    
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate shared secret with Kyber"""
        try:
            shared_secret = self.algorithm.decapsulate(private_key, ciphertext)
            logger.debug(f"Kyber decapsulation successful: {self.variant}")
            return shared_secret
        except Exception as e:
            raise QuantumCryptoError(f"Kyber decapsulation failed: {e}")

class DilithiumSignature:
    """CRYSTALS-Dilithium Digital Signature"""
    
    def __init__(self, variant: QuantumAlgorithm = QuantumAlgorithm.DILITHIUM_5):
        self.variant = variant
        self.algorithm = self._get_algorithm(variant)
        self.random_gen = QuantumRandomGenerator()
    
    def _get_algorithm(self, variant: QuantumAlgorithm):
        """Get Dilithium algorithm implementation"""
        if not PQCRYPTO_AVAILABLE:
            raise QuantumCryptoError("PQCrypto library not available")
        
        algorithms = {
            QuantumAlgorithm.DILITHIUM_2: dilithium2,
            QuantumAlgorithm.DILITHIUM_3: dilithium3,
            QuantumAlgorithm.DILITHIUM_5: dilithium5,
        }
        
        if variant not in algorithms:
            raise QuantumCryptoError(f"Unsupported Dilithium variant: {variant}")
        
        return algorithms[variant]
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Dilithium key pair"""
        try:
            public_key, private_key = self.algorithm.generate_keypair()
            logger.debug(f"Generated Dilithium keypair: {self.variant}")
            return public_key, private_key
        except Exception as e:
            raise QuantumCryptoError(f"Dilithium key generation failed: {e}")
    
    def sign(self, private_key: bytes, message: bytes) -> bytes:
        """Sign message with Dilithium"""
        try:
            signature = self.algorithm.sign(private_key, message)
            logger.debug(f"Dilithium signature created: {self.variant}")
            return signature
        except Exception as e:
            raise QuantumCryptoError(f"Dilithium signing failed: {e}")
    
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify Dilithium signature"""
        try:
            result = self.algorithm.verify(public_key, message, signature)
            logger.debug(f"Dilithium signature verified: {self.variant}")
            return result
        except Exception as e:
            logger.error(f"Dilithium verification failed: {e}")
            return False

class SPHINCSSignature:
    """SPHINCS+ Hash-based Digital Signature"""
    
    def __init__(self, variant: QuantumAlgorithm = QuantumAlgorithm.SPHINCS_SHA256_256S):
        self.variant = variant
        self.algorithm = self._get_algorithm(variant)
        self.random_gen = QuantumRandomGenerator()
    
    def _get_algorithm(self, variant: QuantumAlgorithm):
        """Get SPHINCS+ algorithm implementation"""
        if not PQCRYPTO_AVAILABLE:
            raise QuantumCryptoError("PQCrypto library not available")
        
        algorithms = {
            QuantumAlgorithm.SPHINCS_SHA256_256S: sphincssha256256ssimple,
            QuantumAlgorithm.SPHINCS_HARAKA_256S: sphincsharaka256ssimple,
        }
        
        if variant not in algorithms:
            raise QuantumCryptoError(f"Unsupported SPHINCS+ variant: {variant}")
        
        return algorithms[variant]
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate SPHINCS+ key pair"""
        try:
            public_key, private_key = self.algorithm.generate_keypair()
            logger.debug(f"Generated SPHINCS+ keypair: {self.variant}")
            return public_key, private_key
        except Exception as e:
            raise QuantumCryptoError(f"SPHINCS+ key generation failed: {e}")
    
    def sign(self, private_key: bytes, message: bytes) -> bytes:
        """Sign message with SPHINCS+"""
        try:
            signature = self.algorithm.sign(private_key, message)
            logger.debug(f"SPHINCS+ signature created: {self.variant}")
            return signature
        except Exception as e:
            raise QuantumCryptoError(f"SPHINCS+ signing failed: {e}")
    
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify SPHINCS+ signature"""
        try:
            result = self.algorithm.verify(public_key, message, signature)
            logger.debug(f"SPHINCS+ signature verified: {self.variant}")
            return result
        except Exception as e:
            logger.error(f"SPHINCS+ verification failed: {e}")
            return False

# =============================================================================
# HYBRID CRYPTOGRAPHY
# =============================================================================

class HybridCrypto:
    """Hybrid classical/quantum cryptography"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.quantum_kem = KyberKEM(config.default_kem)
        self.quantum_sig = DilithiumSignature(config.default_signature)
        self.random_gen = QuantumRandomGenerator()
        
        # Classical algorithms for hybrid mode
        self.classical_private_key = None
        self.classical_public_key = None
        
        if config.crypto_mode == CryptoMode.HYBRID:
            self._init_classical_crypto()
    
    def _init_classical_crypto(self):
        """Initialize classical cryptography components"""
        # Generate RSA key pair
        self.classical_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.classical_key_size,
            backend=default_backend()
        )
        self.classical_public_key = self.classical_private_key.public_key()
        
        logger.info("Hybrid cryptography initialized")
    
    def generate_hybrid_keypair(self) -> Dict[str, Any]:
        """Generate hybrid key pair"""
        # Generate quantum-resistant keys
        quantum_pub, quantum_priv = self.quantum_kem.generate_keypair()
        quantum_sig_pub, quantum_sig_priv = self.quantum_sig.generate_keypair()
        
        keypair = {
            'quantum_kem_public': quantum_pub,
            'quantum_kem_private': quantum_priv,
            'quantum_sig_public': quantum_sig_pub,
            'quantum_sig_private': quantum_sig_priv,
            'timestamp': datetime.utcnow().isoformat(),
            'algorithm_info': {
                'kem': self.config.default_kem.value,
                'signature': self.config.default_signature.value,
                'security_level': self.config.security_level.value
            }
        }
        
        # Add classical keys if hybrid mode
        if self.config.crypto_mode == CryptoMode.HYBRID:
            keypair['classical_public'] = self.classical_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            keypair['classical_private'] = self.classical_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        
        logger.info("Hybrid keypair generated successfully")
        return keypair
    
    def hybrid_encrypt(self, public_keys: Dict[str, bytes], message: bytes) -> Dict[str, Any]:
        """Hybrid encryption with quantum and classical algorithms"""
        # Quantum encryption
        quantum_ciphertext, quantum_shared_secret = self.quantum_kem.encapsulate(
            public_keys['quantum_kem_public']
        )
        
        # Derive encryption key from quantum shared secret
        quantum_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'quantum_encryption',
            backend=default_backend()
        ).derive(quantum_shared_secret)
        
        # Encrypt message with quantum-derived key
        iv = self.random_gen.generate_bytes(16)
        cipher = Cipher(
            algorithms.AES(quantum_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Pad message to block size
        padded_message = self._pad_message(message)
        quantum_encrypted_message = encryptor.update(padded_message) + encryptor.finalize()
        
        result = {
            'quantum_ciphertext': quantum_ciphertext,
            'quantum_encrypted_message': quantum_encrypted_message,
            'iv': iv,
            'timestamp': datetime.utcnow().isoformat(),
            'algorithm': self.config.default_kem.value
        }
        
        # Classical backup encryption if hybrid mode
        if self.config.crypto_mode == CryptoMode.HYBRID and 'classical_public' in public_keys:
            classical_pub_key = serialization.load_pem_public_key(
                public_keys['classical_public'],
                backend=default_backend()
            )
            
            # Encrypt with classical RSA
            classical_ciphertext = classical_pub_key.encrypt(
                message[:245],  # RSA limitation
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            result['classical_ciphertext'] = classical_ciphertext
            result['classical_backup'] = True
        
        logger.info("Hybrid encryption completed")
        return result
    
    def hybrid_decrypt(self, private_keys: Dict[str, bytes], encrypted_data: Dict[str, Any]) -> bytes:
        """Hybrid decryption with quantum and classical algorithms"""
        try:
            # Quantum decryption
            quantum_shared_secret = self.quantum_kem.decapsulate(
                private_keys['quantum_kem_private'],
                encrypted_data['quantum_ciphertext']
            )
            
            # Derive decryption key
            quantum_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'quantum_encryption',
                backend=default_backend()
            ).derive(quantum_shared_secret)
            
            # Decrypt message
            cipher = Cipher(
                algorithms.AES(quantum_key),
                modes.CBC(encrypted_data['iv']),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            padded_message = decryptor.update(encrypted_data['quantum_encrypted_message']) + decryptor.finalize()
            message = self._unpad_message(padded_message)
            
            logger.info("Hybrid decryption completed (quantum)")
            return message
            
        except Exception as e:
            logger.warning(f"Quantum decryption failed: {e}")
            
            # Fall back to classical decryption if available
            if (self.config.crypto_mode == CryptoMode.HYBRID and 
                'classical_ciphertext' in encrypted_data and
                'classical_private' in private_keys):
                
                try:
                    classical_priv_key = serialization.load_pem_private_key(
                        private_keys['classical_private'],
                        password=None,
                        backend=default_backend()
                    )
                    
                    message = classical_priv_key.decrypt(
                        encrypted_data['classical_ciphertext'],
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    
                    logger.info("Hybrid decryption completed (classical fallback)")
                    return message
                    
                except Exception as e2:
                    logger.error(f"Classical decryption also failed: {e2}")
            
            raise QuantumCryptoError(f"Hybrid decryption failed: {e}")
    
    def hybrid_sign(self, private_keys: Dict[str, bytes], message: bytes) -> Dict[str, Any]:
        """Hybrid signing with quantum and classical algorithms"""
        # Quantum signature
        quantum_signature = self.quantum_sig.sign(
            private_keys['quantum_sig_private'],
            message
        )
        
        result = {
            'quantum_signature': quantum_signature,
            'message_hash': hashlib.sha256(message).hexdigest(),
            'timestamp': datetime.utcnow().isoformat(),
            'algorithm': self.config.default_signature.value
        }
        
        # Classical signature if hybrid mode
        if self.config.crypto_mode == CryptoMode.HYBRID and 'classical_private' in private_keys:
            classical_priv_key = serialization.load_pem_private_key(
                private_keys['classical_private'],
                password=None,
                backend=default_backend()
            )
            
            classical_signature = classical_priv_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            result['classical_signature'] = classical_signature
            result['classical_backup'] = True
        
        logger.info("Hybrid signing completed")
        return result
    
    def hybrid_verify(self, public_keys: Dict[str, bytes], message: bytes, signatures: Dict[str, Any]) -> bool:
        """Hybrid signature verification"""
        # Verify quantum signature
        try:
            quantum_valid = self.quantum_sig.verify(
                public_keys['quantum_sig_public'],
                message,
                signatures['quantum_signature']
            )
            
            if quantum_valid:
                logger.info("Hybrid verification successful (quantum)")
                return True
                
        except Exception as e:
            logger.warning(f"Quantum signature verification failed: {e}")
        
        # Fall back to classical verification if available
        if (self.config.crypto_mode == CryptoMode.HYBRID and 
            'classical_signature' in signatures and
            'classical_public' in public_keys):
            
            try:
                classical_pub_key = serialization.load_pem_public_key(
                    public_keys['classical_public'],
                    backend=default_backend()
                )
                
                classical_pub_key.verify(
                    signatures['classical_signature'],
                    message,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                logger.info("Hybrid verification successful (classical fallback)")
                return True
                
            except Exception as e:
                logger.error(f"Classical signature verification failed: {e}")
        
        logger.error("Hybrid verification failed")
        return False
    
    def _pad_message(self, message: bytes) -> bytes:
        """PKCS7 padding"""
        block_size = 16
        padding_length = block_size - (len(message) % block_size)
        padding = bytes([padding_length] * padding_length)
        return message + padding
    
    def _unpad_message(self, padded_message: bytes) -> bytes:
        """Remove PKCS7 padding"""
        padding_length = padded_message[-1]
        return padded_message[:-padding_length]

# =============================================================================
# QUANTUM THREAT ASSESSMENT
# =============================================================================

class QuantumThreatAssessment:
    """Quantum threat timeline and migration planning"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.current_year = datetime.now().year
        self.threat_timeline = config.quantum_threat_timeline
        self.migration_deadline = config.migration_deadline
    
    def assess_threat_level(self) -> Dict[str, Any]:
        """Assess current quantum threat level"""
        years_until_threat = self.threat_timeline - self.current_year
        years_until_migration = (self.migration_deadline - datetime.now()).days / 365
        
        if years_until_threat <= 0:
            threat_level = "CRITICAL"
            urgency = "IMMEDIATE"
        elif years_until_threat <= 2:
            threat_level = "HIGH"
            urgency = "URGENT"
        elif years_until_threat <= 5:
            threat_level = "MEDIUM"
            urgency = "MODERATE"
        else:
            threat_level = "LOW"
            urgency = "PLANNED"
        
        return {
            'threat_level': threat_level,
            'urgency': urgency,
            'years_until_threat': years_until_threat,
            'years_until_migration_deadline': years_until_migration,
            'migration_progress': self._calculate_migration_progress(),
            'recommendations': self._get_recommendations(threat_level)
        }
    
    def _calculate_migration_progress(self) -> float:
        """Calculate migration progress percentage"""
        # This would be based on actual system audit
        # For now, return estimated progress
        if self.config.crypto_mode == CryptoMode.QUANTUM_ONLY:
            return 100.0
        elif self.config.crypto_mode == CryptoMode.HYBRID:
            return 75.0
        else:
            return 0.0
    
    def _get_recommendations(self, threat_level: str) -> List[str]:
        """Get recommendations based on threat level"""
        recommendations = []
        
        if threat_level == "CRITICAL":
            recommendations.extend([
                "Immediately switch to quantum-only mode",
                "Audit all cryptographic implementations",
                "Implement emergency quantum key distribution",
                "Disable all classical-only cryptography"
            ])
        elif threat_level == "HIGH":
            recommendations.extend([
                "Accelerate quantum migration timeline",
                "Implement hybrid cryptography immediately",
                "Begin quantum key distribution deployment",
                "Increase security monitoring"
            ])
        elif threat_level == "MEDIUM":
            recommendations.extend([
                "Begin hybrid cryptography implementation",
                "Plan quantum key distribution infrastructure",
                "Audit current cryptographic dependencies",
                "Establish quantum migration timeline"
            ])
        else:
            recommendations.extend([
                "Monitor quantum computing developments",
                "Begin quantum cryptography research",
                "Evaluate post-quantum algorithms",
                "Prepare migration planning"
            ])
        
        return recommendations

# =============================================================================
# CRYPTOGRAPHIC AGILITY FRAMEWORK
# =============================================================================

class CryptoAgilityManager:
    """Cryptographic agility for algorithm migration"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.algorithms = {}
        self.migration_plan = {}
        self._register_algorithms()
    
    def _register_algorithms(self):
        """Register available cryptographic algorithms"""
        # Quantum-resistant algorithms
        self.algorithms['kyber512'] = KyberKEM(QuantumAlgorithm.KYBER_512)
        self.algorithms['kyber768'] = KyberKEM(QuantumAlgorithm.KYBER_768)
        self.algorithms['kyber1024'] = KyberKEM(QuantumAlgorithm.KYBER_1024)
        
        self.algorithms['dilithium2'] = DilithiumSignature(QuantumAlgorithm.DILITHIUM_2)
        self.algorithms['dilithium3'] = DilithiumSignature(QuantumAlgorithm.DILITHIUM_3)
        self.algorithms['dilithium5'] = DilithiumSignature(QuantumAlgorithm.DILITHIUM_5)
        
        self.algorithms['sphincs_sha256'] = SPHINCSSignature(QuantumAlgorithm.SPHINCS_SHA256_256S)
        self.algorithms['sphincs_haraka'] = SPHINCSSignature(QuantumAlgorithm.SPHINCS_HARAKA_256S)
        
        logger.info("Cryptographic algorithms registered")
    
    def create_migration_plan(self, target_algorithms: Dict[str, str]) -> Dict[str, Any]:
        """Create algorithm migration plan"""
        migration_plan = {
            'current_algorithms': {
                'kem': self.config.default_kem.value,
                'signature': self.config.default_signature.value
            },
            'target_algorithms': target_algorithms,
            'migration_steps': [],
            'timeline': {},
            'rollback_plan': {}
        }
        
        # Generate migration steps
        if target_algorithms.get('kem') != self.config.default_kem.value:
            migration_plan['migration_steps'].append({
                'step': 'kem_migration',
                'description': f"Migrate from {self.config.default_kem.value} to {target_algorithms['kem']}",
                'risk_level': 'medium',
                'estimated_time': '2-4 weeks'
            })
        
        if target_algorithms.get('signature') != self.config.default_signature.value:
            migration_plan['migration_steps'].append({
                'step': 'signature_migration',
                'description': f"Migrate from {self.config.default_signature.value} to {target_algorithms['signature']}",
                'risk_level': 'medium',
                'estimated_time': '2-4 weeks'
            })
        
        # Add timeline
        migration_plan['timeline'] = {
            'start_date': datetime.now().isoformat(),
            'estimated_completion': (datetime.now() + timedelta(weeks=8)).isoformat(),
            'phases': [
                {'phase': 'planning', 'duration': '1 week'},
                {'phase': 'implementation', 'duration': '4-6 weeks'},
                {'phase': 'testing', 'duration': '2 weeks'},
                {'phase': 'deployment', 'duration': '1 week'}
            ]
        }
        
        self.migration_plan = migration_plan
        logger.info("Migration plan created")
        return migration_plan
    
    def execute_migration_step(self, step_name: str) -> bool:
        """Execute a specific migration step"""
        try:
            if step_name == 'kem_migration':
                # Implement KEM migration logic
                logger.info("Executing KEM migration")
                return True
            elif step_name == 'signature_migration':
                # Implement signature migration logic
                logger.info("Executing signature migration")
                return True
            else:
                logger.error(f"Unknown migration step: {step_name}")
                return False
        except Exception as e:
            logger.error(f"Migration step failed: {e}")
            return False
    
    def validate_migration(self) -> Dict[str, Any]:
        """Validate migration success"""
        validation_results = {
            'success': True,
            'errors': [],
            'warnings': [],
            'performance_impact': {}
        }
        
        # Test all algorithms
        for algo_name, algorithm in self.algorithms.items():
            try:
                if hasattr(algorithm, 'generate_keypair'):
                    pub, priv = algorithm.generate_keypair()
                    validation_results['performance_impact'][algo_name] = 'tested'
            except Exception as e:
                validation_results['errors'].append(f"Algorithm {algo_name} failed: {e}")
                validation_results['success'] = False
        
        return validation_results

# =============================================================================
# QUANTUM CRYPTO MANAGER
# =============================================================================

class QuantumCryptoManager:
    """Main quantum cryptography manager"""
    
    def __init__(self, config: Optional[QuantumConfig] = None):
        self.config = config or QuantumConfig()
        self.hybrid_crypto = HybridCrypto(self.config)
        self.threat_assessment = QuantumThreatAssessment(self.config)
        self.agility_manager = CryptoAgilityManager(self.config)
        self.random_gen = QuantumRandomGenerator()
        
        logger.info("Quantum cryptography manager initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        threat_status = self.threat_assessment.assess_threat_level()
        
        return {
            'configuration': {
                'kem_algorithm': self.config.default_kem.value,
                'signature_algorithm': self.config.default_signature.value,
                'security_level': self.config.security_level.value,
                'crypto_mode': self.config.crypto_mode.value
            },
            'threat_assessment': threat_status,
            'pqcrypto_available': PQCRYPTO_AVAILABLE,
            'algorithms_available': list(self.agility_manager.algorithms.keys()),
            'migration_status': self.agility_manager.migration_plan,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def generate_keypair(self) -> Dict[str, Any]:
        """Generate quantum-safe key pair"""
        return self.hybrid_crypto.generate_hybrid_keypair()
    
    def encrypt(self, public_keys: Dict[str, bytes], message: bytes) -> Dict[str, Any]:
        """Encrypt with quantum-safe algorithms"""
        return self.hybrid_crypto.hybrid_encrypt(public_keys, message)
    
    def decrypt(self, private_keys: Dict[str, bytes], encrypted_data: Dict[str, Any]) -> bytes:
        """Decrypt with quantum-safe algorithms"""
        return self.hybrid_crypto.hybrid_decrypt(private_keys, encrypted_data)
    
    def sign(self, private_keys: Dict[str, bytes], message: bytes) -> Dict[str, Any]:
        """Sign with quantum-safe algorithms"""
        return self.hybrid_crypto.hybrid_sign(private_keys, message)
    
    def verify(self, public_keys: Dict[str, bytes], message: bytes, signatures: Dict[str, Any]) -> bool:
        """Verify quantum-safe signatures"""
        return self.hybrid_crypto.hybrid_verify(public_keys, message, signatures)

# =============================================================================
# GLOBAL QUANTUM CRYPTO MANAGER
# =============================================================================

# Global quantum crypto manager instance
quantum_crypto_manager: Optional[QuantumCryptoManager] = None

def init_quantum_crypto_manager(config: Optional[QuantumConfig] = None) -> QuantumCryptoManager:
    """Initialize global quantum crypto manager"""
    global quantum_crypto_manager
    if not quantum_crypto_manager:
        quantum_crypto_manager = QuantumCryptoManager(config)
    return quantum_crypto_manager

def get_quantum_crypto_manager() -> QuantumCryptoManager:
    """Get global quantum crypto manager"""
    if not quantum_crypto_manager:
        return init_quantum_crypto_manager()
    return quantum_crypto_manager

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of quantum cryptography"""
    
    # Initialize quantum crypto manager
    manager = init_quantum_crypto_manager()
    
    # Get system status
    status = manager.get_system_status()
    print("System Status:", json.dumps(status, indent=2))
    
    # Generate keypair
    keypair = manager.generate_keypair()
    print("Keypair generated")
    
    # Encrypt message
    message = b"This is a secret message that needs quantum protection!"
    encrypted = manager.encrypt({
        'quantum_kem_public': keypair['quantum_kem_public'],
        'classical_public': keypair.get('classical_public', b'')
    }, message)
    
    print("Message encrypted")
    
    # Decrypt message
    decrypted = manager.decrypt({
        'quantum_kem_private': keypair['quantum_kem_private'],
        'classical_private': keypair.get('classical_private', b'')
    }, encrypted)
    
    print(f"Decrypted message: {decrypted}")
    
    # Sign message
    signature = manager.sign({
        'quantum_sig_private': keypair['quantum_sig_private'],
        'classical_private': keypair.get('classical_private', b'')
    }, message)
    
    print("Message signed")
    
    # Verify signature
    valid = manager.verify({
        'quantum_sig_public': keypair['quantum_sig_public'],
        'classical_public': keypair.get('classical_public', b'')
    }, message, signature)
    
    print(f"Signature valid: {valid}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())