"""
Cryptographic Core Test Suite

Comprehensive tests for HMAC-SHA256 cryptographic validation system:
- Message signing and validation
- Key generation and management
- Replay attack prevention
- Key rotation
- Performance requirements

Author: Agent 2 - Consensus Security Engineer
Version: 1.0 - Production Ready
"""

import pytest
import time
import secrets
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.consensus.cryptographic_core import (
    CryptographicCore, MessageValidator, CryptoKey, MessageSignature
)
from src.consensus.pbft_engine import PBFTMessage, MessageType, PBFTPhase


class TestCryptographicCore:
    """Test suite for cryptographic core"""
    
    @pytest.fixture
    def crypto_core(self):
        """Cryptographic core instance for testing"""
        return CryptographicCore()
    
    @pytest.fixture
    def agent_ids(self):
        """Standard agent IDs for testing"""
        return ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
    
    def test_master_secret_generation(self, crypto_core):
        """Test master secret generation"""
        assert len(crypto_core.master_secret) == 32  # 256 bits
        assert isinstance(crypto_core.master_secret, bytes)
        
        # Different instances should have different secrets
        crypto_core2 = CryptographicCore()
        assert crypto_core.master_secret != crypto_core2.master_secret
    
    def test_agent_key_initialization(self, crypto_core, agent_ids):
        """Test agent key initialization"""
        fingerprints = crypto_core.initialize_agent_keys(agent_ids)
        
        # Verify all agents have keys
        assert len(fingerprints) == len(agent_ids)
        for agent_id in agent_ids:
            assert agent_id in fingerprints
            assert len(fingerprints[agent_id]) == 16  # Fingerprint length
            assert agent_id in crypto_core.hmac_keys
            assert agent_id in crypto_core.rsa_keys
    
    def test_hmac_key_derivation(self, crypto_core):
        """Test HMAC key derivation"""
        agent_id = 'test_agent'
        
        # Generate key twice - should be same
        key1 = crypto_core._derive_hmac_key(agent_id)
        key2 = crypto_core._derive_hmac_key(agent_id)
        
        assert key1 == key2
        assert len(key1) == 32  # 256-bit key
        
        # Different agents should have different keys
        key3 = crypto_core._derive_hmac_key('different_agent')
        assert key1 != key3
    
    def test_rsa_key_generation(self, crypto_core):
        """Test RSA key pair generation"""
        private_key, public_key = crypto_core._generate_rsa_key_pair()
        
        assert private_key is not None
        assert public_key is not None
        assert public_key.key_size == 2048
        
        # Different calls should generate different keys
        private_key2, public_key2 = crypto_core._generate_rsa_key_pair()
        assert private_key != private_key2
    
    def test_message_signing_and_validation(self, crypto_core, agent_ids):
        """Test message signing and validation"""
        crypto_core.initialize_agent_keys(agent_ids)
        
        message_hash = "test_message_hash_123456789abcdef"
        signer_id = agent_ids[0]
        
        # Sign message
        signature = crypto_core.sign_message(message_hash, signer_id)
        
        assert signature is not None
        assert isinstance(signature, str)
        
        # Validate signature
        is_valid = crypto_core.validate_signature(message_hash, signature, signer_id)
        assert is_valid is True
        
        # Invalid signature should fail
        is_valid_wrong = crypto_core.validate_signature("wrong_hash", signature, signer_id)
        assert is_valid_wrong is False
        
        # Wrong signer should fail
        is_valid_wrong_signer = crypto_core.validate_signature(message_hash, signature, agent_ids[1])
        assert is_valid_wrong_signer is False
    
    def test_nonce_replay_attack_prevention(self, crypto_core, agent_ids):
        """Test replay attack prevention with nonces"""
        crypto_core.initialize_agent_keys(agent_ids)
        
        message_hash = "test_message_hash"
        signer_id = agent_ids[0]
        
        # Sign message twice
        signature1 = crypto_core.sign_message(message_hash, signer_id)
        signature2 = crypto_core.sign_message(message_hash, signer_id)
        
        # Signatures should be different due to nonces
        assert signature1 != signature2
        
        # Both should validate
        assert crypto_core.validate_signature(message_hash, signature1, signer_id) is True
        assert crypto_core.validate_signature(message_hash, signature2, signer_id) is True
        
        # But trying to use same signature again should fail
        assert crypto_core.validate_signature(message_hash, signature1, signer_id) is False
    
    def test_timestamp_validation(self, crypto_core, agent_ids):
        """Test timestamp validation for replay attack prevention"""
        crypto_core.initialize_agent_keys(agent_ids)
        
        message_hash = "test_message"
        signer_id = agent_ids[0]
        
        # Create signature with old timestamp
        with patch('time.time', return_value=time.time() - 400):  # 400 seconds ago
            old_signature = crypto_core.sign_message(message_hash, signer_id)
        
        # Should fail validation due to old timestamp
        is_valid = crypto_core.validate_signature(message_hash, old_signature, signer_id)
        assert is_valid is False
    
    def test_nonce_cleanup(self, crypto_core, agent_ids):
        """Test expired nonce cleanup"""
        crypto_core.initialize_agent_keys(agent_ids)
        
        # Add some old nonces
        old_time = time.time() - 400  # 400 seconds ago
        crypto_core.used_nonces['old_nonce_1'] = old_time
        crypto_core.used_nonces['old_nonce_2'] = old_time
        crypto_core.used_nonces['recent_nonce'] = time.time()
        
        initial_count = len(crypto_core.used_nonces)
        
        # Cleanup expired nonces
        crypto_core.cleanup_expired_nonces()
        
        # Should remove old nonces but keep recent ones
        assert len(crypto_core.used_nonces) < initial_count
        assert 'recent_nonce' in crypto_core.used_nonces
        assert 'old_nonce_1' not in crypto_core.used_nonces
    
    def test_key_rotation(self, crypto_core, agent_ids):
        """Test cryptographic key rotation"""
        # Initialize keys
        initial_fingerprints = crypto_core.initialize_agent_keys(agent_ids)
        
        # Record initial key timestamp
        initial_rotation_time = crypto_core.last_key_rotation
        
        # Force key rotation
        new_fingerprints = crypto_core.rotate_keys(agent_ids)
        
        # Verify rotation occurred
        assert crypto_core.last_key_rotation > initial_rotation_time
        assert crypto_core.security_metrics['key_rotations'] >= 1
        
        # New fingerprints should be different
        for agent_id in agent_ids:
            assert new_fingerprints[agent_id] != initial_fingerprints[agent_id]
    
    def test_key_rotation_triggering(self, crypto_core, agent_ids):
        """Test automatic key rotation triggering"""
        crypto_core.initialize_agent_keys(agent_ids)
        
        # Should not need rotation initially
        assert crypto_core.should_rotate_keys() is False
        
        # Simulate old keys (time-based rotation)
        crypto_core.last_key_rotation = time.time() - 4000  # More than 1 hour ago
        assert crypto_core.should_rotate_keys() is True
        
        # Reset time but simulate high failure rate
        crypto_core.last_key_rotation = time.time()
        crypto_core.security_metrics['signature_failures'] = 50
        crypto_core.security_metrics['signatures_validated'] = 100
        assert crypto_core.should_rotate_keys() is True
    
    def test_public_key_export_import(self, crypto_core, agent_ids):
        """Test public key export and import"""
        crypto_core.initialize_agent_keys(agent_ids)
        
        # Export public keys
        exported_keys = crypto_core.export_public_keys()
        
        assert len(exported_keys) == len(agent_ids)
        for agent_id in agent_ids:
            assert agent_id in exported_keys
            assert 'BEGIN PUBLIC KEY' in exported_keys[agent_id]
        
        # Create new crypto core and import keys
        crypto_core2 = CryptographicCore()
        crypto_core2.import_public_keys(exported_keys)
        
        # Verify keys were imported
        for agent_id in agent_ids:
            assert agent_id in crypto_core2.rsa_keys
            public_key_info = crypto_core2.get_public_key_info(agent_id)
            assert public_key_info is not None
            assert public_key_info['agent_id'] == agent_id
    
    def test_security_metrics_tracking(self, crypto_core, agent_ids):
        """Test security metrics tracking"""
        crypto_core.initialize_agent_keys(agent_ids)
        
        initial_metrics = crypto_core.get_security_metrics()
        
        # Perform some operations
        message_hash = "test_message"
        signer_id = agent_ids[0]
        
        signature = crypto_core.sign_message(message_hash, signer_id)
        crypto_core.validate_signature(message_hash, signature, signer_id)
        
        # Try invalid validation
        crypto_core.validate_signature("wrong_hash", signature, signer_id)
        
        updated_metrics = crypto_core.get_security_metrics()
        
        # Verify metrics updated
        assert updated_metrics['signatures_created'] > initial_metrics['signatures_created']
        assert updated_metrics['signatures_validated'] > initial_metrics['signatures_validated']
        assert updated_metrics['signature_failures'] > initial_metrics['signature_failures']
        assert 'signature_success_rate' in updated_metrics
        assert 'failure_rate' in updated_metrics
    
    def test_crypto_key_container(self):
        """Test CryptoKey container class"""
        key_data = secrets.token_bytes(32)
        
        crypto_key = CryptoKey(
            key_id='test_key',
            key_data=key_data,
            key_type='hmac',
            created_at=time.time()
        )
        
        assert crypto_key.key_id == 'test_key'
        assert crypto_key.key_data == key_data
        assert crypto_key.key_type == 'hmac'
        assert crypto_key.is_active is True
    
    def test_message_signature_container(self):
        """Test MessageSignature container class"""
        signature = MessageSignature(
            signature='test_signature',
            signer_id='test_signer',
            timestamp=time.time(),
            nonce='test_nonce'
        )
        
        assert signature.signature == 'test_signature'
        assert signature.signer_id == 'test_signer'
        assert signature.algorithm == 'HMAC-SHA256'
        assert signature.nonce == 'test_nonce'


class TestMessageValidator:
    """Test suite for message validator"""
    
    @pytest.fixture
    def crypto_core(self):
        """Cryptographic core for validator testing"""
        crypto = CryptographicCore()
        agent_ids = ['validator_agent']
        crypto.initialize_agent_keys(agent_ids)
        return crypto
    
    @pytest.fixture
    def message_validator(self, crypto_core):
        """Message validator instance"""
        return MessageValidator(crypto_core)
    
    def test_valid_message_validation(self, message_validator, crypto_core):
        """Test validation of valid PBFT message"""
        # Create valid message
        msg = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=0,
            sequence_number=1,
            sender_id='validator_agent',
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.PRE_PREPARE
        )
        
        # Sign message
        msg.signature = crypto_core.sign_message(msg.get_hash(), 'validator_agent')
        
        # Validate
        assert message_validator.validate_message(msg) is True
    
    def test_invalid_signature_validation(self, message_validator):
        """Test validation with invalid signature"""
        msg = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=0,
            sequence_number=1,
            sender_id='validator_agent',
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.PRE_PREPARE
        )
        
        # No signature
        assert message_validator.validate_message(msg) is False
        
        # Invalid signature
        msg.signature = 'invalid_signature'
        assert message_validator.validate_message(msg) is False
    
    def test_message_structure_validation(self, message_validator, crypto_core):
        """Test message structure validation"""
        # Valid message
        msg = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=0,
            sequence_number=1,
            sender_id='validator_agent',
            payload={'test': 'data'},
            timestamp=time.time(),
            phase=PBFTPhase.PRE_PREPARE
        )
        msg.signature = crypto_core.sign_message(msg.get_hash(), 'validator_agent')
        
        assert message_validator._validate_message_structure(msg) is True
        
        # Invalid: negative view number
        msg.view_number = -1
        assert message_validator._validate_message_structure(msg) is False
        
        # Invalid: invalid sequence number
        msg.view_number = 0
        msg.sequence_number = -2
        assert message_validator._validate_message_structure(msg) is False
        
        # Invalid: no sender ID
        msg.sequence_number = 1
        msg.sender_id = ''
        assert message_validator._validate_message_structure(msg) is False
    
    def test_timestamp_validation(self, message_validator, crypto_core):
        """Test message timestamp validation"""
        current_time = time.time()
        
        # Old timestamp
        msg_old = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=0,
            sequence_number=1,
            sender_id='validator_agent',
            payload={'test': 'data'},
            timestamp=current_time - 120,  # 2 minutes old
            phase=PBFTPhase.PRE_PREPARE
        )
        msg_old.signature = crypto_core.sign_message(msg_old.get_hash(), 'validator_agent')
        
        assert message_validator._validate_message_structure(msg_old) is False
        
        # Future timestamp
        msg_future = PBFTMessage(
            message_type=MessageType.PRE_PREPARE,
            view_number=0,
            sequence_number=1,
            sender_id='validator_agent',
            payload={'test': 'data'},
            timestamp=current_time + 20,  # 20 seconds in future
            phase=PBFTPhase.PRE_PREPARE
        )
        msg_future.signature = crypto_core.sign_message(msg_future.get_hash(), 'validator_agent')
        
        assert message_validator._validate_message_structure(msg_future) is False


class TestCryptographicPerformance:
    """Performance tests for cryptographic operations"""
    
    @pytest.fixture
    def perf_crypto_core(self):
        """Crypto core for performance testing"""
        crypto = CryptographicCore()
        agent_ids = [f'perf_agent_{i}' for i in range(10)]
        crypto.initialize_agent_keys(agent_ids)
        return crypto
    
    def test_signing_performance(self, perf_crypto_core):
        """Test message signing performance"""
        message_hash = "performance_test_message_hash"
        signer_id = 'perf_agent_0'
        
        # Measure signing time
        start_time = time.time()
        signatures = []
        
        for _ in range(100):
            signature = perf_crypto_core.sign_message(message_hash, signer_id)
            signatures.append(signature)
        
        total_time = time.time() - start_time
        avg_time = total_time / 100
        
        # Should be able to sign at least 100 messages per second
        assert avg_time < 0.01, f"Signing too slow: {avg_time:.4f}s per signature"
        
        # All signatures should be unique (due to nonces)
        assert len(set(signatures)) == 100
    
    def test_validation_performance(self, perf_crypto_core):
        """Test message validation performance"""
        message_hash = "performance_test_validation"
        signer_id = 'perf_agent_0'
        
        # Pre-generate signatures
        signatures = []
        for i in range(50):
            signature = perf_crypto_core.sign_message(f"{message_hash}_{i}", signer_id)
            signatures.append((f"{message_hash}_{i}", signature))
        
        # Measure validation time
        start_time = time.time()
        
        valid_count = 0
        for msg_hash, signature in signatures:
            if perf_crypto_core.validate_signature(msg_hash, signature, signer_id):
                valid_count += 1
        
        total_time = time.time() - start_time
        avg_time = total_time / len(signatures)
        
        # Should be able to validate at least 50 messages per second
        assert avg_time < 0.02, f"Validation too slow: {avg_time:.4f}s per validation"
        assert valid_count == len(signatures)
    
    def test_key_generation_performance(self):
        """Test key generation performance"""
        start_time = time.time()
        
        # Generate keys for multiple agents
        crypto = CryptographicCore()
        agent_ids = [f'keygen_agent_{i}' for i in range(20)]
        fingerprints = crypto.initialize_agent_keys(agent_ids)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(agent_ids)
        
        # Should generate keys quickly
        assert avg_time < 0.5, f"Key generation too slow: {avg_time:.3f}s per agent"
        assert len(fingerprints) == len(agent_ids)


@pytest.mark.integration
class TestCryptographicIntegration:
    """Integration tests for cryptographic core"""
    
    def test_end_to_end_message_flow(self):
        """Test complete end-to-end message flow with crypto"""
        # Setup
        crypto = CryptographicCore()
        agent_ids = ['sender', 'receiver']
        crypto.initialize_agent_keys(agent_ids)
        validator = MessageValidator(crypto)
        
        # Create message
        msg = PBFTMessage(
            message_type=MessageType.PREPARE,
            view_number=1,
            sequence_number=5,
            sender_id='sender',
            payload={'action': 'trade', 'amount': 1000},
            timestamp=time.time(),
            phase=PBFTPhase.PREPARE
        )
        
        # Sign message
        msg.signature = crypto.sign_message(msg.get_hash(), 'sender')
        
        # Validate message
        assert validator.validate_message(msg) is True
        
        # Message should be deterministically hashable
        hash1 = msg.get_hash()
        hash2 = msg.get_hash()
        assert hash1 == hash2
    
    def test_multi_agent_cross_validation(self):
        """Test cross-validation between multiple agents"""
        crypto = CryptographicCore()
        agent_ids = ['agent_a', 'agent_b', 'agent_c']
        crypto.initialize_agent_keys(agent_ids)
        
        message_hash = "cross_validation_test"
        
        # Each agent signs the same message
        signatures = {}
        for agent_id in agent_ids:
            signatures[agent_id] = crypto.sign_message(message_hash, agent_id)
        
        # Each signature should validate for its respective agent
        for agent_id, signature in signatures.items():
            assert crypto.validate_signature(message_hash, signature, agent_id) is True
            
            # But not for other agents
            for other_agent in agent_ids:
                if other_agent != agent_id:
                    assert crypto.validate_signature(message_hash, signature, other_agent) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])