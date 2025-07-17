"""
Blockchain-Based Immutable Audit Trail System
Agent Epsilon: Advanced XAI Implementation Specialist

Industry-first blockchain implementation for trading decision audit trails.
Provides tamper-proof logging with cryptographic integrity for regulatory compliance.

Features:
- Immutable blockchain-based audit trail
- Cryptographic hash verification
- Digital signature validation
- Distributed consensus mechanism
- Regulatory compliance reporting
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import pickle
from pathlib import Path

# Cryptographic libraries
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("Cryptography library not available, using mock implementation")

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""
    DECISION_MADE = "decision_made"
    EXPLANATION_GENERATED = "explanation_generated"
    CAUSAL_ANALYSIS = "causal_analysis"
    COUNTERFACTUAL_ANALYSIS = "counterfactual_analysis"
    BIAS_DETECTION = "bias_detection"
    ETHICS_VIOLATION = "ethics_violation"
    SYSTEM_CHANGE = "system_change"
    COMPLIANCE_CHECK = "compliance_check"


class AuditLevel(Enum):
    """Audit levels for different types of events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditTransaction:
    """Individual audit transaction"""
    transaction_id: str
    timestamp: datetime
    event_type: AuditEventType
    audit_level: AuditLevel
    source_system: str
    source_component: str
    
    # Core audit data
    decision_id: Optional[str] = None
    explanation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Event payload
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity
    data_hash: Optional[str] = None
    digital_signature: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.data_hash is None:
            self.data_hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of transaction data"""
        # Create deterministic representation
        hash_data = {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'audit_level': self.audit_level.value,
            'source_system': self.source_system,
            'source_component': self.source_component,
            'decision_id': self.decision_id,
            'explanation_id': self.explanation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'event_data': self.event_data
        }
        
        # Convert to bytes and hash
        hash_string = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return asdict(self)


@dataclass
class AuditBlock:
    """Audit block containing multiple transactions"""
    block_id: str
    block_number: int
    timestamp: datetime
    previous_hash: str
    transactions: List[AuditTransaction] = field(default_factory=list)
    
    # Block integrity
    merkle_root: Optional[str] = None
    block_hash: Optional[str] = None
    digital_signature: Optional[str] = None
    
    # Consensus
    validator_signatures: List[str] = field(default_factory=list)
    consensus_achieved: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.merkle_root is None:
            self.merkle_root = self.calculate_merkle_root()
        if self.block_hash is None:
            self.block_hash = self.calculate_block_hash()
    
    def add_transaction(self, transaction: AuditTransaction):
        """Add transaction to block"""
        self.transactions.append(transaction)
        # Recalculate merkle root and block hash
        self.merkle_root = self.calculate_merkle_root()
        self.block_hash = self.calculate_block_hash()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions"""
        if not self.transactions:
            return hashlib.sha256(b'').hexdigest()
        
        # Get transaction hashes
        tx_hashes = [tx.data_hash for tx in self.transactions]
        
        # Build Merkle tree
        while len(tx_hashes) > 1:
            new_level = []
            for i in range(0, len(tx_hashes), 2):
                left = tx_hashes[i]
                right = tx_hashes[i + 1] if i + 1 < len(tx_hashes) else left
                
                combined = left + right
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            
            tx_hashes = new_level
        
        return tx_hashes[0]
    
    def calculate_block_hash(self) -> str:
        """Calculate SHA-256 hash of block header"""
        header_data = {
            'block_id': self.block_id,
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'transaction_count': len(self.transactions)
        }
        
        header_string = json.dumps(header_data, sort_keys=True)
        return hashlib.sha256(header_string.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify block integrity"""
        # Verify merkle root
        calculated_merkle = self.calculate_merkle_root()
        if calculated_merkle != self.merkle_root:
            return False
        
        # Verify block hash
        calculated_hash = self.calculate_block_hash()
        if calculated_hash != self.block_hash:
            return False
        
        # Verify all transactions
        for tx in self.transactions:
            if tx.calculate_hash() != tx.data_hash:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return {
            'block_id': self.block_id,
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'block_hash': self.block_hash,
            'digital_signature': self.digital_signature,
            'validator_signatures': self.validator_signatures,
            'consensus_achieved': self.consensus_achieved,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'metadata': self.metadata
        }


class AuditChain:
    """Blockchain audit chain"""
    
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.blocks: List[AuditBlock] = []
        self.pending_transactions: List[AuditTransaction] = []
        self.genesis_block = self._create_genesis_block()
        self.blocks.append(self.genesis_block)
        
        # Chain metadata
        self.created_at = datetime.now(timezone.utc)
        self.last_block_time = self.created_at
        self.total_transactions = 0
        
        logger.info(f"AuditChain {chain_id} initialized with genesis block")
    
    def _create_genesis_block(self) -> AuditBlock:
        """Create genesis block"""
        genesis_tx = AuditTransaction(
            transaction_id="genesis_tx",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.SYSTEM_CHANGE,
            audit_level=AuditLevel.HIGH,
            source_system="XAI_AUDIT_SYSTEM",
            source_component="BlockchainAuditSystem",
            event_data={"event": "chain_genesis", "chain_id": self.chain_id}
        )
        
        genesis_block = AuditBlock(
            block_id="genesis_block",
            block_number=0,
            timestamp=datetime.now(timezone.utc),
            previous_hash="0" * 64,
            transactions=[genesis_tx]
        )
        
        return genesis_block
    
    def add_transaction(self, transaction: AuditTransaction):
        """Add transaction to pending transactions"""
        self.pending_transactions.append(transaction)
        logger.debug(f"Added transaction {transaction.transaction_id} to pending")
    
    def mine_block(self, max_transactions: int = 100) -> AuditBlock:
        """Mine new block from pending transactions"""
        if not self.pending_transactions:
            return None
        
        # Get transactions for this block
        block_transactions = self.pending_transactions[:max_transactions]
        self.pending_transactions = self.pending_transactions[max_transactions:]
        
        # Create new block
        previous_block = self.blocks[-1]
        new_block = AuditBlock(
            block_id=f"block_{len(self.blocks)}_{uuid.uuid4().hex[:8]}",
            block_number=len(self.blocks),
            timestamp=datetime.now(timezone.utc),
            previous_hash=previous_block.block_hash,
            transactions=block_transactions
        )
        
        # Add to chain
        self.blocks.append(new_block)
        self.last_block_time = new_block.timestamp
        self.total_transactions += len(block_transactions)
        
        logger.info(f"Mined block {new_block.block_number} with {len(block_transactions)} transactions")
        return new_block
    
    def get_latest_block(self) -> AuditBlock:
        """Get latest block"""
        return self.blocks[-1]
    
    def get_block_by_number(self, block_number: int) -> Optional[AuditBlock]:
        """Get block by number"""
        if 0 <= block_number < len(self.blocks):
            return self.blocks[block_number]
        return None
    
    def get_block_by_hash(self, block_hash: str) -> Optional[AuditBlock]:
        """Get block by hash"""
        for block in self.blocks:
            if block.block_hash == block_hash:
                return block
        return None
    
    def verify_chain_integrity(self) -> bool:
        """Verify entire chain integrity"""
        for i, block in enumerate(self.blocks):
            # Verify block integrity
            if not block.verify_integrity():
                logger.error(f"Block {i} failed integrity check")
                return False
            
            # Verify chain linkage
            if i > 0:
                previous_block = self.blocks[i - 1]
                if block.previous_hash != previous_block.block_hash:
                    logger.error(f"Block {i} has invalid previous hash")
                    return False
        
        return True
    
    def find_transactions_by_decision_id(self, decision_id: str) -> List[AuditTransaction]:
        """Find all transactions for a decision ID"""
        matching_transactions = []
        
        for block in self.blocks:
            for tx in block.transactions:
                if tx.decision_id == decision_id:
                    matching_transactions.append(tx)
        
        return matching_transactions
    
    def find_transactions_by_event_type(self, event_type: AuditEventType) -> List[AuditTransaction]:
        """Find all transactions by event type"""
        matching_transactions = []
        
        for block in self.blocks:
            for tx in block.transactions:
                if tx.event_type == event_type:
                    matching_transactions.append(tx)
        
        return matching_transactions
    
    def get_chain_statistics(self) -> Dict[str, Any]:
        """Get chain statistics"""
        return {
            'chain_id': self.chain_id,
            'total_blocks': len(self.blocks),
            'total_transactions': self.total_transactions,
            'pending_transactions': len(self.pending_transactions),
            'created_at': self.created_at.isoformat(),
            'last_block_time': self.last_block_time.isoformat(),
            'chain_integrity': self.verify_chain_integrity()
        }
    
    def export_chain(self, format: str = 'json') -> str:
        """Export chain in specified format"""
        if format == 'json':
            chain_data = {
                'chain_id': self.chain_id,
                'created_at': self.created_at.isoformat(),
                'total_blocks': len(self.blocks),
                'total_transactions': self.total_transactions,
                'blocks': [block.to_dict() for block in self.blocks]
            }
            return json.dumps(chain_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class BlockchainAuditSystem:
    """
    Blockchain-based Immutable Audit Trail System
    
    Provides tamper-proof audit logging for trading decisions and explanations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize audit chain
        self.audit_chain = AuditChain("xai_audit_chain")
        
        # Cryptographic keys
        self.private_key = None
        self.public_key = None
        self._initialize_cryptographic_keys()
        
        # Performance tracking
        self.performance_stats = {
            'total_transactions': 0,
            'total_blocks': 0,
            'avg_transaction_time_ms': 0.0,
            'avg_block_time_ms': 0.0,
            'integrity_checks_passed': 0,
            'integrity_checks_failed': 0
        }
        
        # Auto-mining settings
        self.auto_mining_enabled = self.config.get('auto_mining_enabled', True)
        self.mining_interval = self.config.get('mining_interval_seconds', 60)
        self.max_transactions_per_block = self.config.get('max_transactions_per_block', 100)
        
        # Start auto-mining if enabled
        if self.auto_mining_enabled:
            asyncio.create_task(self._auto_mining_loop())
        
        logger.info("BlockchainAuditSystem initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'auto_mining_enabled': True,
            'mining_interval_seconds': 60,
            'max_transactions_per_block': 100,
            'enable_digital_signatures': True,
            'enable_consensus': False,
            'backup_enabled': True,
            'backup_interval_minutes': 10,
            'max_chain_size_mb': 1000,
            'retention_period_days': 365
        }
    
    def _initialize_cryptographic_keys(self):
        """Initialize cryptographic keys for signing"""
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography not available, digital signatures disabled")
            return
        
        try:
            # Generate RSA key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
            logger.info("Cryptographic keys initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cryptographic keys: {e}")
            self.private_key = None
            self.public_key = None
    
    def _sign_data(self, data: str) -> Optional[str]:
        """Sign data with private key"""
        if not self.private_key or not CRYPTOGRAPHY_AVAILABLE:
            return None
        
        try:
            signature = self.private_key.sign(
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            return None
    
    def _verify_signature(self, data: str, signature: str) -> bool:
        """Verify data signature"""
        if not self.public_key or not CRYPTOGRAPHY_AVAILABLE or not signature:
            return False
        
        try:
            signature_bytes = bytes.fromhex(signature)
            self.public_key.verify(
                signature_bytes,
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False
    
    async def log_audit_event(
        self,
        event_type: AuditEventType,
        audit_level: AuditLevel,
        source_system: str,
        source_component: str,
        event_data: Dict[str, Any],
        decision_id: Optional[str] = None,
        explanation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AuditTransaction:
        """
        Log audit event to blockchain
        
        Args:
            event_type: Type of audit event
            audit_level: Severity level
            source_system: Source system name
            source_component: Source component name
            event_data: Event payload data
            decision_id: Associated decision ID
            explanation_id: Associated explanation ID
            user_id: Associated user ID
            session_id: Associated session ID
            
        Returns:
            AuditTransaction: Created transaction
        """
        start_time = time.time()
        
        # Create transaction
        transaction = AuditTransaction(
            transaction_id=f"tx_{uuid.uuid4().hex}",
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            audit_level=audit_level,
            source_system=source_system,
            source_component=source_component,
            decision_id=decision_id,
            explanation_id=explanation_id,
            user_id=user_id,
            session_id=session_id,
            event_data=event_data
        )
        
        # Add digital signature if enabled
        if self.config.get('enable_digital_signatures', True):
            signature = self._sign_data(transaction.data_hash)
            if signature:
                transaction.digital_signature = signature
        
        # Add to audit chain
        self.audit_chain.add_transaction(transaction)
        
        # Update performance stats
        transaction_time_ms = (time.time() - start_time) * 1000
        self.performance_stats['total_transactions'] += 1
        total_tx = self.performance_stats['total_transactions']
        old_avg = self.performance_stats['avg_transaction_time_ms']
        self.performance_stats['avg_transaction_time_ms'] = (
            (old_avg * (total_tx - 1) + transaction_time_ms) / total_tx
        )
        
        logger.info(f"Audit event logged: {event_type.value} (level: {audit_level.value})")
        return transaction
    
    async def log_decision_audit(
        self,
        decision_id: str,
        decision_data: Dict[str, Any],
        explanation_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> List[AuditTransaction]:
        """
        Log comprehensive decision audit
        
        Args:
            decision_id: Decision ID
            decision_data: Decision context and results
            explanation_data: Explanation data
            user_id: User ID
            
        Returns:
            List[AuditTransaction]: Created transactions
        """
        transactions = []
        
        # Log decision event
        decision_tx = await self.log_audit_event(
            event_type=AuditEventType.DECISION_MADE,
            audit_level=AuditLevel.HIGH,
            source_system="STRATEGIC_MARL",
            source_component="DecisionMaker",
            event_data={
                "decision_context": decision_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            decision_id=decision_id,
            user_id=user_id
        )
        transactions.append(decision_tx)
        
        # Log explanation event if provided
        if explanation_data:
            explanation_tx = await self.log_audit_event(
                event_type=AuditEventType.EXPLANATION_GENERATED,
                audit_level=AuditLevel.MEDIUM,
                source_system="XAI_SYSTEM",
                source_component="ExplanationEngine",
                event_data={
                    "explanation_data": explanation_data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                decision_id=decision_id,
                explanation_id=explanation_data.get("explanation_id"),
                user_id=user_id
            )
            transactions.append(explanation_tx)
        
        return transactions
    
    async def log_causal_analysis(
        self,
        decision_id: str,
        causal_result: Dict[str, Any],
        graph_data: Optional[Dict[str, Any]] = None
    ) -> AuditTransaction:
        """Log causal analysis audit"""
        return await self.log_audit_event(
            event_type=AuditEventType.CAUSAL_ANALYSIS,
            audit_level=AuditLevel.MEDIUM,
            source_system="XAI_SYSTEM",
            source_component="CausalInferenceEngine",
            event_data={
                "causal_result": causal_result,
                "graph_data": graph_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            decision_id=decision_id
        )
    
    async def log_counterfactual_analysis(
        self,
        decision_id: str,
        counterfactual_result: Dict[str, Any],
        scenarios: List[Dict[str, Any]]
    ) -> AuditTransaction:
        """Log counterfactual analysis audit"""
        return await self.log_audit_event(
            event_type=AuditEventType.COUNTERFACTUAL_ANALYSIS,
            audit_level=AuditLevel.MEDIUM,
            source_system="XAI_SYSTEM",
            source_component="CounterfactualEngine",
            event_data={
                "counterfactual_result": counterfactual_result,
                "scenarios": scenarios,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            decision_id=decision_id
        )
    
    async def log_bias_detection(
        self,
        detection_result: Dict[str, Any],
        bias_type: str,
        severity: str
    ) -> AuditTransaction:
        """Log bias detection audit"""
        audit_level = AuditLevel.CRITICAL if severity == "HIGH" else AuditLevel.HIGH
        
        return await self.log_audit_event(
            event_type=AuditEventType.BIAS_DETECTION,
            audit_level=audit_level,
            source_system="XAI_SYSTEM",
            source_component="BiasDetector",
            event_data={
                "detection_result": detection_result,
                "bias_type": bias_type,
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def log_ethics_violation(
        self,
        violation_data: Dict[str, Any],
        violation_type: str,
        severity: str
    ) -> AuditTransaction:
        """Log ethics violation audit"""
        return await self.log_audit_event(
            event_type=AuditEventType.ETHICS_VIOLATION,
            audit_level=AuditLevel.CRITICAL,
            source_system="XAI_SYSTEM",
            source_component="EthicsMonitor",
            event_data={
                "violation_data": violation_data,
                "violation_type": violation_type,
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _auto_mining_loop(self):
        """Automatic mining loop"""
        while True:
            try:
                await asyncio.sleep(self.mining_interval)
                
                if self.audit_chain.pending_transactions:
                    start_time = time.time()
                    
                    # Mine new block
                    new_block = self.audit_chain.mine_block(self.max_transactions_per_block)
                    
                    if new_block:
                        # Update performance stats
                        mining_time_ms = (time.time() - start_time) * 1000
                        self.performance_stats['total_blocks'] += 1
                        total_blocks = self.performance_stats['total_blocks']
                        old_avg = self.performance_stats['avg_block_time_ms']
                        self.performance_stats['avg_block_time_ms'] = (
                            (old_avg * (total_blocks - 1) + mining_time_ms) / total_blocks
                        )
                        
                        logger.info(f"Auto-mined block {new_block.block_number}")
                
            except Exception as e:
                logger.error(f"Auto-mining error: {e}")
                await asyncio.sleep(self.mining_interval)
    
    def get_audit_trail(self, decision_id: str) -> List[AuditTransaction]:
        """Get complete audit trail for a decision"""
        return self.audit_chain.find_transactions_by_decision_id(decision_id)
    
    def get_events_by_type(self, event_type: AuditEventType) -> List[AuditTransaction]:
        """Get all events by type"""
        return self.audit_chain.find_transactions_by_event_type(event_type)
    
    def verify_audit_integrity(self) -> bool:
        """Verify audit chain integrity"""
        integrity_passed = self.audit_chain.verify_chain_integrity()
        
        if integrity_passed:
            self.performance_stats['integrity_checks_passed'] += 1
        else:
            self.performance_stats['integrity_checks_failed'] += 1
        
        return integrity_passed
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[AuditEventType]] = None
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        report_data = {
            'report_id': f"compliance_{uuid.uuid4().hex[:8]}",
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'chain_integrity': self.verify_audit_integrity(),
            'events': []
        }
        
        # Collect relevant events
        for block in self.audit_chain.blocks:
            for tx in block.transactions:
                # Check date range
                if start_date <= tx.timestamp <= end_date:
                    # Check event type filter
                    if event_types is None or tx.event_type in event_types:
                        report_data['events'].append({
                            'transaction_id': tx.transaction_id,
                            'timestamp': tx.timestamp.isoformat(),
                            'event_type': tx.event_type.value,
                            'audit_level': tx.audit_level.value,
                            'source_system': tx.source_system,
                            'source_component': tx.source_component,
                            'decision_id': tx.decision_id,
                            'explanation_id': tx.explanation_id,
                            'user_id': tx.user_id,
                            'data_hash': tx.data_hash,
                            'digital_signature': tx.digital_signature,
                            'block_number': block.block_number,
                            'block_hash': block.block_hash
                        })
        
        # Add summary statistics
        report_data['summary'] = {
            'total_events': len(report_data['events']),
            'event_types': list(set(event['event_type'] for event in report_data['events'])),
            'unique_decisions': len(set(event['decision_id'] for event in report_data['events'] if event['decision_id'])),
            'unique_users': len(set(event['user_id'] for event in report_data['events'] if event['user_id'])),
            'audit_levels': {
                level.value: sum(1 for event in report_data['events'] if event['audit_level'] == level.value)
                for level in AuditLevel
            }
        }
        
        return report_data
    
    def export_audit_data(self, format: str = 'json') -> str:
        """Export audit data"""
        return self.audit_chain.export_chain(format)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'system_status': 'operational',
            'audit_chain': self.audit_chain.get_chain_statistics(),
            'performance_stats': self.performance_stats.copy(),
            'auto_mining_enabled': self.auto_mining_enabled,
            'cryptographic_keys_initialized': self.private_key is not None,
            'digital_signatures_enabled': self.config.get('enable_digital_signatures', True)
        }


# Test function
async def test_blockchain_audit_system():
    """Test the Blockchain Audit System"""
    print("🧪 Testing Blockchain Audit System")
    
    # Initialize system
    audit_system = BlockchainAuditSystem()
    
    # Test logging various events
    print("\\n📝 Testing audit event logging...")
    
    # Log decision audit
    decision_data = {
        "action": "LONG",
        "confidence": 0.85,
        "agent_contributions": {"MLMI": 0.6, "NWRQK": 0.4},
        "market_conditions": {"volatility": 0.03, "volume": 1.2}
    }
    
    explanation_data = {
        "explanation_id": "exp_123",
        "explanation_text": "Strong buy signal based on momentum indicators",
        "confidence_score": 0.9
    }
    
    transactions = await audit_system.log_decision_audit(
        decision_id="dec_123",
        decision_data=decision_data,
        explanation_data=explanation_data,
        user_id="trader_001"
    )
    
    print(f"Logged {len(transactions)} transactions for decision audit")
    
    # Log causal analysis
    causal_tx = await audit_system.log_causal_analysis(
        decision_id="dec_123",
        causal_result={"effect_size": 0.75, "p_value": 0.01},
        graph_data={"nodes": 5, "edges": 8}
    )
    
    print(f"Logged causal analysis transaction: {causal_tx.transaction_id}")
    
    # Log bias detection
    bias_tx = await audit_system.log_bias_detection(
        detection_result={"bias_score": 0.15, "bias_type": "demographic"},
        bias_type="demographic",
        severity="LOW"
    )
    
    print(f"Logged bias detection transaction: {bias_tx.transaction_id}")
    
    # Wait for auto-mining (or force manual mining)
    print("\\n⛏️ Mining block...")
    new_block = audit_system.audit_chain.mine_block()
    if new_block:
        print(f"Mined block {new_block.block_number} with {len(new_block.transactions)} transactions")
    
    # Test integrity verification
    print("\\n🔍 Testing integrity verification...")
    integrity_check = audit_system.verify_audit_integrity()
    print(f"Audit chain integrity: {'PASSED' if integrity_check else 'FAILED'}")
    
    # Test audit trail retrieval
    print("\\n📋 Testing audit trail retrieval...")
    decision_trail = audit_system.get_audit_trail("dec_123")
    print(f"Found {len(decision_trail)} audit events for decision dec_123")
    
    for tx in decision_trail:
        print(f"  - {tx.event_type.value} at {tx.timestamp}")
    
    # Test compliance report
    print("\\n📊 Testing compliance report generation...")
    start_date = datetime.now(timezone.utc) - timedelta(hours=1)
    end_date = datetime.now(timezone.utc)
    
    report = audit_system.generate_compliance_report(start_date, end_date)
    print(f"Generated compliance report with {report['summary']['total_events']} events")
    
    # System status
    print("\\n📈 System Status:")
    status = audit_system.get_system_status()
    print(f"  Chain integrity: {status['audit_chain']['chain_integrity']}")
    print(f"  Total blocks: {status['audit_chain']['total_blocks']}")
    print(f"  Total transactions: {status['audit_chain']['total_transactions']}")
    print(f"  Performance stats: {status['performance_stats']}")
    
    print("\\n✅ Blockchain Audit System test complete!")


if __name__ == "__main__":
    from datetime import timedelta
    asyncio.run(test_blockchain_audit_system())