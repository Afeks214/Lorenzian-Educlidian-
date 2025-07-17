"""
Data lifecycle management and archival system

This module implements intelligent data archival and lifecycle management
for high-frequency market data with automated retention policies.
"""

import time
import threading
import schedule
import shutil
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import tarfile
import gzip
from collections import defaultdict, deque

from .optimized_storage import CompressedDataStore, CompressionType

logger = logging.getLogger(__name__)

class DataTier(Enum):
    """Data storage tiers based on access frequency"""
    HOT = "hot"        # Frequently accessed, high-performance storage
    WARM = "warm"      # Occasionally accessed, balanced storage
    COLD = "cold"      # Rarely accessed, archive storage
    FROZEN = "frozen"  # Long-term archive, compressed storage

class LifecycleAction(Enum):
    """Lifecycle actions that can be taken on data"""
    RETAIN = "retain"
    COMPRESS = "compress"
    ARCHIVE = "archive"
    DELETE = "delete"
    REPLICATE = "replicate"

@dataclass
class DataClassification:
    """Data classification for lifecycle management"""
    importance: str  # critical, high, medium, low
    access_pattern: str  # real_time, batch, historical, archive
    retention_period: int  # days
    compliance_requirements: List[str]
    business_value: float  # 0-1 scale

@dataclass
class LifecycleRule:
    """Lifecycle rule definition"""
    name: str
    conditions: Dict[str, Any]
    action: LifecycleAction
    target_tier: Optional[DataTier] = None
    compression_type: Optional[CompressionType] = None
    enabled: bool = True
    priority: int = 0
    created_at: float = field(default_factory=time.time)

@dataclass
class DataAsset:
    """Data asset tracking"""
    key: str
    path: str
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    tier: DataTier
    classification: DataClassification
    metadata: Dict[str, Any]
    checksum: str
    lifecycle_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LifecycleMetrics:
    """Lifecycle management metrics"""
    total_assets: int = 0
    assets_by_tier: Dict[str, int] = field(default_factory=dict)
    total_storage_mb: float = 0.0
    storage_by_tier: Dict[str, float] = field(default_factory=dict)
    archived_assets: int = 0
    deleted_assets: int = 0
    compression_savings_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

class DataLifecycleManager:
    """Intelligent data lifecycle management system"""
    
    def __init__(self, 
                 storage_path: str,
                 hot_storage_path: str,
                 warm_storage_path: str,
                 cold_storage_path: str,
                 frozen_storage_path: str,
                 max_workers: int = 4,
                 enable_auto_lifecycle: bool = True):
        
        self.storage_path = Path(storage_path)
        self.hot_storage_path = Path(hot_storage_path)
        self.warm_storage_path = Path(warm_storage_path)
        self.cold_storage_path = Path(cold_storage_path)
        self.frozen_storage_path = Path(frozen_storage_path)
        
        # Create storage directories
        for path in [self.storage_path, self.hot_storage_path, self.warm_storage_path, 
                    self.cold_storage_path, self.frozen_storage_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.enable_auto_lifecycle = enable_auto_lifecycle
        
        # Initialize database for metadata
        self.db_path = self.storage_path / 'lifecycle.db'
        self._init_database()
        
        # Lifecycle rules
        self.rules = []
        self.rules_lock = threading.RLock()
        
        # Asset tracking
        self.assets = {}
        self.assets_lock = threading.RLock()
        
        # Performance metrics
        self.metrics = LifecycleMetrics()
        self.metrics_lock = threading.Lock()
        
        # Thread pool for lifecycle operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Storage managers for each tier
        self.storage_managers = {
            DataTier.HOT: CompressedDataStore(
                str(self.hot_storage_path),
                compression_type=CompressionType.NONE,  # No compression for hot data
                block_size=64 * 1024  # 64KB blocks
            ),
            DataTier.WARM: CompressedDataStore(
                str(self.warm_storage_path),
                compression_type=CompressionType.LZ4,
                compression_level=1
            ),
            DataTier.COLD: CompressedDataStore(
                str(self.cold_storage_path),
                compression_type=CompressionType.ZSTD,
                compression_level=3
            ),
            DataTier.FROZEN: CompressedDataStore(
                str(self.frozen_storage_path),
                compression_type=CompressionType.ZSTD,
                compression_level=19  # Maximum compression
            )
        }
        
        # Load existing assets
        self._load_assets()
        
        # Setup default lifecycle rules
        self._setup_default_rules()
        
        # Start automatic lifecycle management
        if enable_auto_lifecycle:
            self._start_lifecycle_scheduler()
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("DataLifecycleManager cleanup completed")
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS assets (
                    key TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    tier TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    checksum TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lifecycle_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    source_tier TEXT,
                    target_tier TEXT,
                    timestamp REAL NOT NULL,
                    details TEXT,
                    FOREIGN KEY (asset_key) REFERENCES assets (key)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lifecycle_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_tier TEXT,
                    compression_type TEXT,
                    enabled BOOLEAN DEFAULT TRUE,
                    priority INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
            ''')
            
            conn.commit()
    
    def _load_assets(self):
        """Load existing assets from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM assets')
            for row in cursor.fetchall():
                key, path, size_bytes, created_at, last_accessed, access_count, tier, classification, metadata, checksum = row
                
                # Parse stored data
                tier_enum = DataTier(tier)
                classification_dict = json.loads(classification)
                metadata_dict = json.loads(metadata)
                
                # Load lifecycle history
                history_cursor = conn.execute(
                    'SELECT action, source_tier, target_tier, timestamp, details FROM lifecycle_history WHERE asset_key = ?',
                    (key,)
                )
                history = []
                for hist_row in history_cursor.fetchall():
                    history.append({
                        'action': hist_row[0],
                        'source_tier': hist_row[1],
                        'target_tier': hist_row[2],
                        'timestamp': hist_row[3],
                        'details': hist_row[4]
                    })
                
                # Create asset
                asset = DataAsset(
                    key=key,
                    path=path,
                    size_bytes=size_bytes,
                    created_at=created_at,
                    last_accessed=last_accessed,
                    access_count=access_count,
                    tier=tier_enum,
                    classification=DataClassification(**classification_dict),
                    metadata=metadata_dict,
                    checksum=checksum,
                    lifecycle_history=history
                )
                
                self.assets[key] = asset
        
        logger.info(f"Loaded {len(self.assets)} assets from database")
    
    def _setup_default_rules(self):
        """Setup default lifecycle rules"""
        # Rule 1: Move hot data to warm after 1 day
        self.add_rule(LifecycleRule(
            name="hot_to_warm",
            conditions={
                "tier": DataTier.HOT,
                "age_days": 1,
                "access_count_threshold": 100
            },
            action=LifecycleAction.ARCHIVE,
            target_tier=DataTier.WARM,
            compression_type=CompressionType.LZ4,
            priority=1
        ))
        
        # Rule 2: Move warm data to cold after 7 days
        self.add_rule(LifecycleRule(
            name="warm_to_cold",
            conditions={
                "tier": DataTier.WARM,
                "age_days": 7,
                "last_accessed_days": 3
            },
            action=LifecycleAction.ARCHIVE,
            target_tier=DataTier.COLD,
            compression_type=CompressionType.ZSTD,
            priority=2
        ))
        
        # Rule 3: Move cold data to frozen after 30 days
        self.add_rule(LifecycleRule(
            name="cold_to_frozen",
            conditions={
                "tier": DataTier.COLD,
                "age_days": 30,
                "last_accessed_days": 14
            },
            action=LifecycleAction.ARCHIVE,
            target_tier=DataTier.FROZEN,
            compression_type=CompressionType.ZSTD,
            priority=3
        ))
        
        # Rule 4: Delete low-value data after 90 days
        self.add_rule(LifecycleRule(
            name="delete_low_value",
            conditions={
                "tier": DataTier.FROZEN,
                "age_days": 90,
                "business_value_threshold": 0.3
            },
            action=LifecycleAction.DELETE,
            priority=4
        ))
    
    def add_rule(self, rule: LifecycleRule):
        """Add lifecycle rule"""
        with self.rules_lock:
            self.rules.append(rule)
            self.rules.sort(key=lambda r: r.priority)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO lifecycle_rules (name, conditions, action, target_tier, compression_type, enabled, priority, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (rule.name, json.dumps(rule.conditions), rule.action.value, 
                 rule.target_tier.value if rule.target_tier else None,
                 rule.compression_type.value if rule.compression_type else None,
                 rule.enabled, rule.priority, rule.created_at)
            )
            conn.commit()
    
    def store_data(self, key: str, data: Any, 
                  tier: DataTier = DataTier.HOT,
                  classification: Optional[DataClassification] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data with lifecycle management"""
        # Default classification
        if classification is None:
            classification = DataClassification(
                importance="medium",
                access_pattern="real_time",
                retention_period=30,
                compliance_requirements=[],
                business_value=0.5
            )
        
        # Store data in appropriate tier
        storage_manager = self.storage_managers[tier]
        block_id = storage_manager.store_data(key, data, metadata)
        
        # Calculate checksum
        if isinstance(data, bytes):
            checksum = hashlib.sha256(data).hexdigest()
        else:
            checksum = hashlib.sha256(str(data).encode()).hexdigest()
        
        # Create asset record
        asset = DataAsset(
            key=key,
            path=block_id,
            size_bytes=len(str(data).encode()) if not isinstance(data, bytes) else len(data),
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            tier=tier,
            classification=classification,
            metadata=metadata or {},
            checksum=checksum
        )
        
        # Store asset
        with self.assets_lock:
            self.assets[key] = asset
        
        # Save to database
        self._save_asset_to_db(asset)
        
        # Update metrics
        self._update_metrics()
        
        return block_id
    
    def retrieve_data(self, key: str) -> Tuple[Any, Dict[str, Any]]:
        """Retrieve data and update access metrics"""
        if key not in self.assets:
            raise KeyError(f"Asset not found: {key}")
        
        asset = self.assets[key]
        
        # Retrieve from appropriate storage tier
        storage_manager = self.storage_managers[asset.tier]
        data, metadata = storage_manager.retrieve_data(key)
        
        # Update access metrics
        with self.assets_lock:
            asset.last_accessed = time.time()
            asset.access_count += 1
        
        # Update in database
        self._update_asset_access(key, asset.last_accessed, asset.access_count)
        
        return data, metadata
    
    def _save_asset_to_db(self, asset: DataAsset):
        """Save asset to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO assets (key, path, size_bytes, created_at, last_accessed, access_count, tier, classification, metadata, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (asset.key, asset.path, asset.size_bytes, asset.created_at, asset.last_accessed, 
                 asset.access_count, asset.tier.value, json.dumps(asset.classification.__dict__), 
                 json.dumps(asset.metadata), asset.checksum)
            )
            conn.commit()
    
    def _update_asset_access(self, key: str, last_accessed: float, access_count: int):
        """Update asset access metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE assets SET last_accessed = ?, access_count = ? WHERE key = ?',
                (last_accessed, access_count, key)
            )
            conn.commit()
    
    def _start_lifecycle_scheduler(self):
        """Start automatic lifecycle management"""
        # Schedule lifecycle evaluation every hour
        schedule.every().hour.do(self._evaluate_lifecycle_rules)
        
        # Schedule cleanup every day
        schedule.every().day.do(self._cleanup_storage)
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Lifecycle scheduler started")
    
    def _run_scheduler(self):
        """Run scheduled tasks"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _evaluate_lifecycle_rules(self):
        """Evaluate lifecycle rules for all assets"""
        logger.info("Evaluating lifecycle rules")
        
        with self.assets_lock:
            assets_to_process = list(self.assets.values())
        
        for asset in assets_to_process:
            try:
                self._apply_lifecycle_rules(asset)
            except Exception as e:
                logger.error(f"Error applying lifecycle rules to asset {asset.key}: {str(e)}")
    
    def _apply_lifecycle_rules(self, asset: DataAsset):
        """Apply lifecycle rules to a single asset"""
        current_time = time.time()
        
        with self.rules_lock:
            applicable_rules = [rule for rule in self.rules if rule.enabled]
        
        for rule in applicable_rules:
            if self._evaluate_rule_conditions(asset, rule, current_time):
                try:
                    self._execute_lifecycle_action(asset, rule)
                    break  # Only apply the first matching rule
                except Exception as e:
                    logger.error(f"Error executing lifecycle action for asset {asset.key}: {str(e)}")
    
    def _evaluate_rule_conditions(self, asset: DataAsset, rule: LifecycleRule, current_time: float) -> bool:
        """Evaluate if rule conditions are met"""
        conditions = rule.conditions
        
        # Check tier condition
        if 'tier' in conditions and asset.tier != conditions['tier']:
            return False
        
        # Check age condition
        if 'age_days' in conditions:
            age_days = (current_time - asset.created_at) / (24 * 3600)
            if age_days < conditions['age_days']:
                return False
        
        # Check last accessed condition
        if 'last_accessed_days' in conditions:
            last_accessed_days = (current_time - asset.last_accessed) / (24 * 3600)
            if last_accessed_days < conditions['last_accessed_days']:
                return False
        
        # Check access count condition
        if 'access_count_threshold' in conditions:
            if asset.access_count < conditions['access_count_threshold']:
                return False
        
        # Check business value condition
        if 'business_value_threshold' in conditions:
            if asset.classification.business_value > conditions['business_value_threshold']:
                return False
        
        # Check importance condition
        if 'importance' in conditions:
            if asset.classification.importance != conditions['importance']:
                return False
        
        return True
    
    def _execute_lifecycle_action(self, asset: DataAsset, rule: LifecycleRule):
        """Execute lifecycle action on asset"""
        if rule.action == LifecycleAction.ARCHIVE and rule.target_tier:
            self._archive_asset(asset, rule.target_tier, rule.compression_type)
        elif rule.action == LifecycleAction.DELETE:
            self._delete_asset(asset)
        elif rule.action == LifecycleAction.COMPRESS:
            self._compress_asset(asset, rule.compression_type)
        elif rule.action == LifecycleAction.REPLICATE:
            self._replicate_asset(asset)
        
        # Record lifecycle action
        self._record_lifecycle_action(asset, rule)
    
    def _archive_asset(self, asset: DataAsset, target_tier: DataTier, compression_type: Optional[CompressionType]):
        """Archive asset to different tier"""
        if asset.tier == target_tier:
            return
        
        # Retrieve data from current tier
        current_storage = self.storage_managers[asset.tier]
        data, metadata = current_storage.retrieve_data(asset.key)
        
        # Store in target tier
        target_storage = self.storage_managers[target_tier]
        new_block_id = target_storage.store_data(asset.key, data, metadata)
        
        # Update asset record
        old_tier = asset.tier
        asset.tier = target_tier
        asset.path = new_block_id
        
        # Save to database
        self._save_asset_to_db(asset)
        
        # Remove from old tier
        current_storage.delete_key(asset.key)
        
        logger.info(f"Archived asset {asset.key} from {old_tier.value} to {target_tier.value}")
    
    def _delete_asset(self, asset: DataAsset):
        """Delete asset permanently"""
        # Remove from storage
        storage_manager = self.storage_managers[asset.tier]
        storage_manager.delete_key(asset.key)
        
        # Remove from tracking
        with self.assets_lock:
            del self.assets[asset.key]
        
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM assets WHERE key = ?', (asset.key,))
            conn.execute('DELETE FROM lifecycle_history WHERE asset_key = ?', (asset.key,))
            conn.commit()
        
        logger.info(f"Deleted asset {asset.key}")
    
    def _compress_asset(self, asset: DataAsset, compression_type: Optional[CompressionType]):
        """Compress asset with higher compression"""
        # This would involve recompressing the data with a higher compression ratio
        # Implementation depends on storage backend
        pass
    
    def _replicate_asset(self, asset: DataAsset):
        """Replicate asset to backup storage"""
        # This would involve copying the asset to a backup location
        # Implementation depends on backup strategy
        pass
    
    def _record_lifecycle_action(self, asset: DataAsset, rule: LifecycleRule):
        """Record lifecycle action in history"""
        history_entry = {
            'action': rule.action.value,
            'rule_name': rule.name,
            'timestamp': time.time()
        }
        
        asset.lifecycle_history.append(history_entry)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO lifecycle_history (asset_key, action, source_tier, target_tier, timestamp, details) VALUES (?, ?, ?, ?, ?, ?)',
                (asset.key, rule.action.value, asset.tier.value, 
                 rule.target_tier.value if rule.target_tier else None,
                 time.time(), json.dumps(history_entry))
            )
            conn.commit()
    
    def _cleanup_storage(self):
        """Cleanup storage and optimize"""
        logger.info("Starting storage cleanup")
        
        # Clean up each storage tier
        for tier, storage_manager in self.storage_managers.items():
            try:
                storage_manager.cleanup_old_data(max_age_hours=24)
            except Exception as e:
                logger.error(f"Error cleaning up {tier.value} storage: {str(e)}")
        
        # Update metrics
        self._update_metrics()
        
        logger.info("Storage cleanup completed")
    
    def _update_metrics(self):
        """Update lifecycle metrics"""
        with self.metrics_lock:
            self.metrics.total_assets = len(self.assets)
            self.metrics.assets_by_tier = defaultdict(int)
            self.metrics.storage_by_tier = defaultdict(float)
            
            total_storage = 0
            for asset in self.assets.values():
                tier_name = asset.tier.value
                self.metrics.assets_by_tier[tier_name] += 1
                
                size_mb = asset.size_bytes / (1024 * 1024)
                self.metrics.storage_by_tier[tier_name] += size_mb
                total_storage += size_mb
            
            self.metrics.total_storage_mb = total_storage
            self.metrics.timestamp = time.time()
    
    def get_metrics(self) -> LifecycleMetrics:
        """Get current lifecycle metrics"""
        with self.metrics_lock:
            return LifecycleMetrics(
                total_assets=self.metrics.total_assets,
                assets_by_tier=dict(self.metrics.assets_by_tier),
                total_storage_mb=self.metrics.total_storage_mb,
                storage_by_tier=dict(self.metrics.storage_by_tier),
                archived_assets=self.metrics.archived_assets,
                deleted_assets=self.metrics.deleted_assets,
                compression_savings_mb=self.metrics.compression_savings_mb,
                timestamp=time.time()
            )
    
    def get_asset_info(self, key: str) -> Optional[DataAsset]:
        """Get asset information"""
        return self.assets.get(key)
    
    def list_assets(self, tier: Optional[DataTier] = None) -> List[DataAsset]:
        """List assets, optionally filtered by tier"""
        with self.assets_lock:
            if tier is None:
                return list(self.assets.values())
            else:
                return [asset for asset in self.assets.values() if asset.tier == tier]
    
    def force_lifecycle_evaluation(self, key: Optional[str] = None):
        """Force lifecycle evaluation for specific asset or all assets"""
        if key:
            if key in self.assets:
                self._apply_lifecycle_rules(self.assets[key])
        else:
            self._evaluate_lifecycle_rules()
    
    def get_storage_utilization(self) -> Dict[str, Dict[str, float]]:
        """Get storage utilization by tier"""
        utilization = {}
        
        for tier, storage_manager in self.storage_managers.items():
            metrics = storage_manager.get_metrics()
            utilization[tier.value] = {
                'total_mb': metrics.storage_utilization_mb,
                'compression_ratio': metrics.compression_ratio,
                'read_latency_us': metrics.avg_read_latency_us,
                'write_latency_us': metrics.avg_write_latency_us
            }
        
        return utilization

# Utility functions
def create_lifecycle_manager(storage_path: str, enable_auto_lifecycle: bool = True) -> DataLifecycleManager:
    """Create lifecycle manager with default settings"""
    base_path = Path(storage_path)
    
    return DataLifecycleManager(
        storage_path=str(base_path),
        hot_storage_path=str(base_path / "hot"),
        warm_storage_path=str(base_path / "warm"),
        cold_storage_path=str(base_path / "cold"),
        frozen_storage_path=str(base_path / "frozen"),
        enable_auto_lifecycle=enable_auto_lifecycle
    )

def create_classification(importance: str, access_pattern: str, 
                        retention_days: int, business_value: float) -> DataClassification:
    """Create data classification"""
    return DataClassification(
        importance=importance,
        access_pattern=access_pattern,
        retention_period=retention_days,
        compliance_requirements=[],
        business_value=business_value
    )
