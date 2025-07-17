"""
Contract Registry
================

Manages contract definitions, versions, and service relationships.
Provides centralized contract storage and retrieval.
"""

import json
import os
import time
import hashlib
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime


class ContractType(Enum):
    """Contract type enumeration"""
    REQUEST_RESPONSE = "request_response"
    EVENT_DRIVEN = "event_driven"
    STREAMING = "streaming"
    GRAPHQL = "graphql"
    GRPC = "grpc"


class ContractStatus(Enum):
    """Contract status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass
class ContractEndpoint:
    """Contract endpoint definition"""
    path: str
    method: str
    request_schema: Dict[str, Any]
    response_schema: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    status_codes: List[int] = field(default_factory=lambda: [200])
    timeout_ms: int = 5000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractEvent:
    """Contract event definition for event-driven contracts"""
    event_type: str
    event_schema: Dict[str, Any]
    routing_key: str = ""
    exchange: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractDefinition:
    """Contract definition structure"""
    contract_id: str
    version: str
    consumer: str
    provider: str
    contract_type: ContractType
    status: ContractStatus
    endpoints: List[ContractEndpoint] = field(default_factory=list)
    events: List[ContractEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if isinstance(self.contract_type, str):
            self.contract_type = ContractType(self.contract_type)
        if isinstance(self.status, str):
            self.status = ContractStatus(self.status)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)


@dataclass
class ServiceRelationship:
    """Service relationship definition"""
    consumer: str
    provider: str
    contracts: List[str]
    relationship_type: str = "direct"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContractRegistry:
    """
    Contract registry for managing contract definitions and versions
    
    Features:
    - Contract storage and retrieval
    - Version management
    - Service relationship tracking
    - Contract validation
    - Backwards compatibility checking
    """
    
    def __init__(self, storage_path: str = "/tmp/contract_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.contracts_cache: Dict[str, Dict[str, ContractDefinition]] = {}
        self.relationships_cache: Dict[str, List[ServiceRelationship]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load existing contracts
        self._load_contracts()
    
    def _load_contracts(self):
        """Load existing contracts from storage"""
        contracts_file = self.storage_path / "contracts.json"
        relationships_file = self.storage_path / "relationships.json"
        
        # Load contracts
        if contracts_file.exists():
            try:
                with open(contracts_file, 'r') as f:
                    data = json.load(f)
                    
                for contract_id, versions in data.items():
                    self.contracts_cache[contract_id] = {}
                    for version, contract_data in versions.items():
                        contract = ContractDefinition(**contract_data)
                        self.contracts_cache[contract_id][version] = contract
                        
            except Exception as e:
                print(f"Error loading contracts: {e}")
        
        # Load relationships
        if relationships_file.exists():
            try:
                with open(relationships_file, 'r') as f:
                    data = json.load(f)
                    
                for service, relationships in data.items():
                    self.relationships_cache[service] = [
                        ServiceRelationship(**rel_data) for rel_data in relationships
                    ]
                    
            except Exception as e:
                print(f"Error loading relationships: {e}")
    
    def _save_contracts(self):
        """Save contracts to storage"""
        contracts_file = self.storage_path / "contracts.json"
        relationships_file = self.storage_path / "relationships.json"
        
        # Save contracts
        contracts_data = {}
        for contract_id, versions in self.contracts_cache.items():
            contracts_data[contract_id] = {}
            for version, contract in versions.items():
                contracts_data[contract_id][version] = self._serialize_contract(contract)
        
        with open(contracts_file, 'w') as f:
            json.dump(contracts_data, f, indent=2, default=str)
        
        # Save relationships
        relationships_data = {}
        for service, relationships in self.relationships_cache.items():
            relationships_data[service] = [
                asdict(rel) for rel in relationships
            ]
        
        with open(relationships_file, 'w') as f:
            json.dump(relationships_data, f, indent=2, default=str)
    
    def _serialize_contract(self, contract: ContractDefinition) -> Dict[str, Any]:
        """Serialize contract for storage"""
        data = asdict(contract)
        data['contract_type'] = contract.contract_type.value
        data['status'] = contract.status.value
        data['created_at'] = contract.created_at.isoformat()
        data['updated_at'] = contract.updated_at.isoformat()
        return data
    
    def register_contract(self, contract: ContractDefinition) -> bool:
        """Register a new contract"""
        with self.lock:
            try:
                # Validate contract
                self._validate_contract(contract)
                
                # Initialize contract versions if not exists
                if contract.contract_id not in self.contracts_cache:
                    self.contracts_cache[contract.contract_id] = {}
                
                # Check if version already exists
                if contract.version in self.contracts_cache[contract.contract_id]:
                    raise ValueError(f"Contract version {contract.version} already exists")
                
                # Set timestamps
                contract.created_at = datetime.now()
                contract.updated_at = datetime.now()
                
                # Store contract
                self.contracts_cache[contract.contract_id][contract.version] = contract
                
                # Update service relationships
                self._update_service_relationships(contract)
                
                # Save to storage
                self._save_contracts()
                
                return True
                
            except Exception as e:
                print(f"Error registering contract: {e}")
                return False
    
    def _validate_contract(self, contract: ContractDefinition):
        """Validate contract definition"""
        if not contract.contract_id:
            raise ValueError("Contract ID is required")
        
        if not contract.version:
            raise ValueError("Contract version is required")
        
        if not contract.consumer:
            raise ValueError("Consumer is required")
        
        if not contract.provider:
            raise ValueError("Provider is required")
        
        # Validate endpoints
        for endpoint in contract.endpoints:
            if not endpoint.path:
                raise ValueError("Endpoint path is required")
            
            if not endpoint.method:
                raise ValueError("Endpoint method is required")
        
        # Validate events
        for event in contract.events:
            if not event.event_type:
                raise ValueError("Event type is required")
            
            if not event.event_schema:
                raise ValueError("Event schema is required")
    
    def _update_service_relationships(self, contract: ContractDefinition):
        """Update service relationships based on contract"""
        consumer = contract.consumer
        provider = contract.provider
        
        # Update consumer relationships
        if consumer not in self.relationships_cache:
            self.relationships_cache[consumer] = []
        
        # Check if relationship already exists
        existing_rel = None
        for rel in self.relationships_cache[consumer]:
            if rel.provider == provider:
                existing_rel = rel
                break
        
        if existing_rel:
            # Add contract to existing relationship
            if contract.contract_id not in existing_rel.contracts:
                existing_rel.contracts.append(contract.contract_id)
        else:
            # Create new relationship
            new_rel = ServiceRelationship(
                consumer=consumer,
                provider=provider,
                contracts=[contract.contract_id]
            )
            self.relationships_cache[consumer].append(new_rel)
    
    def get_contract(self, contract_id: str, version: str = None) -> Optional[ContractDefinition]:
        """Get contract by ID and version"""
        with self.lock:
            if contract_id not in self.contracts_cache:
                return None
            
            versions = self.contracts_cache[contract_id]
            
            if version:
                return versions.get(version)
            else:
                # Return latest version
                if not versions:
                    return None
                
                latest_version = max(versions.keys())
                return versions[latest_version]
    
    def get_contract_versions(self, contract_id: str) -> List[str]:
        """Get all versions of a contract"""
        with self.lock:
            if contract_id not in self.contracts_cache:
                return []
            
            return list(self.contracts_cache[contract_id].keys())
    
    def get_contracts_by_consumer(self, consumer: str) -> List[ContractDefinition]:
        """Get all contracts for a consumer"""
        contracts = []
        
        with self.lock:
            for contract_id, versions in self.contracts_cache.items():
                for version, contract in versions.items():
                    if contract.consumer == consumer:
                        contracts.append(contract)
        
        return contracts
    
    def get_contracts_by_provider(self, provider: str) -> List[ContractDefinition]:
        """Get all contracts for a provider"""
        contracts = []
        
        with self.lock:
            for contract_id, versions in self.contracts_cache.items():
                for version, contract in versions.items():
                    if contract.provider == provider:
                        contracts.append(contract)
        
        return contracts
    
    def get_service_relationships(self, service: str) -> List[ServiceRelationship]:
        """Get service relationships"""
        with self.lock:
            return self.relationships_cache.get(service, [])
    
    def update_contract_status(self, contract_id: str, version: str, 
                             status: ContractStatus) -> bool:
        """Update contract status"""
        with self.lock:
            try:
                contract = self.get_contract(contract_id, version)
                if not contract:
                    return False
                
                contract.status = status
                contract.updated_at = datetime.now()
                
                self._save_contracts()
                return True
                
            except Exception as e:
                print(f"Error updating contract status: {e}")
                return False
    
    def delete_contract(self, contract_id: str, version: str = None) -> bool:
        """Delete contract or specific version"""
        with self.lock:
            try:
                if contract_id not in self.contracts_cache:
                    return False
                
                if version:
                    # Delete specific version
                    if version in self.contracts_cache[contract_id]:
                        del self.contracts_cache[contract_id][version]
                        
                        # Remove contract ID if no versions left
                        if not self.contracts_cache[contract_id]:
                            del self.contracts_cache[contract_id]
                else:
                    # Delete all versions
                    del self.contracts_cache[contract_id]
                
                # Update relationships
                self._cleanup_relationships(contract_id)
                
                self._save_contracts()
                return True
                
            except Exception as e:
                print(f"Error deleting contract: {e}")
                return False
    
    def _cleanup_relationships(self, contract_id: str):
        """Clean up relationships after contract deletion"""
        for service, relationships in self.relationships_cache.items():
            for rel in relationships:
                if contract_id in rel.contracts:
                    rel.contracts.remove(contract_id)
            
            # Remove empty relationships
            self.relationships_cache[service] = [
                rel for rel in relationships if rel.contracts
            ]
    
    def search_contracts(self, query: Dict[str, Any]) -> List[ContractDefinition]:
        """Search contracts by various criteria"""
        results = []
        
        with self.lock:
            for contract_id, versions in self.contracts_cache.items():
                for version, contract in versions.items():
                    if self._matches_query(contract, query):
                        results.append(contract)
        
        return results
    
    def _matches_query(self, contract: ContractDefinition, query: Dict[str, Any]) -> bool:
        """Check if contract matches query criteria"""
        
        # Check consumer
        if 'consumer' in query and contract.consumer != query['consumer']:
            return False
        
        # Check provider
        if 'provider' in query and contract.provider != query['provider']:
            return False
        
        # Check contract type
        if 'contract_type' in query and contract.contract_type != query['contract_type']:
            return False
        
        # Check status
        if 'status' in query and contract.status != query['status']:
            return False
        
        # Check tags
        if 'tags' in query:
            required_tags = set(query['tags'])
            contract_tags = set(contract.tags)
            if not required_tags.issubset(contract_tags):
                return False
        
        # Check metadata
        if 'metadata' in query:
            for key, value in query['metadata'].items():
                if key not in contract.metadata or contract.metadata[key] != value:
                    return False
        
        return True
    
    def get_contract_statistics(self) -> Dict[str, Any]:
        """Get contract registry statistics"""
        with self.lock:
            total_contracts = 0
            total_versions = 0
            contracts_by_type = {}
            contracts_by_status = {}
            
            for contract_id, versions in self.contracts_cache.items():
                total_contracts += 1
                total_versions += len(versions)
                
                for version, contract in versions.items():
                    # Count by type
                    contract_type = contract.contract_type.value
                    contracts_by_type[contract_type] = contracts_by_type.get(contract_type, 0) + 1
                    
                    # Count by status
                    status = contract.status.value
                    contracts_by_status[status] = contracts_by_status.get(status, 0) + 1
            
            return {
                'total_contracts': total_contracts,
                'total_versions': total_versions,
                'contracts_by_type': contracts_by_type,
                'contracts_by_status': contracts_by_status,
                'total_relationships': sum(len(rels) for rels in self.relationships_cache.values()),
                'services': {
                    'consumers': len(set(rel.consumer for rels in self.relationships_cache.values() for rel in rels)),
                    'providers': len(set(rel.provider for rels in self.relationships_cache.values() for rel in rels))
                }
            }
    
    def export_contracts(self, format: str = 'json') -> str:
        """Export contracts in specified format"""
        with self.lock:
            if format.lower() == 'json':
                export_data = {
                    'contracts': {},
                    'relationships': {}
                }
                
                # Export contracts
                for contract_id, versions in self.contracts_cache.items():
                    export_data['contracts'][contract_id] = {}
                    for version, contract in versions.items():
                        export_data['contracts'][contract_id][version] = self._serialize_contract(contract)
                
                # Export relationships
                for service, relationships in self.relationships_cache.items():
                    export_data['relationships'][service] = [
                        asdict(rel) for rel in relationships
                    ]
                
                return json.dumps(export_data, indent=2, default=str)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def import_contracts(self, data: str, format: str = 'json') -> bool:
        """Import contracts from specified format"""
        with self.lock:
            try:
                if format.lower() == 'json':
                    import_data = json.loads(data)
                    
                    # Import contracts
                    if 'contracts' in import_data:
                        for contract_id, versions in import_data['contracts'].items():
                            self.contracts_cache[contract_id] = {}
                            for version, contract_data in versions.items():
                                contract = ContractDefinition(**contract_data)
                                self.contracts_cache[contract_id][version] = contract
                    
                    # Import relationships
                    if 'relationships' in import_data:
                        for service, relationships in import_data['relationships'].items():
                            self.relationships_cache[service] = [
                                ServiceRelationship(**rel_data) for rel_data in relationships
                            ]
                    
                    self._save_contracts()
                    return True
                
                else:
                    raise ValueError(f"Unsupported import format: {format}")
                    
            except Exception as e:
                print(f"Error importing contracts: {e}")
                return False