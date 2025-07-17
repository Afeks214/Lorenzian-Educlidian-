"""
Vault Transit Encryption Engine Integration
Provides encryption-as-a-service capabilities using HashiCorp Vault's transit engine.
"""

import os
import asyncio
import time
import base64
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

from src.monitoring.logger_config import get_logger
from src.security.vault_client import VaultClient, VaultException, get_vault_client

logger = get_logger(__name__)

class EncryptionKeyType(str, Enum):
    """Encryption key types supported by Vault Transit"""
    AES256_GCM96 = "aes256-gcm96"
    AES128_GCM96 = "aes128-gcm96"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    ED25519 = "ed25519"
    ECDSA_P256 = "ecdsa-p256"
    ECDSA_P384 = "ecdsa-p384"
    ECDSA_P521 = "ecdsa-p521"
    RSA_2048 = "rsa-2048"
    RSA_3072 = "rsa-3072"
    RSA_4096 = "rsa-4096"

class EncryptionOperation(str, Enum):
    """Encryption operations"""
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    SIGN = "sign"
    VERIFY = "verify"
    HMAC = "hmac"
    REWRAP = "rewrap"

@dataclass
class EncryptionResult:
    """Result of encryption operation"""
    ciphertext: str
    key_version: int
    operation: EncryptionOperation
    context: Optional[str] = None
    nonce: Optional[str] = None
    
@dataclass
class DecryptionResult:
    """Result of decryption operation"""
    plaintext: str
    key_version: int
    operation: EncryptionOperation
    context: Optional[str] = None

@dataclass
class EncryptionKey:
    """Transit encryption key information"""
    name: str
    key_type: EncryptionKeyType
    creation_time: str
    deletion_allowed: bool
    derived: bool
    exportable: bool
    keys: Dict[str, Any]
    min_decryption_version: int
    min_encryption_version: int
    supports_encryption: bool
    supports_decryption: bool
    supports_derivation: bool
    supports_signing: bool

class VaultTransitEncryption:
    """
    Vault Transit Encryption Engine Integration
    
    Provides encryption-as-a-service capabilities including:
    - Symmetric encryption/decryption
    - Key derivation
    - Digital signatures
    - HMAC operations
    - Key rotation and versioning
    """
    
    def __init__(self, vault_client: Optional[VaultClient] = None, mount_point: str = "transit"):
        """
        Initialize Transit encryption
        
        Args:
            vault_client: Vault client instance
            mount_point: Transit engine mount point
        """
        self.vault_client = vault_client
        self.mount_point = mount_point
        self.key_cache: Dict[str, EncryptionKey] = {}
        
        logger.info("Transit encryption initialized", mount_point=mount_point)
    
    async def _get_vault_client(self) -> VaultClient:
        """Get Vault client instance"""
        if self.vault_client:
            return self.vault_client
        return await get_vault_client()
    
    async def create_key(self, 
                        key_name: str, 
                        key_type: EncryptionKeyType = EncryptionKeyType.AES256_GCM96,
                        derived: bool = False,
                        exportable: bool = False,
                        allow_plaintext_backup: bool = False,
                        context: Optional[str] = None) -> EncryptionKey:
        """
        Create new encryption key
        
        Args:
            key_name: Name of the key
            key_type: Type of encryption key
            derived: Whether key supports derivation
            exportable: Whether key can be exported
            allow_plaintext_backup: Whether plaintext backup is allowed
            context: Key derivation context
            
        Returns:
            EncryptionKey information
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Create key
            client.client.secrets.transit.create_key(
                name=key_name,
                key_type=key_type.value,
                derived=derived,
                exportable=exportable,
                allow_plaintext_backup=allow_plaintext_backup,
                context=context,
                mount_point=self.mount_point
            )
            
            # Get key info
            key_info = await self.get_key_info(key_name)
            
            logger.info(f"Encryption key created: {key_name}", key_type=key_type.value)
            
            return key_info
            
        except Exception as e:
            logger.error(f"Failed to create encryption key: {e}", key_name=key_name)
            raise VaultException(f"Failed to create encryption key {key_name}: {e}")
    
    async def get_key_info(self, key_name: str) -> EncryptionKey:
        """
        Get encryption key information
        
        Args:
            key_name: Name of the key
            
        Returns:
            EncryptionKey information
        """
        try:
            # Check cache first
            if key_name in self.key_cache:
                return self.key_cache[key_name]
            
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Get key info
            response = client.client.secrets.transit.read_key(
                name=key_name,
                mount_point=self.mount_point
            )
            
            key_data = response['data']
            
            key_info = EncryptionKey(
                name=key_name,
                key_type=EncryptionKeyType(key_data['type']),
                creation_time=key_data['creation_time'],
                deletion_allowed=key_data['deletion_allowed'],
                derived=key_data['derived'],
                exportable=key_data['exportable'],
                keys=key_data['keys'],
                min_decryption_version=key_data['min_decryption_version'],
                min_encryption_version=key_data['min_encryption_version'],
                supports_encryption=key_data['supports_encryption'],
                supports_decryption=key_data['supports_decryption'],
                supports_derivation=key_data['supports_derivation'],
                supports_signing=key_data['supports_signing']
            )
            
            # Cache the key info
            self.key_cache[key_name] = key_info
            
            return key_info
            
        except Exception as e:
            logger.error(f"Failed to get key info: {e}", key_name=key_name)
            raise VaultException(f"Failed to get key info {key_name}: {e}")
    
    async def encrypt(self, 
                     key_name: str, 
                     plaintext: str,
                     context: Optional[str] = None,
                     key_version: Optional[int] = None,
                     nonce: Optional[str] = None,
                     batch_input: Optional[List[Dict[str, Any]]] = None) -> Union[EncryptionResult, List[EncryptionResult]]:
        """
        Encrypt data using transit key
        
        Args:
            key_name: Name of the encryption key
            plaintext: Data to encrypt
            context: Additional authenticated data
            key_version: Specific key version to use
            nonce: Nonce for encryption
            batch_input: List of items to encrypt in batch
            
        Returns:
            EncryptionResult or list of results for batch
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Encode plaintext
            encoded_plaintext = base64.b64encode(plaintext.encode()).decode()
            
            if batch_input:
                # Batch encryption
                batch_items = []
                for item in batch_input:
                    batch_item = {
                        'plaintext': base64.b64encode(item['plaintext'].encode()).decode()
                    }
                    if 'context' in item:
                        batch_item['context'] = base64.b64encode(item['context'].encode()).decode()
                    if 'key_version' in item:
                        batch_item['key_version'] = item['key_version']
                    if 'nonce' in item:
                        batch_item['nonce'] = item['nonce']
                    batch_items.append(batch_item)
                
                response = client.client.secrets.transit.encrypt_data(
                    name=key_name,
                    batch_input=batch_items,
                    mount_point=self.mount_point
                )
                
                results = []
                for item in response['data']['batch_results']:
                    results.append(EncryptionResult(
                        ciphertext=item['ciphertext'],
                        key_version=item.get('key_version', 1),
                        operation=EncryptionOperation.ENCRYPT,
                        context=context
                    ))
                
                return results
            else:
                # Single encryption
                encrypt_params = {
                    'name': key_name,
                    'plaintext': encoded_plaintext,
                    'mount_point': self.mount_point
                }
                
                if context:
                    encrypt_params['context'] = base64.b64encode(context.encode()).decode()
                if key_version:
                    encrypt_params['key_version'] = key_version
                if nonce:
                    encrypt_params['nonce'] = nonce
                
                response = client.client.secrets.transit.encrypt_data(**encrypt_params)
                
                return EncryptionResult(
                    ciphertext=response['data']['ciphertext'],
                    key_version=response['data'].get('key_version', 1),
                    operation=EncryptionOperation.ENCRYPT,
                    context=context,
                    nonce=nonce
                )
                
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}", key_name=key_name)
            raise VaultException(f"Failed to encrypt data with key {key_name}: {e}")
    
    async def decrypt(self, 
                     key_name: str, 
                     ciphertext: str,
                     context: Optional[str] = None,
                     nonce: Optional[str] = None,
                     batch_input: Optional[List[Dict[str, Any]]] = None) -> Union[DecryptionResult, List[DecryptionResult]]:
        """
        Decrypt data using transit key
        
        Args:
            key_name: Name of the encryption key
            ciphertext: Data to decrypt
            context: Additional authenticated data
            nonce: Nonce used for encryption
            batch_input: List of items to decrypt in batch
            
        Returns:
            DecryptionResult or list of results for batch
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            if batch_input:
                # Batch decryption
                batch_items = []
                for item in batch_input:
                    batch_item = {
                        'ciphertext': item['ciphertext']
                    }
                    if 'context' in item:
                        batch_item['context'] = base64.b64encode(item['context'].encode()).decode()
                    if 'nonce' in item:
                        batch_item['nonce'] = item['nonce']
                    batch_items.append(batch_item)
                
                response = client.client.secrets.transit.decrypt_data(
                    name=key_name,
                    batch_input=batch_items,
                    mount_point=self.mount_point
                )
                
                results = []
                for item in response['data']['batch_results']:
                    plaintext = base64.b64decode(item['plaintext']).decode()
                    results.append(DecryptionResult(
                        plaintext=plaintext,
                        key_version=item.get('key_version', 1),
                        operation=EncryptionOperation.DECRYPT,
                        context=context
                    ))
                
                return results
            else:
                # Single decryption
                decrypt_params = {
                    'name': key_name,
                    'ciphertext': ciphertext,
                    'mount_point': self.mount_point
                }
                
                if context:
                    decrypt_params['context'] = base64.b64encode(context.encode()).decode()
                if nonce:
                    decrypt_params['nonce'] = nonce
                
                response = client.client.secrets.transit.decrypt_data(**decrypt_params)
                
                plaintext = base64.b64decode(response['data']['plaintext']).decode()
                
                return DecryptionResult(
                    plaintext=plaintext,
                    key_version=response['data'].get('key_version', 1),
                    operation=EncryptionOperation.DECRYPT,
                    context=context
                )
                
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}", key_name=key_name)
            raise VaultException(f"Failed to decrypt data with key {key_name}: {e}")
    
    async def rotate_key(self, key_name: str) -> EncryptionKey:
        """
        Rotate encryption key
        
        Args:
            key_name: Name of the key to rotate
            
        Returns:
            Updated EncryptionKey information
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Rotate key
            client.client.secrets.transit.rotate_key(
                name=key_name,
                mount_point=self.mount_point
            )
            
            # Clear cache and get updated key info
            if key_name in self.key_cache:
                del self.key_cache[key_name]
            
            key_info = await self.get_key_info(key_name)
            
            logger.info(f"Encryption key rotated: {key_name}")
            
            return key_info
            
        except Exception as e:
            logger.error(f"Failed to rotate key: {e}", key_name=key_name)
            raise VaultException(f"Failed to rotate key {key_name}: {e}")
    
    async def rewrap(self, 
                    key_name: str, 
                    ciphertext: str,
                    context: Optional[str] = None,
                    key_version: Optional[int] = None,
                    nonce: Optional[str] = None) -> EncryptionResult:
        """
        Rewrap ciphertext with latest key version
        
        Args:
            key_name: Name of the encryption key
            ciphertext: Ciphertext to rewrap
            context: Additional authenticated data
            key_version: Specific key version to use
            nonce: Nonce for encryption
            
        Returns:
            EncryptionResult with rewrapped ciphertext
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            rewrap_params = {
                'name': key_name,
                'ciphertext': ciphertext,
                'mount_point': self.mount_point
            }
            
            if context:
                rewrap_params['context'] = base64.b64encode(context.encode()).decode()
            if key_version:
                rewrap_params['key_version'] = key_version
            if nonce:
                rewrap_params['nonce'] = nonce
            
            response = client.client.secrets.transit.rewrap_data(**rewrap_params)
            
            return EncryptionResult(
                ciphertext=response['data']['ciphertext'],
                key_version=response['data'].get('key_version', 1),
                operation=EncryptionOperation.REWRAP,
                context=context,
                nonce=nonce
            )
            
        except Exception as e:
            logger.error(f"Failed to rewrap data: {e}", key_name=key_name)
            raise VaultException(f"Failed to rewrap data with key {key_name}: {e}")
    
    async def sign(self, 
                  key_name: str, 
                  input_data: str,
                  hash_algorithm: str = "sha2-256",
                  context: Optional[str] = None,
                  prehashed: bool = False) -> str:
        """
        Sign data using transit key
        
        Args:
            key_name: Name of the signing key
            input_data: Data to sign
            hash_algorithm: Hash algorithm to use
            context: Additional authenticated data
            prehashed: Whether input is already hashed
            
        Returns:
            Signature string
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Encode input data
            encoded_input = base64.b64encode(input_data.encode()).decode()
            
            sign_params = {
                'name': key_name,
                'input': encoded_input,
                'hash_algorithm': hash_algorithm,
                'prehashed': prehashed,
                'mount_point': self.mount_point
            }
            
            if context:
                sign_params['context'] = base64.b64encode(context.encode()).decode()
            
            response = client.client.secrets.transit.sign_data(**sign_params)
            
            return response['data']['signature']
            
        except Exception as e:
            logger.error(f"Failed to sign data: {e}", key_name=key_name)
            raise VaultException(f"Failed to sign data with key {key_name}: {e}")
    
    async def verify(self, 
                    key_name: str, 
                    input_data: str,
                    signature: str,
                    hash_algorithm: str = "sha2-256",
                    context: Optional[str] = None,
                    prehashed: bool = False) -> bool:
        """
        Verify signature using transit key
        
        Args:
            key_name: Name of the signing key
            input_data: Original data
            signature: Signature to verify
            hash_algorithm: Hash algorithm used
            context: Additional authenticated data
            prehashed: Whether input is already hashed
            
        Returns:
            True if signature is valid
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Encode input data
            encoded_input = base64.b64encode(input_data.encode()).decode()
            
            verify_params = {
                'name': key_name,
                'input': encoded_input,
                'signature': signature,
                'hash_algorithm': hash_algorithm,
                'prehashed': prehashed,
                'mount_point': self.mount_point
            }
            
            if context:
                verify_params['context'] = base64.b64encode(context.encode()).decode()
            
            response = client.client.secrets.transit.verify_signed_data(**verify_params)
            
            return response['data']['valid']
            
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}", key_name=key_name)
            raise VaultException(f"Failed to verify signature with key {key_name}: {e}")
    
    async def generate_hmac(self, 
                           key_name: str, 
                           input_data: str,
                           hash_algorithm: str = "sha2-256",
                           key_version: Optional[int] = None) -> str:
        """
        Generate HMAC using transit key
        
        Args:
            key_name: Name of the HMAC key
            input_data: Data to generate HMAC for
            hash_algorithm: Hash algorithm to use
            key_version: Specific key version to use
            
        Returns:
            HMAC string
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Encode input data
            encoded_input = base64.b64encode(input_data.encode()).decode()
            
            hmac_params = {
                'name': key_name,
                'input': encoded_input,
                'hash_algorithm': hash_algorithm,
                'mount_point': self.mount_point
            }
            
            if key_version:
                hmac_params['key_version'] = key_version
            
            response = client.client.secrets.transit.generate_hmac(**hmac_params)
            
            return response['data']['hmac']
            
        except Exception as e:
            logger.error(f"Failed to generate HMAC: {e}", key_name=key_name)
            raise VaultException(f"Failed to generate HMAC with key {key_name}: {e}")
    
    async def verify_hmac(self, 
                         key_name: str, 
                         input_data: str,
                         hmac: str,
                         hash_algorithm: str = "sha2-256",
                         key_version: Optional[int] = None) -> bool:
        """
        Verify HMAC using transit key
        
        Args:
            key_name: Name of the HMAC key
            input_data: Original data
            hmac: HMAC to verify
            hash_algorithm: Hash algorithm used
            key_version: Specific key version to use
            
        Returns:
            True if HMAC is valid
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Encode input data
            encoded_input = base64.b64encode(input_data.encode()).decode()
            
            verify_params = {
                'name': key_name,
                'input': encoded_input,
                'hmac': hmac,
                'hash_algorithm': hash_algorithm,
                'mount_point': self.mount_point
            }
            
            if key_version:
                verify_params['key_version'] = key_version
            
            response = client.client.secrets.transit.verify_hmac(**verify_params)
            
            return response['data']['valid']
            
        except Exception as e:
            logger.error(f"Failed to verify HMAC: {e}", key_name=key_name)
            raise VaultException(f"Failed to verify HMAC with key {key_name}: {e}")
    
    async def delete_key(self, key_name: str) -> bool:
        """
        Delete encryption key
        
        Args:
            key_name: Name of the key to delete
            
        Returns:
            True if successful
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            # Delete key
            client.client.secrets.transit.delete_key(
                name=key_name,
                mount_point=self.mount_point
            )
            
            # Remove from cache
            if key_name in self.key_cache:
                del self.key_cache[key_name]
            
            logger.info(f"Encryption key deleted: {key_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete key: {e}", key_name=key_name)
            raise VaultException(f"Failed to delete key {key_name}: {e}")
    
    async def list_keys(self) -> List[str]:
        """
        List all encryption keys
        
        Returns:
            List of key names
        """
        try:
            client = await self._get_vault_client()
            
            if not client.client:
                raise VaultException("Vault client not initialized")
            
            response = client.client.secrets.transit.list_keys(
                mount_point=self.mount_point
            )
            
            return response['data']['keys']
            
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            raise VaultException(f"Failed to list keys: {e}")
    
    def clear_cache(self):
        """Clear key cache"""
        self.key_cache.clear()
        logger.info("Encryption key cache cleared")

# Global transit encryption instance
transit_encryption: Optional[VaultTransitEncryption] = None

async def get_transit_encryption() -> VaultTransitEncryption:
    """Get global transit encryption instance"""
    global transit_encryption
    
    if not transit_encryption:
        transit_encryption = VaultTransitEncryption()
    
    return transit_encryption

# Convenience functions for common operations
async def encrypt_data(key_name: str, plaintext: str, context: Optional[str] = None) -> EncryptionResult:
    """Encrypt data using transit key"""
    encryption = await get_transit_encryption()
    return await encryption.encrypt(key_name, plaintext, context)

async def decrypt_data(key_name: str, ciphertext: str, context: Optional[str] = None) -> DecryptionResult:
    """Decrypt data using transit key"""
    encryption = await get_transit_encryption()
    return await encryption.decrypt(key_name, ciphertext, context)

async def rotate_encryption_key(key_name: str) -> EncryptionKey:
    """Rotate encryption key"""
    encryption = await get_transit_encryption()
    return await encryption.rotate_key(key_name)

async def sign_data(key_name: str, input_data: str, hash_algorithm: str = "sha2-256") -> str:
    """Sign data using transit key"""
    encryption = await get_transit_encryption()
    return await encryption.sign(key_name, input_data, hash_algorithm)

async def verify_signature(key_name: str, input_data: str, signature: str, hash_algorithm: str = "sha2-256") -> bool:
    """Verify signature using transit key"""
    encryption = await get_transit_encryption()
    return await encryption.verify(key_name, input_data, signature, hash_algorithm)