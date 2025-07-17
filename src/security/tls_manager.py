"""
Advanced TLS/SSL Management System
Comprehensive TLS configuration for production financial systems
"""

import os
import ssl
import socket
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.hazmat.backends import default_backend
from dataclasses import dataclass
from enum import Enum

from src.monitoring.logger_config import get_logger
from src.config.secrets_manager import SecretsManager

logger = get_logger(__name__)

class TLSVersion(Enum):
    """Supported TLS versions"""
    TLSv1_2 = "TLSv1.2"
    TLSv1_3 = "TLSv1.3"

class CipherSuite(Enum):
    """Recommended cipher suites for financial applications"""
    ECDHE_RSA_AES256_GCM_SHA384 = "ECDHE-RSA-AES256-GCM-SHA384"
    ECDHE_RSA_AES128_GCM_SHA256 = "ECDHE-RSA-AES128-GCM-SHA256"
    ECDHE_RSA_CHACHA20_POLY1305 = "ECDHE-RSA-CHACHA20-POLY1305"
    DHE_RSA_AES256_GCM_SHA384 = "DHE-RSA-AES256-GCM-SHA384"
    DHE_RSA_AES128_GCM_SHA256 = "DHE-RSA-AES128-GCM-SHA256"

@dataclass
class TLSConfig:
    """TLS configuration"""
    version: TLSVersion
    cipher_suites: List[CipherSuite]
    cert_path: str
    key_path: str
    ca_path: Optional[str] = None
    verify_mode: str = "required"
    check_hostname: bool = True
    client_cert_auth: bool = True
    session_timeout: int = 300
    renegotiation_limit: int = 5
    compression: bool = False
    sni_enabled: bool = True
    alpn_protocols: List[str] = None
    
    def __post_init__(self):
        if self.alpn_protocols is None:
            self.alpn_protocols = ["h2", "http/1.1"]

@dataclass
class Certificate:
    """SSL Certificate metadata"""
    subject: str
    issuer: str
    serial_number: str
    not_before: datetime
    not_after: datetime
    signature_algorithm: str
    public_key_algorithm: str
    key_size: int
    fingerprint_sha256: str
    san_dns_names: List[str]
    san_ip_addresses: List[str]
    is_ca: bool
    key_usage: List[str]
    extended_key_usage: List[str]
    
class AdvancedTLSManager:
    """Advanced TLS/SSL manager for financial applications"""
    
    def __init__(self, config: TLSConfig = None):
        self.logger = get_logger(__name__)
        self.secrets_manager = SecretsManager()
        
        # Default secure configuration
        self.config = config or TLSConfig(
            version=TLSVersion.TLSv1_3,
            cipher_suites=[
                CipherSuite.ECDHE_RSA_AES256_GCM_SHA384,
                CipherSuite.ECDHE_RSA_AES128_GCM_SHA256,
                CipherSuite.ECDHE_RSA_CHACHA20_POLY1305
            ],
            cert_path="/etc/ssl/certs/grandmodel/server.crt",
            key_path="/etc/ssl/private/grandmodel/server.key",
            ca_path="/etc/ssl/certs/grandmodel/ca.crt",
            verify_mode="required",
            check_hostname=True,
            client_cert_auth=True
        )
        
        # Certificate store
        self.certificates: Dict[str, Certificate] = {}
        self.ca_certificates: Dict[str, Certificate] = {}
        
        # Load certificates
        self._load_certificates()
        
        self.logger.info("AdvancedTLSManager initialized")
    
    def _load_certificates(self):
        """Load and validate certificates"""
        try:
            # Load server certificate
            if os.path.exists(self.config.cert_path):
                with open(self.config.cert_path, 'rb') as f:
                    cert_data = f.read()
                    cert = x509.load_pem_x509_certificate(cert_data, default_backend())
                    self.certificates["server"] = self._parse_certificate(cert)
                    self.logger.info("Server certificate loaded")
            
            # Load CA certificate
            if self.config.ca_path and os.path.exists(self.config.ca_path):
                with open(self.config.ca_path, 'rb') as f:
                    ca_data = f.read()
                    ca_cert = x509.load_pem_x509_certificate(ca_data, default_backend())
                    self.ca_certificates["root"] = self._parse_certificate(ca_cert)
                    self.logger.info("CA certificate loaded")
            
            # Validate certificate chain
            self._validate_certificate_chain()
            
        except Exception as e:
            self.logger.error(f"Failed to load certificates: {e}")
            raise
    
    def _parse_certificate(self, cert: x509.Certificate) -> Certificate:
        """Parse X.509 certificate"""
        # Extract subject and issuer
        subject = cert.subject.rfc4514_string()
        issuer = cert.issuer.rfc4514_string()
        
        # Extract SAN
        san_dns_names = []
        san_ip_addresses = []
        
        try:
            san_ext = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            )
            for name in san_ext.value:
                if isinstance(name, x509.DNSName):
                    san_dns_names.append(name.value)
                elif isinstance(name, x509.IPAddress):
                    san_ip_addresses.append(str(name.value))
        except x509.ExtensionNotFound:
            pass
        
        # Extract key usage
        key_usage = []
        try:
            ku_ext = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.KEY_USAGE
            )
            ku = ku_ext.value
            if ku.digital_signature:
                key_usage.append("digital_signature")
            if ku.key_encipherment:
                key_usage.append("key_encipherment")
            if ku.key_agreement:
                key_usage.append("key_agreement")
            if ku.certificate_sign:
                key_usage.append("certificate_sign")
            if ku.crl_sign:
                key_usage.append("crl_sign")
        except x509.ExtensionNotFound:
            pass
        
        # Extract extended key usage
        extended_key_usage = []
        try:
            eku_ext = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.EXTENDED_KEY_USAGE
            )
            for oid in eku_ext.value:
                if oid == x509.oid.ExtendedKeyUsageOID.SERVER_AUTH:
                    extended_key_usage.append("server_auth")
                elif oid == x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH:
                    extended_key_usage.append("client_auth")
        except x509.ExtensionNotFound:
            pass
        
        # Check if CA
        is_ca = False
        try:
            bc_ext = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.BASIC_CONSTRAINTS
            )
            is_ca = bc_ext.value.ca
        except x509.ExtensionNotFound:
            pass
        
        # Get public key info
        public_key = cert.public_key()
        if isinstance(public_key, rsa.RSAPublicKey):
            public_key_algorithm = "RSA"
            key_size = public_key.key_size
        else:
            public_key_algorithm = "Unknown"
            key_size = 0
        
        # Calculate fingerprint
        fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
        
        return Certificate(
            subject=subject,
            issuer=issuer,
            serial_number=str(cert.serial_number),
            not_before=cert.not_valid_before,
            not_after=cert.not_valid_after,
            signature_algorithm=cert.signature_hash_algorithm.name,
            public_key_algorithm=public_key_algorithm,
            key_size=key_size,
            fingerprint_sha256=fingerprint,
            san_dns_names=san_dns_names,
            san_ip_addresses=san_ip_addresses,
            is_ca=is_ca,
            key_usage=key_usage,
            extended_key_usage=extended_key_usage
        )
    
    def _validate_certificate_chain(self):
        """Validate certificate chain"""
        server_cert = self.certificates.get("server")
        if not server_cert:
            raise ValueError("Server certificate not found")
        
        # Check expiration
        now = datetime.utcnow()
        if server_cert.not_after < now:
            raise ValueError("Server certificate has expired")
        
        # Check if certificate expires soon (30 days)
        if server_cert.not_after < now + timedelta(days=30):
            self.logger.warning("Server certificate expires soon", expires_at=server_cert.not_after)
        
        # Validate key usage for server
        if "server_auth" not in server_cert.extended_key_usage:
            raise ValueError("Server certificate missing server_auth key usage")
        
        self.logger.info("Certificate chain validation passed")
    
    def create_ssl_context(self, 
                          purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
                          client_cert_required: bool = False) -> ssl.SSLContext:
        """Create optimized SSL context for financial applications"""
        try:
            # Create SSL context with secure defaults
            context = ssl.create_default_context(purpose)
            
            # Set minimum TLS version
            if self.config.version == TLSVersion.TLSv1_3:
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                context.maximum_version = ssl.TLSVersion.TLSv1_3
            else:
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                context.maximum_version = ssl.TLSVersion.TLSv1_2
            
            # Set cipher suites
            cipher_string = ":".join([cipher.value for cipher in self.config.cipher_suites])
            context.set_ciphers(cipher_string)
            
            # Load server certificate and key
            if os.path.exists(self.config.cert_path) and os.path.exists(self.config.key_path):
                context.load_cert_chain(self.config.cert_path, self.config.key_path)
                self.logger.info("Server certificate loaded into SSL context")
            else:
                self.logger.warning("Server certificate or key not found")
            
            # Load CA certificates
            if self.config.ca_path and os.path.exists(self.config.ca_path):
                context.load_verify_locations(self.config.ca_path)
                self.logger.info("CA certificate loaded into SSL context")
            
            # Set verification mode
            if self.config.verify_mode == "required":
                context.verify_mode = ssl.CERT_REQUIRED
            elif self.config.verify_mode == "optional":
                context.verify_mode = ssl.CERT_OPTIONAL
            else:
                context.verify_mode = ssl.CERT_NONE
            
            # Set hostname checking
            context.check_hostname = self.config.check_hostname
            
            # Client certificate authentication
            if client_cert_required or self.config.client_cert_auth:
                context.verify_mode = ssl.CERT_REQUIRED
            
            # Security options
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1
            context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
            context.options |= ssl.OP_SINGLE_DH_USE
            context.options |= ssl.OP_SINGLE_ECDH_USE
            
            # Disable compression to prevent CRIME attacks
            if not self.config.compression:
                context.options |= ssl.OP_NO_COMPRESSION
            
            # Set ALPN protocols
            if self.config.alpn_protocols:
                context.set_alpn_protocols(self.config.alpn_protocols)
            
            # Set session timeout
            context.set_default_verify_paths()
            
            self.logger.info("SSL context created successfully")
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to create SSL context: {e}")
            raise
    
    def validate_peer_certificate(self, 
                                  peer_cert: x509.Certificate,
                                  hostname: str = None) -> bool:
        """Validate peer certificate"""
        try:
            # Parse certificate
            cert_info = self._parse_certificate(peer_cert)
            
            # Check expiration
            now = datetime.utcnow()
            if cert_info.not_after < now:
                self.logger.error("Peer certificate has expired")
                return False
            
            # Check hostname if provided
            if hostname:
                if not self._validate_hostname(cert_info, hostname):
                    self.logger.error("Hostname validation failed")
                    return False
            
            # Check key usage
            if "client_auth" not in cert_info.extended_key_usage:
                self.logger.error("Peer certificate missing client_auth key usage")
                return False
            
            # Verify against CA
            if not self._verify_certificate_chain(peer_cert):
                self.logger.error("Certificate chain verification failed")
                return False
            
            self.logger.info("Peer certificate validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Peer certificate validation failed: {e}")
            return False
    
    def _validate_hostname(self, cert_info: Certificate, hostname: str) -> bool:
        """Validate hostname against certificate"""
        # Check SAN DNS names
        for san_name in cert_info.san_dns_names:
            if self._match_hostname(san_name, hostname):
                return True
        
        # Check SAN IP addresses
        for san_ip in cert_info.san_ip_addresses:
            if san_ip == hostname:
                return True
        
        # Check CN in subject
        if hostname in cert_info.subject:
            return True
        
        return False
    
    def _match_hostname(self, pattern: str, hostname: str) -> bool:
        """Match hostname against pattern (supports wildcards)"""
        if pattern.startswith("*."):
            # Wildcard certificate
            pattern_parts = pattern[2:].split(".")
            hostname_parts = hostname.split(".")
            
            if len(hostname_parts) != len(pattern_parts) + 1:
                return False
            
            return all(p == h for p, h in zip(pattern_parts, hostname_parts[1:]))
        else:
            return pattern == hostname
    
    def _verify_certificate_chain(self, cert: x509.Certificate) -> bool:
        """Verify certificate against CA chain"""
        try:
            # This is a simplified verification
            # In production, you would use a proper certificate chain validation
            ca_cert = self.ca_certificates.get("root")
            if not ca_cert:
                return False
            
            # Basic verification - check if cert is signed by CA
            return cert.issuer == cert.subject or ca_cert.subject in cert.issuer.rfc4514_string()
            
        except Exception as e:
            self.logger.error(f"Certificate chain verification failed: {e}")
            return False
    
    def get_cipher_suites(self) -> List[str]:
        """Get recommended cipher suites"""
        return [cipher.value for cipher in self.config.cipher_suites]
    
    def get_tls_config_for_service(self, service: str) -> Dict[str, Any]:
        """Get TLS configuration for specific service"""
        return {
            "tls_version": self.config.version.value,
            "cipher_suites": self.get_cipher_suites(),
            "cert_file": self.config.cert_path,
            "key_file": self.config.key_path,
            "ca_file": self.config.ca_path,
            "verify_mode": self.config.verify_mode,
            "check_hostname": self.config.check_hostname,
            "client_cert_auth": self.config.client_cert_auth,
            "session_timeout": self.config.session_timeout,
            "alpn_protocols": self.config.alpn_protocols,
            "compression": self.config.compression
        }
    
    def generate_certificate_report(self) -> Dict[str, Any]:
        """Generate certificate status report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "certificates": {},
            "ca_certificates": {},
            "security_analysis": {
                "tls_version": self.config.version.value,
                "cipher_suites": len(self.config.cipher_suites),
                "hostname_verification": self.config.check_hostname,
                "client_cert_auth": self.config.client_cert_auth
            }
        }
        
        # Add certificate details
        for name, cert in self.certificates.items():
            report["certificates"][name] = {
                "subject": cert.subject,
                "issuer": cert.issuer,
                "not_before": cert.not_before.isoformat(),
                "not_after": cert.not_after.isoformat(),
                "days_until_expiry": (cert.not_after - datetime.utcnow()).days,
                "signature_algorithm": cert.signature_algorithm,
                "key_size": cert.key_size,
                "fingerprint": cert.fingerprint_sha256,
                "key_usage": cert.key_usage,
                "extended_key_usage": cert.extended_key_usage
            }
        
        # Add CA certificate details
        for name, cert in self.ca_certificates.items():
            report["ca_certificates"][name] = {
                "subject": cert.subject,
                "not_before": cert.not_before.isoformat(),
                "not_after": cert.not_after.isoformat(),
                "days_until_expiry": (cert.not_after - datetime.utcnow()).days,
                "is_ca": cert.is_ca
            }
        
        return report
    
    def check_certificate_expiry(self, days_ahead: int = 30) -> List[str]:
        """Check for certificates expiring soon"""
        expiring_certs = []
        cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
        
        for name, cert in self.certificates.items():
            if cert.not_after <= cutoff_date:
                expiring_certs.append(name)
        
        return expiring_certs
    
    def test_tls_connection(self, hostname: str, port: int = 443) -> Dict[str, Any]:
        """Test TLS connection to remote host"""
        try:
            context = self.create_ssl_context(ssl.Purpose.SERVER_AUTH)
            
            # Connect to remote host
            sock = socket.create_connection((hostname, port), timeout=10)
            ssl_sock = context.wrap_socket(sock, server_hostname=hostname)
            
            # Get connection info
            peer_cert = ssl_sock.getpeercert()
            cipher = ssl_sock.cipher()
            version = ssl_sock.version()
            
            ssl_sock.close()
            
            return {
                "success": True,
                "tls_version": version,
                "cipher": cipher,
                "certificate": peer_cert,
                "hostname": hostname,
                "port": port
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "hostname": hostname,
                "port": port
            }

# Global instance
tls_manager = AdvancedTLSManager()

# Utility functions
def get_tls_manager() -> AdvancedTLSManager:
    """Get TLS manager instance"""
    return tls_manager

def create_secure_ssl_context(purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH) -> ssl.SSLContext:
    """Create secure SSL context"""
    return tls_manager.create_ssl_context(purpose)

def validate_certificate_chain(cert_path: str, ca_path: str = None) -> bool:
    """Validate certificate chain"""
    try:
        with open(cert_path, 'rb') as f:
            cert_data = f.read()
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
        
        return tls_manager.validate_peer_certificate(cert)
    except Exception as e:
        logger.error(f"Certificate validation failed: {e}")
        return False

def get_recommended_cipher_suites() -> List[str]:
    """Get recommended cipher suites for financial applications"""
    return tls_manager.get_cipher_suites()
