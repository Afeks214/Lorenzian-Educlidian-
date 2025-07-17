#!/bin/bash
# SSL Certificate Generation Script
# Agent Epsilon - Production Deployment Specialist

set -euo pipefail

SSL_DIR="/etc/nginx/ssl"
DOMAIN="${XAI_DOMAIN:-localhost}"

echo "🔐 Generating SSL certificates for domain: $DOMAIN"

# Create SSL directory if it doesn't exist
mkdir -p "$SSL_DIR"

# Generate private key
openssl genpkey -algorithm RSA -out "$SSL_DIR/server.key" -pkcs8 -aes256 -pass pass:temppass 2>/dev/null || \
openssl genrsa -out "$SSL_DIR/server.key" 2048

# Remove passphrase from key
openssl rsa -in "$SSL_DIR/server.key" -out "$SSL_DIR/server.key" -passin pass:temppass 2>/dev/null || true

# Generate certificate signing request
openssl req -new -key "$SSL_DIR/server.key" -out "$SSL_DIR/server.csr" -subj "/C=US/ST=NY/L=NYC/O=QuantNova/OU=XAI/CN=$DOMAIN"

# Generate self-signed certificate (valid for 1 year)
openssl x509 -req -days 365 -in "$SSL_DIR/server.csr" -signkey "$SSL_DIR/server.key" -out "$SSL_DIR/server.crt" \
    -extensions v3_req -extfile <(cat <<EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = *.$DOMAIN
DNS.3 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
)

# Set proper permissions
chmod 600 "$SSL_DIR/server.key"
chmod 644 "$SSL_DIR/server.crt"
chmod 644 "$SSL_DIR/server.csr"

# Clean up CSR
rm -f "$SSL_DIR/server.csr"

echo "✅ SSL certificates generated successfully"
echo "📁 Certificate: $SSL_DIR/server.crt"
echo "🔑 Private Key: $SSL_DIR/server.key"

# Verify certificate
echo "🔍 Certificate verification:"
openssl x509 -in "$SSL_DIR/server.crt" -text -noout | grep -E "(Subject:|DNS:|IP Address:|Not After)"