FROM nginx:1.25.3-alpine

# Install additional tools for SSL and security
RUN apk add --no-cache \
    openssl \
    curl \
    bash

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/ /etc/nginx/conf.d/

# Copy SSL generation script
COPY generate-ssl.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/generate-ssl.sh

# Create SSL directory
RUN mkdir -p /etc/nginx/ssl

# Generate self-signed certificates (for development/testing)
# In production, these should be replaced with proper certificates
RUN /usr/local/bin/generate-ssl.sh

# Create custom error pages
COPY error-pages/ /usr/share/nginx/html/error-pages/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Expose ports
EXPOSE 80 443

# Start nginx
CMD ["nginx", "-g", "daemon off;"]