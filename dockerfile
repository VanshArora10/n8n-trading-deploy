# Start with n8n image
FROM n8nio/n8n:latest

# Switch to root to install Python
USER root

# Install Python + pip
RUN apk add --no-cache python3 py3-pip

# Copy your project into the container
COPY ./data /data/data
COPY ./src /data/src
COPY ./output /data/output
COPY ./requirements.txt /data/requirements.txt

# Install Python dependencies
RUN pip3 install -r /data/requirements.txt || true

# Set working directory
WORKDIR /data

# Expose Render port
EXPOSE 10000

# Run n8n
CMD ["n8n"]
