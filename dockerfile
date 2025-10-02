# Start with n8n as the base image (includes n8n already)
FROM n8nio/n8n:latest

# Switch to root to install Python
USER root

# Install Python + pip
RUN apk add --no-cache python3 py3-pip

# Copy your project files into the container
# Adjust these if your repo structure is different
COPY ./data /data/data
COPY ./src /data/src
COPY ./requirements.txt /data/requirements.txt

# Install Python dependencies (from requirements.txt)
RUN pip3 install -r /data/requirements.txt || true

# Set working directory
WORKDIR /data

# Expose Render’s expected port
EXPOSE 10000

# Start n8n when container boots
CMD ["n8n"]
