FROM n8nio/n8n:1.81.1

USER root

# Install Python + pip
RUN apk add --no-cache python3 py3-pip

# Copy project files
COPY ./data /data/data
COPY ./models /data/models
COPY ./output /data/output
COPY ./src /data/src
COPY ./requirements.txt /data/requirements.txt

# Install dependencies (allow break-system-packages for Alpine Python)
RUN pip3 install --break-system-packages --no-cache-dir -r /data/requirements.txt || true

WORKDIR /data
EXPOSE 10000

CMD ["n8n"]
