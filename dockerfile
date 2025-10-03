FROM n8nio/n8n:1.81.1

USER root

# Install Python, pip, and build dependencies (temporary)
RUN apk add --no-cache python3 py3-pip \
    && apk add --no-cache --virtual .build-deps \
       g++ gcc make musl-dev python3-dev libffi-dev

# Copy project files
COPY ./data /data/data
COPY ./models /data/models
COPY ./output /data/output
COPY ./src /data/src
COPY ./requirements.txt /data/requirements.txt

# Install Python dependencies
RUN pip3 install --break-system-packages --no-cache-dir -r /data/requirements.txt || true

# Remove build dependencies to shrink image
RUN apk del .build-deps

WORKDIR /data
EXPOSE 10000

CMD ["n8n"]
