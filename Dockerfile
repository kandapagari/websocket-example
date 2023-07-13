FROM python:3.10-slim as base

WORKDIR /workdir

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /workdir

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7890

ENTRYPOINT [ "python", "server.py" ]
