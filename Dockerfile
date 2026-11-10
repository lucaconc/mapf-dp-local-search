# Reproducible environment for mapf_bench.
#   docker build -t mapf_bench .
#   docker run --rm -v "$PWD/results:/mapf_bench/results" mapf_bench \
#          run configs/paper.json --workers 8 --pin
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates python3 python3-venv python3-pip \
        libboost-program-options-dev libboost-system-dev libboost-filesystem-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /mapf_bench
COPY . .
RUN scripts/setup_local.sh
ENV PATH="/mapf_bench/.venv/bin:${PATH}"
ENTRYPOINT ["python", "-m", "mapf_bench.local_runner"]
CMD ["doctor"]
