FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS reg2026_algorithm_amd64
# PyTorch + CUDA base so GPU is available at inference time.

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure imports like "from core import ..." work from src/ submodules
ENV PYTHONPATH=/opt/app
ENV PATH=/home/user/.local/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgl1 \
        libglib2.0-0 \
        libopenslide0 \
        libvips42 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user REG2026/algorithm_submission_template/requirements.txt /opt/app/requirements.txt
COPY --chown=user:user path_wsi_reasoner /opt/vendor/path_wsi_reasoner
COPY --chown=user:user TRIDENT /opt/vendor/TRIDENT

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
        --user \
        --no-cache-dir \
        --no-color \
        --upgrade pip setuptools wheel \
    && python -m pip install \
        --user \
        --no-cache-dir \
        --no-color \
        --requirement /opt/app/requirements.txt \
    && python -m pip install \
        --user \
        --no-cache-dir \
        --no-color \
        /opt/vendor/TRIDENT \
        /opt/vendor/path_wsi_reasoner \
    && python -c "import path_wsi_reasoner, trident, tiffslide"

COPY --chown=user:user REG2026/algorithm_submission_template/core.py      /opt/app/core.py
COPY --chown=user:user REG2026/algorithm_submission_template/inference.py /opt/app/inference.py
COPY --chown=user:user REG2026/algorithm_submission_template/src/         /opt/app/src/

ENTRYPOINT ["python", "inference.py"]
