# Falcon ReID — two images from one Dockerfile:
#   target "infer"   : batch run that writes the three official files (what the organizers run)
#   target "service" : the same inference code + FastAPI web service (docker-compose.yml)
#
#   docker build --target infer -t falcon-reid .
#   docker run --rm --gpus all --network none \
#       -v /path/to/dataset:/data:ro -v /path/to/output:/out falcon-reid
#   (/data must contain images/, test_query.csv, test_gallery.csv)
#
# Build may use the internet (answer #39); running needs none: every weight is inside the image.
FROM python:3.11-slim-bookworm AS infer

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# 1) PyTorch built for CUDA 12.6. The organizers' GPU driver is CUDA 12.2 (answer #32); any CUDA 12.x
#    build runs on it (NVIDIA minor-version compatibility), while the default PyPI torch is CUDA 13 and
#    would need driver >= 580. The assert makes the build FAIL if a CUDA 13 torch ever sneaks in.
RUN pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
RUN python -c "import torch; assert torch.version.cuda.startswith('12.'), torch.version.cuda; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"

# 2) the rest, pinned with == (answer #39)
COPY requirements-infer.txt .
RUN pip install -r requirements-infer.txt \
 && python -c "import torch; assert torch.version.cuda.startswith('12.'), 'torch was replaced: ' + torch.version.cuda"

# 3) code + weights (downloaded here if not in the build context; sha256 always checked)
COPY reid/ reid/
COPY falcon/ falcon/
COPY tools/fetch_weights.py tools/export_weights.py tools/
COPY weights/ weights/
RUN python tools/fetch_weights.py

# 4) self-test at build time: both modes load fully offline and produce vectors of the right size
RUN python -c "import numpy as np; from PIL import Image; from falcon import Extractor; \
img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype('uint8')); \
[print(m, Extractor(mode=m, device='cpu').extract(img, (10, 10, 300, 200)).shape) for m in ('fast', 'accurate')]"

ENTRYPOINT ["python", "-m", "falcon.predict"]
CMD ["--images", "/data/images", "--query", "/data/test_query.csv", "--gallery", "/data/test_gallery.csv", "--out", "/out"]


# ---------------------------------------------------------------- web service (API) image
FROM infer AS service
COPY requirements-service.txt .
RUN pip install -r requirements-service.txt \
 && python -c "import torch; assert torch.version.cuda.startswith('12.'), 'torch was replaced: ' + torch.version.cuda"
COPY service/__init__.py service/__init__.py
COPY service/api/ service/api/
EXPOSE 8000
ENTRYPOINT []
CMD ["uvicorn", "service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
