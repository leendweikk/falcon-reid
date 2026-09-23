import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # project root (this file lives in experiments/)
sys.path.insert(0, str(ROOT))                   # so `import reid` works when run from anywhere

import torch

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("GPU:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("VRAM (GB):", round(props.total_memory / 1024**3, 1))

# fp16 test: multiply two big random matrices on the GPU in half precision
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
c = a @ b
print("fp16 works, any NaN:", torch.isnan(c).any().item())