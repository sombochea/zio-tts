# ponytail: coqui-tts (XTTS code, unused here) imports a helper removed in transformers 5; shim it.
# Drop when coqui-tts catches up.
import torch
import transformers.pytorch_utils as _pu

if not hasattr(_pu, "isin_mps_friendly"):
    _pu.isin_mps_friendly = torch.isin
