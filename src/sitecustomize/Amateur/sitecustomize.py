# Auto-applied vLLM GPUModelRunner._sample patch in every Python process (incl. spawned workers)
# Put this directory at the front of PYTHONPATH so Python imports this sitecustomize on startup.
# src/sitecustomize/Amateur/sitecustomize.py
import sys
from pathlib import Path
import traceback

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

try:
    import patch_vllm_sample as _p
    import patch_vllm_scheduler as _p2
    _p.install_patch("Amateur")
    _p2.install_patch("Amateur")
    print("[sitecustomize] Installed GPUModelRunner._sample patch")
    print("[sitecustomize] Installed LLM._add_request batch throttle patch")
except Exception as e:
    print("[sitecustomize] ❌ Failed to install patch — aborting startup!(This is necessary for contrastive decoding)", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
