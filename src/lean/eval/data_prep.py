"""
Data Preparation and Vault Ingestion Script for Project LEAN.

Initializes calibration, dev, and locked test splits in vault/
and registers SHA-256 manifests.
"""

import sys
from pathlib import Path

from vault.guard import register_vault_file, load_manifest, VAULT_DIR


SAMPLE_CALIBRATION_TEXT = """The concept of description length in statistical learning theory provides a framework for measuring model complexity.
A neural network parameterization can be viewed as an encoding of regularities found in the training distribution.
Algorithmic components, such as sequence reversal, induction heads, and copy mechanisms, constitute reusable subroutines.
These computational subroutines exhibit high execution frequency across diverse linguistic domains.
"""

SAMPLE_DEV_TEXT = """In transformer architectures, attention mechanisms route information dynamically between token representations.
Multi-layer perceptron blocks store factual associations as key-value associative memories.
When subjected to parameter quantization, factual recall exhibits sharp degradation before procedural reasoning collapses.
"""

SAMPLE_LOCKED_T1_TEXT = """Held-out test tranche 1 evaluation split: Qwen2.5 and SmolLM2 benchmark texts.
This text is cryptographically protected by the Project LEAN vault guard.
Premature evaluation before formal preregistration is strictly prohibited.
"""

SAMPLE_LOCKED_T2_TEXT = """Held-out test tranche 2 evaluation split: Pythia scale extrapolation (1.4B and 2.8B).
Cryptographically protected text.
"""

SAMPLE_LOCKED_T3_TEXT = """Held-out test tranche 3 evaluation split: OLMo-1B Dolma validation and future post-lock models.
Cryptographically protected text.
"""


def init_vault_splits():
    """Create directory structure and initialize vault datasets."""
    raw_dir = VAULT_DIR / "raw"
    locked_dir = VAULT_DIR / "locked"
    raw_dir.mkdir(parents=True, exist_ok=True)
    locked_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ("raw/calibration.txt", "calibration", SAMPLE_CALIBRATION_TEXT),
        ("raw/dev.txt", "dev", SAMPLE_DEV_TEXT),
        ("locked/t1_test.txt", "t1", SAMPLE_LOCKED_T1_TEXT),
        ("locked/t2_test.txt", "t2", SAMPLE_LOCKED_T2_TEXT),
        ("locked/t3_test.txt", "t3", SAMPLE_LOCKED_T3_TEXT),
    ]
    
    for rel_path, split, text in splits:
        fp = VAULT_DIR / rel_path
        with open(fp, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        sha = register_vault_file(rel_path, split)
        print(f"Registered [{split}] {rel_path} -> SHA256: {sha[:16]}...")
        
    manifest = load_manifest()
    print(f"\nManifest successfully updated with {len(manifest.get('files', {}))} files.")


if __name__ == "__main__":
    init_vault_splits()
