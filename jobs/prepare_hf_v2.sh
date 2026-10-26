#!/bin/bash
# Download snowballlab/30-keywords-v2 and write processed/<label>/{train,val,test}_*.wav
# Run on a machine with internet (Anvil login node is fine; GPU nodes often are not).

set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q datasets
python src/prepare_hf_v2.py
