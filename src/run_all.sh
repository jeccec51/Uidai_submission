#!/bin/bash
# Run both models exactly as used in the report (resumable).
#   JIPNet: all pairs of all 8 FVC databases.
#   AFR-Net-style baseline: fixed-length templates on FVC2002 DB1 and DB3 (400 genuine pairs each).
cd "$(dirname "$0")/.."
python3 src/run_pairs.py JIPNet 2002_DB1 2002_DB2 2002_DB3 2002_DB4 2004_DB1 2004_DB2 2004_DB3 2004_DB4
python3 src/run_afr_embed.py 2002_DB1 2002_DB3
