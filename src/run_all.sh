#!/bin/bash
# resumable: JIPNet on all 8 DBs, then AFR-Net re-implementation (paper test sets, never seen, then seen)
cd "$(dirname "$0")/.."
python3 src/run_pairs.py JIPNet 2002_DB1 2002_DB3 2002_DB2 2004_DB3 2002_DB4 2004_DB4 2004_DB1 2004_DB2
python3 src/run_pairs.py AFRNet 2002_DB1 2002_DB3 2002_DB2 2004_DB3 2002_DB4 2004_DB4 2004_DB1 2004_DB2
