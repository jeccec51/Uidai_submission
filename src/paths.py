"""Project root: the repository folder (override with the UIDAI_ROOT environment variable)."""
import os
ROOT = os.environ.get("UIDAI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
