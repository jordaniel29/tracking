"""Person detection.

    factory.py    ``build_detector(cfg)`` — config block → detector (YOLO26 via piaspace_yolo26)
    normalize.py  any detector's output → the ``Detection`` schema the tracker reads
"""

from .factory import build_detector
from .normalize import to_schema_detection

__all__ = ["build_detector", "to_schema_detection"]
