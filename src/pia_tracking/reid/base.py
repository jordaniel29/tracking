"""Abstract base for ReID embedders."""
from abc import ABC, abstractmethod

import numpy as np


class ReIDBackend(ABC):
    """Interface for person/vehicle ReID embedding models.

    All implementations must produce a fixed-size embedding vector from a BGR
    image crop.  The pipeline routes person crops to a person embedder and
    vehicle crops to a vehicle embedder (configured in YAML).
    """

    @abstractmethod
    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of BGR image crops.

        Args:
            crops_bgr: list of (H, W, 3) uint8 BGR images (variable sizes OK;
                       implementations resize internally).

        Returns:
            (N, embed_dim) float32 L2-normalized embeddings.
        """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Embedding dimension."""
