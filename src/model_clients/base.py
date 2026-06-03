from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionModelClient(Protocol):
    def query(self, image_path: str, instruction: str) -> str:
        """Return the model response for an image and instruction."""
        ...
