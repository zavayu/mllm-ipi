from dataclasses import dataclass


@dataclass
class MockVisionModelClient:
    response: str = "mock response"

    def query(self, image_path: str, instruction: str) -> str:
        return self.response
