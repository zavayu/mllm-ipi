from src.model_clients import MockVisionModelClient, VisionModelClient


def test_mock_vision_model_client_returns_default_response():
    client = MockVisionModelClient()

    assert client.query("image.png", "Describe the image") == "mock response"


def test_mock_vision_model_client_returns_configured_response():
    client = MockVisionModelClient(response="configured answer")

    assert client.query("image.png", "Describe the image") == "configured answer"


def test_mock_vision_model_client_satisfies_vision_model_client_protocol():
    client = MockVisionModelClient(response="ok")

    assert isinstance(client, VisionModelClient)
