from types import SimpleNamespace

import pytest

from src import query_model
from src.model_clients import QWEN25_VL_FAMILY, QWEN3_VL_FAMILY
from src.model_clients import QwenVisionModelClient, VisionModelClient
from src.model_clients import qwen_client


class FakeInputs(dict):
    def __init__(self):
        super().__init__()
        self.input_ids = [[1, 2, 3]]
        self["input_ids"] = self.input_ids
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeProcessor:
    from_pretrained_calls = []
    last_inputs = None
    last_messages = None

    @classmethod
    def from_pretrained(cls, model_id):
        cls.from_pretrained_calls.append(model_id)
        return cls()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.__class__.last_messages = messages
        if len(messages[0]["content"]) > 1:
            assert messages[0]["content"][1]["text"] == "Describe this image."
        return "chat template"

    def __call__(self, **kwargs):
        self.last_inputs = FakeInputs()
        assert kwargs["text"] == ["chat template"]
        assert kwargs["images"] == ["image input"]
        assert kwargs["videos"] is None
        assert kwargs["padding"] is True
        assert kwargs["return_tensors"] == "pt"
        return self.last_inputs

    def batch_decode(self, generated_ids, **kwargs):
        assert generated_ids == [[4, 5]]
        assert kwargs["skip_special_tokens"] is True
        assert kwargs["clean_up_tokenization_spaces"] is False
        return [" decoded answer "]


class FakeQwen3Processor(FakeProcessor):
    qwen3_chat_template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.__class__.qwen3_chat_template_calls.append((messages, kwargs))
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["return_dict"] is True
        assert kwargs["return_tensors"] == "pt"
        return FakeInputs()


class FakeModel:
    from_pretrained_calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.from_pretrained_calls.append((model_id, kwargs))
        return cls()

    def eval(self):
        self.did_eval = True

    def generate(self, **kwargs):
        assert kwargs["input_ids"] == [[1, 2, 3]]
        assert kwargs["max_new_tokens"] == 7
        return [[1, 2, 3, 4, 5]]


def fake_process_vision_info(messages):
    assert messages[0]["content"][0]["type"] == "image"
    return ["image input"], None


def fake_torch(cuda_available=True):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeProcessor.from_pretrained_calls = []
    FakeProcessor.last_inputs = None
    FakeProcessor.last_messages = None
    FakeQwen3Processor.from_pretrained_calls = []
    FakeQwen3Processor.qwen3_chat_template_calls = []
    FakeModel.from_pretrained_calls = []


def test_qwen_client_satisfies_vision_model_client_protocol():
    client = QwenVisionModelClient()

    assert isinstance(client, VisionModelClient)


def test_qwen_client_is_lazy_until_query(monkeypatch):
    def fail_if_loaded():
        raise AssertionError("torch should not be imported during construction")

    monkeypatch.setattr(qwen_client, "_import_torch", fail_if_loaded)

    QwenVisionModelClient(model_id="test/model")


def test_qwen_client_query_loads_model_once_and_decodes(monkeypatch, tmp_path):
    image = tmp_path / "example.png"
    image.write_bytes(b"not a real png; tests mock image processing")
    monkeypatch.setattr(qwen_client, "_import_torch", lambda: fake_torch(True))
    monkeypatch.setattr(
        qwen_client,
        "_import_qwen_libraries",
        lambda: (FakeProcessor, FakeModel, fake_process_vision_info),
    )
    client = QwenVisionModelClient(model_id="test/model", max_new_tokens=7)

    first = client.query(str(image), "Describe this image.")
    second = client.query(str(image), "Describe this image.")

    assert first == "decoded answer"
    assert second == "decoded answer"
    assert FakeProcessor.from_pretrained_calls == ["test/model"]
    assert FakeModel.from_pretrained_calls == [
        (
            "test/model",
            {"torch_dtype": "auto", "device_map": "auto"},
        )
    ]
    assert client._processor.last_inputs.moved_to == "cuda"
    assert client._resolved_model_family == QWEN25_VL_FAMILY


def test_qwen_client_auto_detects_qwen3_and_decodes(monkeypatch, tmp_path):
    image = tmp_path / "example.png"
    image.write_bytes(b"not a real png; tests mock image processing")
    monkeypatch.setattr(qwen_client, "_import_torch", lambda: fake_torch(True))
    monkeypatch.setattr(
        qwen_client,
        "_import_qwen3_libraries",
        lambda: (FakeQwen3Processor, FakeModel),
    )
    client = QwenVisionModelClient(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        model_family="auto",
        max_new_tokens=7,
    )

    assert client.query(str(image), "Describe this image.") == "decoded answer"
    assert client._resolved_model_family == QWEN3_VL_FAMILY
    assert FakeQwen3Processor.from_pretrained_calls == ["Qwen/Qwen3-VL-4B-Instruct"]
    assert FakeModel.from_pretrained_calls == [
        (
            "Qwen/Qwen3-VL-4B-Instruct",
            {"torch_dtype": "auto", "device_map": "auto"},
        )
    ]
    messages, _kwargs = FakeQwen3Processor.qwen3_chat_template_calls[0]
    assert messages[0]["content"] == [
        {"type": "image", "image": str(image)},
        {"type": "text", "text": "Describe this image."},
    ]


def test_qwen_client_accepts_explicit_qwen3_family(monkeypatch, tmp_path):
    image = tmp_path / "example.png"
    image.write_bytes(b"not a real png; tests mock image processing")
    monkeypatch.setattr(qwen_client, "_import_torch", lambda: fake_torch(True))
    monkeypatch.setattr(
        qwen_client,
        "_import_qwen3_libraries",
        lambda: (FakeQwen3Processor, FakeModel),
    )
    client = QwenVisionModelClient(
        model_id="some/local/path",
        model_family=QWEN3_VL_FAMILY,
        max_new_tokens=7,
    )

    assert client.query(str(image), "Describe this image.") == "decoded answer"
    assert client._resolved_model_family == QWEN3_VL_FAMILY


def test_qwen_client_rejects_unknown_family(monkeypatch, tmp_path):
    image = tmp_path / "example.png"
    image.write_bytes(b"image")
    monkeypatch.setattr(qwen_client, "_import_torch", lambda: fake_torch(True))
    client = QwenVisionModelClient(model_id="test/model", model_family="not-real")

    with pytest.raises(ValueError, match="unsupported Qwen model family"):
        client.query(str(image), "Describe this image.")


def test_qwen_client_allows_empty_instruction(monkeypatch, tmp_path):
    image = tmp_path / "example.png"
    image.write_bytes(b"not a real png; tests mock image processing")
    monkeypatch.setattr(qwen_client, "_import_torch", lambda: fake_torch(True))
    monkeypatch.setattr(
        qwen_client,
        "_import_qwen_libraries",
        lambda: (FakeProcessor, FakeModel, fake_process_vision_info),
    )
    client = QwenVisionModelClient(model_id="test/model", max_new_tokens=7)

    assert client.query(str(image), "") == "decoded answer"
    assert FakeProcessor.last_messages[0]["content"] == [
        {"type": "image", "image": str(image)}
    ]


def test_qwen_client_raises_clear_error_without_gpu(monkeypatch, tmp_path):
    image = tmp_path / "example.png"
    image.write_bytes(b"image")
    monkeypatch.setattr(qwen_client, "_import_torch", lambda: fake_torch(False))

    client = QwenVisionModelClient(model_id="test/model")

    with pytest.raises(RuntimeError, match="CUDA/ROCm.*no GPU backend"):
        client.query(str(image), "Describe this image.")


def test_query_model_cli_prints_response(monkeypatch, capsys):
    created_clients = []

    class FakeQwenVisionModelClient:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)

        def query(self, image_path, instruction):
            assert image_path == "data/generated/example.png"
            assert instruction == "Describe this image."
            return "cli response"

    monkeypatch.setattr(query_model, "QwenVisionModelClient", FakeQwenVisionModelClient)

    result = query_model.main(
        [
            "--image",
            "data/generated/example.png",
            "--instruction",
            "Describe this image.",
            "--model-id",
            "Qwen/Qwen2.5-VL-3B-Instruct",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "cli response\n"
    assert created_clients == [
        {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_family": "auto",
            "max_new_tokens": 256,
            "require_gpu": True,
        }
    ]


def test_query_model_cli_accepts_qwen3_family(monkeypatch, capsys):
    created_clients = []

    class FakeQwenVisionModelClient:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)

        def query(self, image_path, instruction):
            return f"{image_path}: {instruction}"

    monkeypatch.setattr(query_model, "QwenVisionModelClient", FakeQwenVisionModelClient)

    result = query_model.main(
        [
            "--image",
            "data/generated/example.png",
            "--instruction",
            "Describe this image.",
            "--model-id",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--model-family",
            "qwen3-vl",
            "--max-new-tokens",
            "32",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "data/generated/example.png: Describe this image.\n"
    assert created_clients == [
        {
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "model_family": "qwen3-vl",
            "max_new_tokens": 32,
            "require_gpu": True,
        }
    ]
