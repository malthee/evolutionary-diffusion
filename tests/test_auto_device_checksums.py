import hashlib

import pytest

from evolutionary_model_helpers import auto_device


def test_calculate_file_sha256_matches_python_hashlib(tmp_path):
    test_file = tmp_path / "sample.bin"
    payload = b"evolutionary-diffusion"
    test_file.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert auto_device.calculate_file_sha256(str(test_file)) == expected


def test_load_torch_model_rejects_cached_file_with_wrong_checksum(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"cached-content")

    download_called = {"value": False}
    load_called = {"value": False}

    def fake_download_url_to_file(*, url_or_request, destination_path, timeout=None):
        download_called["value"] = True
        raise AssertionError("download_url_to_file should not be called for an existing cached file")

    def fake_torch_load(*args, **kwargs):
        load_called["value"] = True
        raise AssertionError("torch.load should not be reached when checksum validation fails")

    monkeypatch.setattr(auto_device, "download_url_to_file", fake_download_url_to_file)
    monkeypatch.setattr(auto_device.torch, "load", fake_torch_load)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        auto_device.load_torch_model(
            model_path=str(model_path),
            url="https://example.com/model.pth",
            expected_sha256="0" * 64,
            strict_checksum=True,
        )

    assert download_called["value"] is False
    assert load_called["value"] is False


def test_load_torch_model_downloads_then_verifies_checksum(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pth"
    payload = b"downloaded-model-bytes"
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    download_called = {"value": False}
    load_called = {"value": False}

    def fake_download_url_to_file(*, url_or_request, destination_path, timeout=None):
        download_called["value"] = True
        with open(destination_path, "wb") as handle:
            handle.write(payload)

    def fake_torch_load(path, **kwargs):
        load_called["value"] = True
        assert path == str(model_path)
        return {"weights": 1}

    monkeypatch.setattr(auto_device, "download_url_to_file", fake_download_url_to_file)
    monkeypatch.setattr(auto_device.torch, "load", fake_torch_load)

    model = auto_device.load_torch_model(
        model_path=str(model_path),
        url="https://example.com/model.pth",
        expected_sha256=expected_sha256,
        strict_checksum=True,
    )

    assert download_called["value"] is True
    assert load_called["value"] is True
    assert model == {"weights": 1}
