import hashlib
import os
import shutil
import ssl
import urllib.request
from typing import Any, Optional
from urllib.error import URLError

import torch

"""
This module contains helper functions to automatically select the best device for torch and
instantiates models and pipelines with it.
"""

INSECURE_SSL_FALLBACK_ENV = "EVOLUTIONARY_DIFFUSION_ALLOW_INSECURE_SSL"


def _is_truthy_env(value: str | None) -> bool:
    """Return True when an environment variable value is explicitly truthy."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _allow_insecure_ssl_fallback() -> bool:
    """
    Return whether insecure SSL retry is explicitly enabled.

    Insecure retry is disabled by default and must be opted in by setting
    EVOLUTIONARY_DIFFUSION_ALLOW_INSECURE_SSL to a truthy value.
    """
    return _is_truthy_env(os.getenv(INSECURE_SSL_FALLBACK_ENV))


def _resolve_url_for_error(url_or_request: Any) -> str:
    """Extract a URL-like string for error messages."""
    return str(getattr(url_or_request, "full_url", url_or_request))


def _build_verified_ssl_context() -> ssl.SSLContext:
    """
    Build an HTTPS context with certificate verification enabled.

    Prefers the certifi CA bundle when available, which avoids platform-specific
    OpenSSL certificate path issues.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _build_certificate_error_help_text(url_or_request: Any) -> str:
    """Create actionable troubleshooting guidance for certificate failures."""
    url = _resolve_url_for_error(url_or_request)
    return (
        f"SSL certificate verification failed while accessing {url}. "
        "Install Python certificates with '/Applications/Python 3.13/Install Certificates.command' "
        "or set SSL_CERT_FILE to a valid CA bundle (for example certifi.where()). "
        f"As a temporary insecure workaround only, set {INSECURE_SSL_FALLBACK_ENV}=1."
    )


def open_url_with_tls(url_or_request: Any, timeout: Optional[float] = None):
    """
    Open a URL with verified TLS and optional explicit insecure fallback.

    :param url_or_request: URL string or urllib Request object.
    :param timeout: Optional timeout in seconds.
    :return: File-like HTTP response object.
    :raises RuntimeError: On certificate failures when insecure fallback is disabled.
    """
    request_kwargs: dict[str, Any] = {"context": _build_verified_ssl_context()}
    if timeout is not None:
        request_kwargs["timeout"] = timeout

    try:
        return urllib.request.urlopen(url_or_request, **request_kwargs)
    except URLError as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error):
            raise
        if not _allow_insecure_ssl_fallback():
            raise RuntimeError(_build_certificate_error_help_text(url_or_request)) from error

        print(
            "Warning: SSL certificate verification failed and insecure fallback is enabled via "
            f"{INSECURE_SSL_FALLBACK_ENV}. Proceeding without certificate verification."
        )
        request_kwargs["context"] = ssl._create_unverified_context()
        return urllib.request.urlopen(url_or_request, **request_kwargs)


def download_url_to_file(url_or_request: Any, destination_path: str, timeout: Optional[float] = None):
    """
    Download a URL or Request to a local file path using verified TLS.

    :param url_or_request: URL string or urllib Request object.
    :param destination_path: Output file path.
    :param timeout: Optional timeout in seconds.
    """
    with open_url_with_tls(url_or_request=url_or_request, timeout=timeout) as source, open(
        destination_path,
        "wb",
    ) as destination:
        shutil.copyfileobj(source, destination)


def auto_clip_device():
    """
    Edit: Causes Problems and has been fixed to CPU on any device. Previously selected CUDA if available.
    """
    return "cpu"


def auto_device():
    """
    Gets the most appropriate device as string in the order "cuda" -> "mps" -> "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def auto_to_device(model_or_pipeline, attention_slicing=False):
    """
    Automatically moves the given PyTorch model or pipeline to the most appropriate device.
    It checks for CUDA or MPS availability and defaults to CPU if neither is available.

    :param model_or_pipeline: A PyTorch model or pipeline.
    :param attention_slicing: If True, attention slicing is enabled (MAC: Recommended if your computer has < 64 GB of RAM, may lead to black images in some cases)
    :return: The model or pipeline moved to the appropriate device.
    """

    device = auto_device()
    pipe = model_or_pipeline.to(device)
    if attention_slicing:
        pipe.enable_attention_slicing()
    return pipe


def auto_generator(seed=None):
    """
    Automatically returns a generator for the installed device.

    :param seed: The seed to use for the generator. If None, the default generator is returned.
    :return: A generator.
    """

    device = auto_device()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def auto_batch_pipeline_arguments(prompt, inference_steps=1, batch_size=1, deterministic=True, guidance_scale=None):
    """
     Returns a dictionary of pipeline arguments for the appropriate installed device.

    :param prompt: The prompt to use for the pipeline.
    :param inference_steps: The number of inference steps to use for the pipeline.
    :param batch_size: How many images to generate in parallel.
    :param deterministic: If True, the pipeline will be deterministic and use a fixed seed for the images.
    :param guidance_scale: The guidance scale to use for the pipeline. If None, default is used.
    :return: A dictionary of pipeline arguments.
    """

    pipeline_args = {
        "prompt": [prompt] * batch_size,
        "num_inference_steps": inference_steps,
    }

    if guidance_scale is not None:
        pipeline_args["guidance_scale"] = guidance_scale

    if deterministic:
        pipeline_args["generator"] = [auto_generator(i) for i in range(batch_size)]

    return pipeline_args


def calculate_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute the SHA-256 checksum of a file.

    :param file_path: Path to the file.
    :param chunk_size: Size of each read chunk in bytes.
    :return: Lowercase hex encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(file_path: str, expected_sha256: str):
    """
    Verify that a file matches the expected SHA-256 checksum.

    :param file_path: Path to the file.
    :param expected_sha256: Expected lowercase or uppercase SHA-256 checksum.
    :raises ValueError: If the checksum does not match.
    """
    normalized_expected = expected_sha256.lower()
    actual = calculate_file_sha256(file_path)
    if actual != normalized_expected:
        raise ValueError(
            f"Checksum mismatch for {file_path}. "
            f"Expected {normalized_expected}, got {actual}."
        )


def load_torch_model(
    model_path,
    url,
    device=auto_device(),
    expected_sha256: Optional[str] = None,
    strict_checksum: bool = True,
):
    """
    Loads a PyTorch model. Downloads the model from a URL if it's not present locally.

    :param model_path: Path where the model is saved or will be saved.
    :param url: URL to download the model if not present locally.
    :param device: Device the model is loaded on.
    :param expected_sha256: Expected SHA-256 checksum of the model file.
    :param strict_checksum: If True, abort loading on checksum mismatch.
    :return: Loaded PyTorch model.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found. Downloading from {url}...")
        # Create the models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Download the file from `url` and save it locally under `model_path`
        download_url_to_file(url_or_request=url, destination_path=model_path)

    if expected_sha256 is not None:
        try:
            verify_file_sha256(model_path, expected_sha256)
        except ValueError:
            if strict_checksum:
                raise
            print("Warning: Model checksum mismatch, continuing because strict_checksum=False.")

    # Load the model
    model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    print("Model loaded successfully.")
    return model
