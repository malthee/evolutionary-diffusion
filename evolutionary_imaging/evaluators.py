import os
import json
import io
import shutil
import sys
import tarfile
import urllib.request
import warnings
import zipfile

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from typing import Union, Tuple, Literal, Dict, Any, Callable, List
from torchmetrics.multimodal import CLIPScore, CLIPImageQualityAssessment
from transformers import pipeline

from evolutionary.evolution_base import SingleObjectiveEvaluator, SingleObjectiveFitness, MultiObjectiveFitness, \
    Evaluator
from evolutionary_imaging.image_base import ImageSolutionData
from evolutionary_model_helpers.auto_device import (
    auto_clip_device,
    auto_device,
    download_url_to_file,
    load_torch_model,
    open_url_with_tls,
    verify_file_sha256,
)
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

_model_cache: Dict[str, Any] = {}
"""
Cache for models used in evaluation to avoid keeping multiple copies in memory.
Can be cleared with `clear_model_cache`.
"""


def clear_model_cache():
    """
    Clear the model cache of all evaluators to free up memory.
    """
    global _model_cache
    _model_cache.clear()


def get_or_create_model(model_id: str, creator: Callable[[], Any]) -> Any:
    """
    Get a model from the cache or create it with the creator function.
    """
    global _model_cache
    if model_id not in _model_cache:
        _model_cache[model_id] = creator()

    return _model_cache[model_id]


def _normalized(a, axis=-1, order=2):
    """
    Utility function for normalizing an array with axis and order.
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def _as_rgb_pil_image(image: Any) -> Image.Image:
    """
    Normalize evaluator image input to a PIL RGB image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image input, received {type(image)!r}.")
    return image if image.mode == "RGB" else image.convert("RGB")


class AestheticsImageEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Aesthetics Predictor V2 from the improved-aesthetic-predictor repository.
    """

    DEFAULT_MODEL_PATH = "./models/sac+logos+ava1-l14-linearMSE.pth"
    MODEL_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14"
                 "-linearMSE.pth")
    MODEL_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
    CLIP_MODEL_NAME = "ViT-L/14"
    CLIP_EMBEDDING_SIZE = 768

    class _MLP(pl.LightningModule):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.layers(x)

    def _setup_model(self, model_path: str):
        predictor_model = load_torch_model(
            model_path=model_path,
            url=AestheticsImageEvaluator.MODEL_URL,
            device=self.device,
            expected_sha256=AestheticsImageEvaluator.MODEL_SHA256,
            strict_checksum=True,
        )
        model = AestheticsImageEvaluator._MLP(input_size=AestheticsImageEvaluator.CLIP_EMBEDDING_SIZE)
        model.load_state_dict(predictor_model)
        model.to(self.device)
        model.eval()
        clip_model, preprocess = clip.load(AestheticsImageEvaluator.CLIP_MODEL_NAME, device=self.device)
        return model, clip_model, preprocess

    def __init__(self,
                 device: torch.device = auto_clip_device(),
                 model_path: str = DEFAULT_MODEL_PATH):
        self.device = device
        self.model, self.clip_model, self.preprocess = get_or_create_model(f"AestheticsImageEvaluator_{model_path}",
                                                                           lambda: self._setup_model(model_path))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        scores = []
        for img in result.images:
            image = self.preprocess(img).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image)
            im_emb_arr = _normalized(image_features.cpu().detach())
            prediction = self.model(im_emb_arr.to(self.device))
            scores.append(prediction.item())
        return np.mean(scores) if scores else 0.0


class AIDetectionImageEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluate the AI-likeliness of an image.
    Maximizes human-likeness.
    """

    SupportedModels = Literal["umm-maybe/AI-image-detector", "Organika/sdxl-detector"]
    """
    The original AI-image-detector was designed for detecting VQGAN+CLIP
    the sdxl-detector was fine-tuned on SDXL generated images.
    """

    def _setup_model(self, model: SupportedModels):
        return pipeline("image-classification", model=model, device=self.device)

    def __init__(self,
                 device: torch.device = auto_clip_device(),
                 model: SupportedModels = "Organika/sdxl-detector"):
        self.device = device
        self.model = get_or_create_model(f"AIDetectionImageEvaluator_{model}",
                                         lambda: self._setup_model(model))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        scores = []
        for img in result.images:
            predictions = self.model(_as_rgb_pil_image(img))
            for pred in predictions:
                if pred["label"] == "human":
                    scores.append(pred["score"] * 100)  # Convert from percentages for better comparison
                    break
        return np.mean(scores) if scores else 0.0


class SSPAIDetectionImageEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluate human-likeliness of images with SSP (Single Simple Patch).

    Uses a simple patch extractor, fixed SRM filters and a binary ResNet-50 classifier.
    """

    SupportedCheckpoints = Literal["adm", "biggan", "glide", "midjourney", "sd4", "sd5", "vqdm", "wukong"]
    PatchSamplingMode = Literal["paper-random", "deterministic"]

    DEFAULT_CHECKPOINT = "sd4"
    CHECKPOINT_FILENAME = "Net_epoch_best.pth"
    DEFAULT_MODELS_DIR = "./models/ssp"
    ARCHIVE_PATH = "./models/ssp/pretrained_checkpoints.zip"
    PRETRAINED_ARCHIVE_SHA256 = "0a74b52144fe5803c2916bd5f0e82be819803eaa17ee009e01280f7a08e78069"
    PRETRAINED_ARCHIVE_URLS = (
        "https://www.dropbox.com/scl/fo/2xdvtew4rjrrsl6cseq30/AElVHGO84W1DSmlImMg1ruM?rlkey=e1a2hnzh62wuuxrnbkfv6f90v&dl=1",
        "https://www.dropbox.com/scl/fo/2xdvtew4rjrrsl6cseq30/AElVHGO84W1DSmlImMg1ruM?rlkey=e1a2hnzh62wuuxrnbkfv6f90v&st=getws7lp&dl=1",
    )
    CHECKPOINT_DIRECTORIES: Dict[SupportedCheckpoints, str] = {
        "adm": "adm",
        "biggan": "biggan",
        "glide": "glide",
        "midjourney": "midjourney",
        "sd4": "sd4",
        "sd5": "sd5",
        "vqdm": "vqdm",
        "wukong": "wukong",
    }
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    PATCH_SIZE = 32
    TRAIN_IMAGE_SIZE = 256

    class _SRMConv2dSimple(nn.Module):
        """Fixed SRM high-pass filters used before the classifier."""

        def __init__(self, in_channels: int = 3):
            super().__init__()
            self._truncate = nn.Hardtanh(-3, 3)
            kernel = self._build_kernel(in_channels)
            self.kernel = nn.Parameter(data=kernel, requires_grad=False)

        @staticmethod
        def _build_kernel(in_channels: int) -> torch.Tensor:
            filter1 = np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=float,
            ) / 4.0
            filter2 = np.asarray(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=float,
            ) / 12.0
            filter3 = np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=float,
            ) / 2.0

            filters = np.array([[filter1], [filter2], [filter3]])
            filters = np.repeat(filters, in_channels, axis=1)
            return torch.FloatTensor(filters)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = nn.functional.conv2d(x, self.kernel, stride=1, padding=2)
            return self._truncate(out)

    class _SSPNetwork(nn.Module):
        """SSP classifier network: SRM + ResNet-50 (binary head)."""

        def __init__(self):
            super().__init__()
            self.srm = SSPAIDetectionImageEvaluator._SRMConv2dSimple()
            self.disc = resnet50(weights=None)
            self.disc.fc = nn.Linear(2048, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = nn.functional.interpolate(x, (256, 256), mode='bilinear')
            x = self.srm(x)
            return self.disc(x)

    def __init__(
        self,
        device: torch.device = auto_device(),
        checkpoint: SupportedCheckpoints = DEFAULT_CHECKPOINT,
        checkpoint_path: str | None = None,
        download_if_missing: bool = True,
        patch_sampling_mode: PatchSamplingMode = "paper-random",
        paper_random_candidate_count: int = 64,
        paper_random_passes: int = 1,
    ):
        """
        Initialize the SSP evaluator.

        Args:
            device: Torch device where inference is executed.
            checkpoint: Name of one of the official pretrained SSP checkpoints.
            checkpoint_path: Optional local path to a custom checkpoint file.
            download_if_missing: If True, downloads the official checkpoint archive when missing.
            patch_sampling_mode: "paper-random" for paper-like random patch sampling or "deterministic".
            paper_random_candidate_count: Number of random 32x32 patch candidates per paper-random pass.
            paper_random_passes: Number of paper-random passes per image. Scores are averaged across passes.
        """
        if patch_sampling_mode not in ("paper-random", "deterministic"):
            raise ValueError(f"Unsupported patch_sampling_mode: {patch_sampling_mode}")
        if paper_random_candidate_count <= 0:
            raise ValueError("paper_random_candidate_count must be > 0.")
        if paper_random_passes <= 0:
            raise ValueError("paper_random_passes must be > 0.")

        self.device = device
        self._patch_sampling_mode = patch_sampling_mode
        self._paper_random_candidate_count = paper_random_candidate_count
        self._paper_random_passes = paper_random_passes
        self._deterministic_patches_per_side = (
            SSPAIDetectionImageEvaluator.TRAIN_IMAGE_SIZE // SSPAIDetectionImageEvaluator.PATCH_SIZE
        )
        self._resize = transforms.Resize(
            (SSPAIDetectionImageEvaluator.TRAIN_IMAGE_SIZE, SSPAIDetectionImageEvaluator.TRAIN_IMAGE_SIZE)
        )
        self._random_crop = transforms.RandomCrop(SSPAIDetectionImageEvaluator.PATCH_SIZE)
        self._preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(SSPAIDetectionImageEvaluator.IMAGENET_MEAN, SSPAIDetectionImageEvaluator.IMAGENET_STD),
            ]
        )

        resolved_checkpoint_path = self._resolve_checkpoint_path(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            download_if_missing=download_if_missing,
        )
        resolved_checkpoint_path = os.path.abspath(resolved_checkpoint_path)

        model_cache_id = f"SSPAIDetectionImageEvaluator_{resolved_checkpoint_path}_{self.device}"
        self.model = get_or_create_model(model_cache_id, lambda: self._setup_model(resolved_checkpoint_path))

    def _setup_model(self, checkpoint_path: str) -> nn.Module:
        model = SSPAIDetectionImageEvaluator._SSPNetwork()
        state_dict = self._load_state_dict(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _load_state_dict(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        loaded = torch.load(checkpoint_path, map_location=torch.device(self.device))

        if not isinstance(loaded, dict):
            raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")

        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state_dict = loaded["state_dict"]
        elif "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

        return state_dict

    def _resolve_checkpoint_path(
        self,
        checkpoint: SupportedCheckpoints,
        checkpoint_path: str | None,
        download_if_missing: bool,
    ) -> str:
        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                return checkpoint_path
            raise FileNotFoundError(f"Custom SSP checkpoint not found: {checkpoint_path}")

        checkpoint_dir = os.path.join(
            SSPAIDetectionImageEvaluator.DEFAULT_MODELS_DIR,
            SSPAIDetectionImageEvaluator.CHECKPOINT_DIRECTORIES[checkpoint],
        )
        resolved_path = os.path.join(checkpoint_dir, SSPAIDetectionImageEvaluator.CHECKPOINT_FILENAME)
        if os.path.exists(resolved_path):
            return resolved_path

        if not download_if_missing:
            raise FileNotFoundError(
                f"Checkpoint not found at {resolved_path}. "
                f"Enable download_if_missing or provide checkpoint_path."
            )

        self._download_archive_if_needed()
        self._extract_checkpoint_from_archive(checkpoint, resolved_path)
        return resolved_path

    def _download_archive_if_needed(self):
        if os.path.exists(SSPAIDetectionImageEvaluator.ARCHIVE_PATH):
            self._verify_archive_checksum()
            return

        os.makedirs(SSPAIDetectionImageEvaluator.DEFAULT_MODELS_DIR, exist_ok=True)
        last_error: Exception | None = None

        for url in SSPAIDetectionImageEvaluator.PRETRAINED_ARCHIVE_URLS:
            try:
                self._download_archive(url)
                return
            except Exception as error:
                last_error = error

        raise RuntimeError(
            "Failed to download SSP pretrained checkpoint archive from official links."
        ) from last_error

    def _verify_archive_checksum(self):
        verify_file_sha256(
            SSPAIDetectionImageEvaluator.ARCHIVE_PATH,
            SSPAIDetectionImageEvaluator.PRETRAINED_ARCHIVE_SHA256,
        )

    def _download_archive(self, url: str):
        destination = SSPAIDetectionImageEvaluator.ARCHIVE_PATH
        try:
            download_url_to_file(url_or_request=url, destination_path=destination)
            self._verify_archive_checksum()
        except Exception:
            if os.path.exists(destination):
                os.remove(destination)
            raise

    def _extract_checkpoint_from_archive(self, checkpoint: SupportedCheckpoints, destination_path: str):
        checkpoint_dir = os.path.dirname(destination_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        target_member = (
            f"{SSPAIDetectionImageEvaluator.CHECKPOINT_DIRECTORIES[checkpoint]}/"
            f"{SSPAIDetectionImageEvaluator.CHECKPOINT_FILENAME}"
        )

        with zipfile.ZipFile(SSPAIDetectionImageEvaluator.ARCHIVE_PATH, "r") as archive:
            members = archive.namelist()
            match = target_member if target_member in members else None
            if match is None:
                for member in members:
                    if member.endswith(target_member):
                        match = member
                        break

            if match is None:
                raise FileNotFoundError(
                    f"Checkpoint {target_member} not found inside archive {SSPAIDetectionImageEvaluator.ARCHIVE_PATH}."
                )

            with archive.open(match) as src, open(destination_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    @staticmethod
    def _patch_diversity(patch) -> float:
        """Compute texture diversity score for a patch (lower means simpler)."""
        patch_array = np.array(patch).astype(np.int64)
        diff_horizontal = np.abs(patch_array[:, :-1, :] - patch_array[:, 1:, :]).sum()
        diff_vertical = np.abs(patch_array[:-1, :, :] - patch_array[1:, :, :]).sum()
        diff_diagonal = np.abs(patch_array[:-1, :-1, :] - patch_array[1:, 1:, :]).sum()
        diff_anti_diagonal = np.abs(patch_array[1:, :-1, :] - patch_array[:-1, 1:, :]).sum()
        return float(diff_horizontal + diff_vertical + diff_diagonal + diff_anti_diagonal)

    def _create_patch_candidates(self, image) -> list:
        patch_size = SSPAIDetectionImageEvaluator.PATCH_SIZE
        if self._patch_sampling_mode == "paper-random":
            return [self._random_crop(image) for _ in range(self._paper_random_candidate_count)]

        width, height = image.size
        max_x = max(0, width - patch_size)
        max_y = max(0, height - patch_size)
        x_positions = np.linspace(0, max_x, num=self._deterministic_patches_per_side, dtype=int)
        y_positions = np.linspace(0, max_y, num=self._deterministic_patches_per_side, dtype=int)

        return [
            image.crop((int(x), int(y), int(x) + patch_size, int(y) + patch_size))
            for y in y_positions
            for x in x_positions
        ]

    def _extract_simplest_patch(self, image):
        image = _as_rgb_pil_image(image)
        if min(image.size) < SSPAIDetectionImageEvaluator.PATCH_SIZE:
            image = self._resize(image)

        patches = self._create_patch_candidates(image)
        return min(patches, key=self._patch_diversity)

    def _score_patch(self, patch) -> float:
        input_tensor = self._preprocess(patch).unsqueeze(0).to(self.device)
        logit = self.model(input_tensor).ravel()[0]
        return float(torch.sigmoid(logit).item() * 100.0)

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        """Return mean human-likeliness in percent for all images in the batch."""
        scores = []
        for image in result.images:
            if self._patch_sampling_mode == "paper-random":
                pass_scores = []
                for _ in range(self._paper_random_passes):
                    simplest_patch = self._extract_simplest_patch(image)
                    pass_scores.append(self._score_patch(simplest_patch))
                scores.append(float(np.mean(pass_scores)))
            else:
                simplest_patch = self._extract_simplest_patch(image)
                scores.append(self._score_patch(simplest_patch))
        return float(np.mean(scores)) if scores else 0.0


class DIREAIDetectionImageEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluate human-likeliness of images with DIRE (Diffusion Reconstruction Error).

    The default backend follows the official DIRE setup: ADM-based DDIM inversion/reconstruction,
    absolute residual DIRE, and a ResNet-50 binary classifier on DIRE images.
    """

    SupportedBackends = Literal["adm-ddim-official", "sdxl-turbo-experimental"]
    SupportedReconstructionModels = Literal["lsun_bedroom", "imagenet_uncond"]
    SupportedClassifierCheckpoints = Literal["lsun_adm", "imagenet_adm"]
    OfficialCodecMode = Literal["match-source-format", "none"]

    DEFAULT_BACKEND: SupportedBackends = "adm-ddim-official"
    DEFAULT_RECONSTRUCTION_MODEL: SupportedReconstructionModels = "lsun_bedroom"
    DEFAULT_CLASSIFIER_CHECKPOINT: SupportedClassifierCheckpoints = "lsun_adm"
    DEFAULT_OFFICIAL_CODEC_MODE: OfficialCodecMode = "match-source-format"
    DEFAULT_DDIM_STEPS = 20
    DEFAULT_MODELS_DIR = "./models/dire"
    DEFAULT_RECONSTRUCTION_DIR = "./models/dire/reconstruction"
    DEFAULT_CLASSIFIER_DIR = "./models/dire/classifier"
    DEFAULT_GUIDED_DIFFUSION_DIR = "./models/dire/guided-diffusion"
    GUIDED_DIFFUSION_ARCHIVE_PATH = "./models/dire/guided-diffusion/dire-main.zip"
    GUIDED_DIFFUSION_ARCHIVE_URL = "https://github.com/ZhendongWang6/DIRE/archive/refs/heads/main.zip"

    RECONSTRUCTION_MODELS: Dict[SupportedReconstructionModels, Dict[str, str]] = {
        "lsun_bedroom": {
            "filename": "lsun_bedroom.pt",
            "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt",
        },
        "imagenet_uncond": {
            "filename": "256x256_diffusion_uncond.pt",
            "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
        },
    }
    CLASSIFIER_CHECKPOINT_FILES: Dict[SupportedClassifierCheckpoints, str] = {
        "lsun_adm": "lsun_adm.pth",
        "imagenet_adm": "imagenet_adm.pth",
    }
    CLASSIFIER_CHECKPOINT_SHA256: Dict[SupportedClassifierCheckpoints, str] = {
        "lsun_adm": "e61a33a6066f23771f75aa717da8e6d7c7ecea6daa3110108368990f55e3e523",
        "imagenet_adm": "09ffb75d8e597702be8462bd5249978c1a5a427f1b481efd3cdb2deb83036dcb",
    }
    RECDRIVE_API_BASE_URL = "https://recapi.ustc.edu.cn/api/v2"
    RECDRIVE_SHARE_NUMBER = "ec980150-4615-11ee-be0a-eb822f25e070"
    RECDRIVE_SHARE_PASSWORD = "dire"
    RECDRIVE_REQUEST_TIMEOUT_SECONDS = 120
    CLASSIFIER_DOWNLOAD_URLS = (
        "https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070",
        "https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg",
        "https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EtKXrn4cjWtBi0H3v4j1ICsBKraCxnZiTWU4VzqRr0ilCw?e=trkgDR",
    )

    _OFFICIAL_MODEL_FLAGS = {
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "diffusion_steps": 1000,
        "dropout": 0.1,
        "image_size": 256,
        "learn_sigma": True,
        "noise_schedule": "linear",
        "num_channels": 256,
        "num_head_channels": 64,
        "num_res_blocks": 2,
        "resblock_updown": True,
        "use_scale_shift_norm": True,
    }
    _CLASSIFIER_MEAN = (0.485, 0.456, 0.406)
    _CLASSIFIER_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        device: torch.device = auto_device(),
        backend: SupportedBackends = DEFAULT_BACKEND,
        ddim_steps: int = DEFAULT_DDIM_STEPS,
        reconstruction_model: SupportedReconstructionModels = DEFAULT_RECONSTRUCTION_MODEL,
        classifier_checkpoint: SupportedClassifierCheckpoints = DEFAULT_CLASSIFIER_CHECKPOINT,
        classifier_checkpoint_path: str | None = None,
        reconstruction_model_path: str | None = None,
        download_if_missing: bool = True,
        official_codec_mode: OfficialCodecMode = DEFAULT_OFFICIAL_CODEC_MODE,
        sdxl_model_id: str = "stabilityai/sdxl-turbo",
        sdxl_num_inference_steps: int = 4,
        sdxl_strength: float = 0.35,
    ):
        """
        Initialize the DIRE evaluator.

        Args:
            device: Torch device used for inference.
            backend: "adm-ddim-official" for paper-faithful DIRE, "sdxl-turbo-experimental" for proxy mode.
            ddim_steps: Number of DDIM steps for official inversion/reconstruction (paper default is 20).
            reconstruction_model: ADM reconstruction checkpoint choice.
            classifier_checkpoint: Pretrained DIRE classifier checkpoint alias.
                Default is "lsun_adm" for strict paper-faithful LSUN setup;
                use "imagenet_adm" for broader natural-image coverage.
            classifier_checkpoint_path: Optional local override for classifier checkpoint.
            reconstruction_model_path: Optional local override for reconstruction model checkpoint.
            download_if_missing: If True, download missing reconstruction/classifier assets when possible.
            official_codec_mode:
                How DIRE tensors are converted before classifier inference in official mode.
                "match-source-format" emulates released dataset preprocessing behavior (.jpg -> JPEG encode/decode).
                "none" disables codec emulation for codec-agnostic custom evaluations.
                Default is "match-source-format" because the released lsun_adm classifier behavior is tied to this
                preprocessing path; use "none" explicitly for custom codec-controlled studies.
            sdxl_model_id: SDXL-Turbo model id for experimental backend.
            sdxl_num_inference_steps: Number of img2img denoising steps for experimental backend.
            sdxl_strength: Img2img strength for experimental backend.
        """
        if backend not in ("adm-ddim-official", "sdxl-turbo-experimental"):
            raise ValueError(f"Unsupported backend: {backend}")
        if official_codec_mode not in ("match-source-format", "none"):
            raise ValueError(f"Unsupported official_codec_mode: {official_codec_mode}")
        if ddim_steps <= 0:
            raise ValueError("ddim_steps must be > 0.")
        if sdxl_num_inference_steps <= 0:
            raise ValueError("sdxl_num_inference_steps must be > 0.")
        if not (0.0 < sdxl_strength <= 1.0):
            raise ValueError("sdxl_strength must be in (0.0, 1.0].")

        self.backend = backend
        self.requested_device = str(device)
        self.device = self.requested_device
        self.ddim_steps = int(ddim_steps)
        self.sdxl_model_id = sdxl_model_id
        self.sdxl_num_inference_steps = sdxl_num_inference_steps
        self.sdxl_strength = sdxl_strength
        self.official_codec_mode = official_codec_mode
        self._download_if_missing = bool(download_if_missing)
        self._official_cpu_fallback_applied = False
        self._codec_domain_warning_emitted = False

        self._classifier_preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self._CLASSIFIER_MEAN, self._CLASSIFIER_STD),
            ]
        )
        self._to_tensor = transforms.ToTensor()

        self._classifier_checkpoint_path = self._resolve_classifier_checkpoint_path(
            checkpoint=classifier_checkpoint,
            classifier_checkpoint_path=classifier_checkpoint_path,
            download_if_missing=download_if_missing,
        )
        self._classifier = self._load_classifier_for_current_device()

        if self.backend == "adm-ddim-official":
            self._reconstruction_model_path = self._resolve_reconstruction_model_path(
                reconstruction_model=reconstruction_model,
                reconstruction_model_path=reconstruction_model_path,
                download_if_missing=download_if_missing,
            )
            self._official_model, self._official_diffusion = self._load_official_components_for_current_device()
            self._sdxl_pipeline = None
        else:
            warnings.warn(
                "DIRE backend 'sdxl-turbo-experimental' is not paper-faithful and is for comparison only.",
                RuntimeWarning,
                stacklevel=2,
            )
            sdxl_cache_id = (
                f"DIRE_sdxl_proxy_{self.sdxl_model_id}_{self.device}_{self.sdxl_num_inference_steps}_"
                f"{self.sdxl_strength}"
            )
            self._sdxl_pipeline = get_or_create_model(
                sdxl_cache_id,
                lambda: self._setup_sdxl_proxy_pipeline(self.sdxl_model_id),
            )
            self._reconstruction_model_path = reconstruction_model_path or ""
            self._official_model, self._official_diffusion = None, None

    def __getstate__(self):
        """
        Serialize only configuration/state and drop heavyweight runtime model objects.

        This avoids pickling large tensors and non-picklable runtime hooks from
        diffusers/transformers pipelines (notably in SDXL experimental mode).
        """
        state = self.__dict__.copy()
        for runtime_key in (
            "_classifier",
            "_official_model",
            "_official_diffusion",
            "_sdxl_pipeline",
        ):
            state.pop(runtime_key, None)
        return state

    def __setstate__(self, state):
        """
        Restore evaluator configuration and rebuild runtime model objects.
        """
        self.__dict__.update(state)

        self._classifier = self._load_classifier_for_current_device()
        if self.backend == "adm-ddim-official":
            self._official_model, self._official_diffusion = self._load_official_components_for_current_device()
            self._sdxl_pipeline = None
        else:
            sdxl_cache_id = (
                f"DIRE_sdxl_proxy_{self.sdxl_model_id}_{self.device}_{self.sdxl_num_inference_steps}_"
                f"{self.sdxl_strength}"
            )
            self._sdxl_pipeline = get_or_create_model(
                sdxl_cache_id,
                lambda: self._setup_sdxl_proxy_pipeline(self.sdxl_model_id),
            )
            self._official_model, self._official_diffusion = None, None

    def _classifier_cache_id_for_current_device(self) -> str:
        return f"DIRE_classifier_{os.path.abspath(self._classifier_checkpoint_path)}_{self.device}"

    def _official_cache_id_for_current_device(self) -> str:
        return (
            f"DIRE_official_{os.path.abspath(self._reconstruction_model_path)}_"
            f"ddim{self.ddim_steps}_{self.device}"
        )

    def _load_classifier_for_current_device(self) -> nn.Module:
        return get_or_create_model(
            self._classifier_cache_id_for_current_device(),
            lambda: self._load_dire_classifier(self._classifier_checkpoint_path),
        )

    def _load_official_components_for_current_device(self) -> tuple[nn.Module, Any]:
        return get_or_create_model(
            self._official_cache_id_for_current_device(),
            lambda: self._setup_official_reconstruction_model(
                model_path=self._reconstruction_model_path,
                ddim_steps=self.ddim_steps,
                download_if_missing=self._download_if_missing,
            ),
        )

    def _resolve_reconstruction_model_path(
        self,
        reconstruction_model: SupportedReconstructionModels,
        reconstruction_model_path: str | None,
        download_if_missing: bool,
    ) -> str:
        if reconstruction_model_path is not None:
            if os.path.exists(reconstruction_model_path):
                return reconstruction_model_path
            raise FileNotFoundError(f"Custom reconstruction model not found: {reconstruction_model_path}")

        model_info = self.RECONSTRUCTION_MODELS[reconstruction_model]
        destination = os.path.join(self.DEFAULT_RECONSTRUCTION_DIR, model_info["filename"])
        if os.path.exists(destination):
            return destination

        if not download_if_missing:
            raise FileNotFoundError(
                f"Reconstruction model missing at {destination}. "
                f"Enable download_if_missing or provide reconstruction_model_path."
            )

        os.makedirs(self.DEFAULT_RECONSTRUCTION_DIR, exist_ok=True)
        download_url_to_file(url_or_request=model_info["url"], destination_path=destination)
        return destination

    def _resolve_classifier_checkpoint_path(
        self,
        checkpoint: SupportedClassifierCheckpoints,
        classifier_checkpoint_path: str | None,
        download_if_missing: bool,
    ) -> str:
        if classifier_checkpoint_path is not None:
            if os.path.exists(classifier_checkpoint_path):
                return classifier_checkpoint_path
            raise FileNotFoundError(f"Custom DIRE classifier checkpoint not found: {classifier_checkpoint_path}")

        destination = os.path.join(self.DEFAULT_CLASSIFIER_DIR, self.CLASSIFIER_CHECKPOINT_FILES[checkpoint])
        if os.path.exists(destination):
            self._verify_classifier_checkpoint_checksum(checkpoint=checkpoint, checkpoint_path=destination)
            return destination

        if not download_if_missing:
            raise FileNotFoundError(
                f"DIRE classifier checkpoint missing at {destination}. "
                f"Set download_if_missing=True or provide classifier_checkpoint_path."
            )

        try:
            self._attempt_classifier_download(checkpoint=checkpoint, destination=destination)
        except Exception as error:
            raise FileNotFoundError(
                "Failed to auto-resolve DIRE classifier checkpoint from official shares. "
                "Please download it manually and pass classifier_checkpoint_path."
            ) from error

        if not os.path.exists(destination):
            raise FileNotFoundError(
                "DIRE classifier checkpoint could not be resolved automatically. "
                "Please provide classifier_checkpoint_path."
            )
        self._verify_classifier_checkpoint_checksum(checkpoint=checkpoint, checkpoint_path=destination)
        return destination

    def _verify_classifier_checkpoint_checksum(
        self,
        checkpoint: SupportedClassifierCheckpoints,
        checkpoint_path: str,
    ):
        """
        Verify checksum for official DIRE classifier checkpoints.
        """
        expected_sha256 = self.CLASSIFIER_CHECKPOINT_SHA256.get(checkpoint)
        if expected_sha256 is None:
            return
        verify_file_sha256(checkpoint_path, expected_sha256)

    def _attempt_classifier_download(self, checkpoint: SupportedClassifierCheckpoints, destination: str):
        os.makedirs(self.DEFAULT_CLASSIFIER_DIR, exist_ok=True)
        expected_name = self.CLASSIFIER_CHECKPOINT_FILES[checkpoint]
        temp_file = f"{destination}.download"
        last_error: Exception | None = None

        try:
            if self._attempt_classifier_download_from_recdrive(checkpoint=checkpoint, destination=destination):
                return
        except Exception as error:
            last_error = error

        for url in self.CLASSIFIER_DOWNLOAD_URLS:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                download_url_to_file(url_or_request=url, destination_path=temp_file)

                if self._extract_checkpoint_from_archive(temp_file, expected_name, destination):
                    return
                if self._is_probably_html(temp_file):
                    raise ValueError("Download resolved to an HTML page, not a checkpoint file.")
                if self._is_torch_checkpoint(temp_file):
                    shutil.move(temp_file, destination)
                    return
                raise ValueError("Downloaded artifact is neither archive checkpoint nor raw torch checkpoint.")
            except Exception as error:
                last_error = error
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        raise RuntimeError("Could not download DIRE classifier checkpoint from configured sources.") from last_error

    def _recdrive_post(self, endpoint: str, payload: dict) -> dict:
        """
        POST JSON payload to RecDrive API and return decoded JSON response.
        """
        url = f"{self.RECDRIVE_API_BASE_URL}{endpoint}"
        request_data = json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json", "User-Agent": "evolutionary-diffusion/DIRE"}
        request = urllib.request.Request(url=url, data=request_data, headers=request_headers, method="POST")
        with open_url_with_tls(url_or_request=request, timeout=self.RECDRIVE_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8-sig")
        parsed = json.loads(body)
        if parsed.get("status_code") != 200:
            raise RuntimeError(f"RecDrive API error on {endpoint}: {parsed.get('status_code')} {parsed.get('message')}")
        return parsed

    def _resolve_recdrive_checkpoint_number(self, checkpoint: SupportedClassifierCheckpoints) -> str | None:
        """
        Resolve checkpoint file id from the public RecDrive share using a bounded BFS.
        """
        target_filename = self.CLASSIFIER_CHECKPOINT_FILES[checkpoint]
        target_basename = os.path.splitext(target_filename)[0].lower()
        target_ext = os.path.splitext(target_filename)[1].removeprefix(".").lower()

        queue: list[str | None] = [None]
        visited: set[str | None] = set()
        max_folders = 400

        while queue and len(visited) < max_folders:
            folder_number = queue.pop(0)
            if folder_number in visited:
                continue
            visited.add(folder_number)

            payload = {
                "share_number": self.RECDRIVE_SHARE_NUMBER,
                "share_resource_number": folder_number,
                "is_rec": "false",
                "share_constraint": {"password": self.RECDRIVE_SHARE_PASSWORD},
            }
            items = self._recdrive_post("/share/target/resource/list", payload).get("entity", [])
            for item in items:
                item_type = item.get("type")
                item_number = item.get("number")
                if not item_number:
                    continue
                if item_type == "folder":
                    queue.append(item_number)
                    continue
                if item_type != "file":
                    continue

                item_name = str(item.get("name", "")).lower()
                item_ext = str(item.get("file_ext", "")).lower()
                full_name = f"{item_name}.{item_ext}" if item_ext else item_name

                if (
                    full_name == target_filename.lower()
                    or (item_name == target_basename and item_ext == target_ext)
                ):
                    return str(item_number)
        return None

    def _attempt_classifier_download_from_recdrive(
        self,
        checkpoint: SupportedClassifierCheckpoints,
        destination: str,
    ) -> bool:
        """
        Download classifier checkpoint from the public RecDrive API if available.
        """
        resource_number = self._resolve_recdrive_checkpoint_number(checkpoint)
        if resource_number is None:
            return False

        payload = {
            "share_number": self.RECDRIVE_SHARE_NUMBER,
            "share_constraint": {"password": self.RECDRIVE_SHARE_PASSWORD},
            "share_resources_list": [resource_number],
        }
        response = self._recdrive_post("/share/download", payload)
        download_url = response.get("entity", {}).get(resource_number)
        if not download_url:
            return False

        expected_name = self.CLASSIFIER_CHECKPOINT_FILES[checkpoint]
        temp_file = f"{destination}.recdownload"
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            download_url_to_file(
                url_or_request=download_url,
                destination_path=temp_file,
                timeout=self.RECDRIVE_REQUEST_TIMEOUT_SECONDS,
            )

            if self._extract_checkpoint_from_archive(temp_file, expected_name, destination):
                return True
            if self._is_probably_html(temp_file):
                raise ValueError("RecDrive download resolved to HTML, not a checkpoint.")
            if self._is_torch_checkpoint(temp_file):
                shutil.move(temp_file, destination)
                return True
            raise ValueError("RecDrive artifact is neither archive checkpoint nor raw torch checkpoint.")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    @staticmethod
    def _extract_checkpoint_from_archive(archive_path: str, expected_name: str, destination: str) -> bool:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as archive:
                for member in archive.namelist():
                    if member.endswith(expected_name):
                        with archive.open(member) as src, open(destination, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        return True

        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as archive:
                for member in archive.getmembers():
                    if member.isfile() and member.name.endswith(expected_name):
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            continue
                        with extracted as src, open(destination, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        return True
        return False

    @staticmethod
    def _is_probably_html(file_path: str) -> bool:
        with open(file_path, "rb") as file_handle:
            head = file_handle.read(2048).lower()
        return b"<html" in head or b"<!doctype html" in head

    def _is_torch_checkpoint(self, file_path: str) -> bool:
        try:
            _ = torch.load(file_path, map_location="cpu")
            return True
        except Exception:
            return False

    def _load_dire_classifier(self, checkpoint_path: str) -> nn.Module:
        model = resnet50(weights=None)
        model.fc = nn.Linear(2048, 1)

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        elif not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported classifier checkpoint format: {checkpoint_path}")

        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _setup_official_reconstruction_model(
        self,
        model_path: str,
        ddim_steps: int,
        download_if_missing: bool,
    ) -> tuple[nn.Module, Any]:
        script_util = self._import_guided_diffusion(download_if_missing=download_if_missing)
        defaults = script_util.model_and_diffusion_defaults()
        defaults.update(self._OFFICIAL_MODEL_FLAGS)
        defaults["timestep_respacing"] = f"ddim{ddim_steps}"
        defaults["use_ddim"] = True
        defaults["use_fp16"] = self.device.startswith("cuda")

        model, diffusion = script_util.create_model_and_diffusion(
            **{
                key: defaults[key]
                for key in script_util.model_and_diffusion_defaults().keys()
            }
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(self.device)
        if defaults["use_fp16"]:
            model.convert_to_fp16()
        model.eval()
        return model, diffusion

    def _patch_guided_diffusion_extract_for_mps(self, gaussian_diffusion: Any):
        """
        Patch guided-diffusion extraction to avoid float64 tensor creation on MPS.
        """
        if not (self.backend == "adm-ddim-official" and self.requested_device.startswith("mps")):
            return

        if getattr(gaussian_diffusion, "_evolutionary_mps_float32_patch", False):
            return

        def _extract_into_tensor_float32(arr, timesteps, broadcast_shape):
            arr_np = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
            arr_tensor = torch.from_numpy(arr_np).to(dtype=torch.float32)
            indices = timesteps.detach().to(device="cpu", dtype=torch.long)
            result = arr_tensor[indices].to(device=timesteps.device, dtype=torch.float32)
            while len(result.shape) < len(broadcast_shape):
                result = result[..., None]
            return result.expand(broadcast_shape)

        gaussian_diffusion._extract_into_tensor = _extract_into_tensor_float32
        gaussian_diffusion._evolutionary_mps_float32_patch = True

    def _import_guided_diffusion(self, download_if_missing: bool):
        try:
            from guided_diffusion import gaussian_diffusion, script_util
            if hasattr(gaussian_diffusion.GaussianDiffusion, "ddim_reverse_sample_loop"):
                self._patch_guided_diffusion_extract_for_mps(gaussian_diffusion)
                return script_util
        except ImportError:
            pass

        local_repo_root = os.path.join(self.DEFAULT_GUIDED_DIFFUSION_DIR, "DIRE-main", "guided-diffusion")
        if os.path.isdir(local_repo_root):
            if local_repo_root not in sys.path:
                sys.path.insert(0, local_repo_root)
            try:
                from guided_diffusion import gaussian_diffusion, script_util
                if hasattr(gaussian_diffusion.GaussianDiffusion, "ddim_reverse_sample_loop"):
                    self._patch_guided_diffusion_extract_for_mps(gaussian_diffusion)
                    return script_util
            except ImportError:
                pass

        if not download_if_missing:
            raise ImportError(
                "guided_diffusion is required for backend='adm-ddim-official'. "
                "Install from https://github.com/ZhendongWang6/DIRE or enable download_if_missing."
            )

        os.makedirs(self.DEFAULT_GUIDED_DIFFUSION_DIR, exist_ok=True)
        download_url_to_file(
            url_or_request=self.GUIDED_DIFFUSION_ARCHIVE_URL,
            destination_path=self.GUIDED_DIFFUSION_ARCHIVE_PATH,
        )
        with zipfile.ZipFile(self.GUIDED_DIFFUSION_ARCHIVE_PATH, "r") as archive:
            archive.extractall(self.DEFAULT_GUIDED_DIFFUSION_DIR)

        if local_repo_root not in sys.path:
            sys.path.insert(0, local_repo_root)
        from guided_diffusion import gaussian_diffusion, script_util
        if not hasattr(gaussian_diffusion.GaussianDiffusion, "ddim_reverse_sample_loop"):
            raise ImportError(
                "Resolved guided_diffusion package does not include ddim_reverse_sample_loop required by DIRE."
            )
        self._patch_guided_diffusion_extract_for_mps(gaussian_diffusion)
        return script_util

    @staticmethod
    def _center_crop_arr_like_guided_diffusion(image: Image.Image, image_size: int = 256) -> np.ndarray:
        """
        Match guided-diffusion center-crop preprocessing used by official DIRE scripts.
        """
        pil_image = image
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size),
                resample=Image.Resampling.BOX,
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size),
            resample=Image.Resampling.BICUBIC,
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

    @staticmethod
    def _center_crop_square(image):
        width, height = image.size
        if width == height:
            return image
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))

    def _prepare_official_input(self, image: Any) -> torch.Tensor:
        image = _as_rgb_pil_image(image)
        arr = self._center_crop_arr_like_guided_diffusion(image=image, image_size=256)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 127.5 - 1.0
        return tensor

    def _compute_dire_official(self, image: Any) -> torch.Tensor:
        """
        Compute DIRE in paper-faithful mode using ADM + DDIM inversion/reconstruction.
        """
        batch = self._prepare_official_input(image).unsqueeze(0).to(self.device)
        model_kwargs = {}
        reverse_fn = self._official_diffusion.ddim_reverse_sample_loop
        latent = reverse_fn(
            self._official_model,
            (1, 3, 256, 256),
            noise=batch,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            real_step=0,
        )
        recons = self._official_diffusion.ddim_sample_loop(
            self._official_model,
            (1, 3, 256, 256),
            noise=latent,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            real_step=0,
        )
        dire = torch.abs(batch - recons)
        return (dire / 2.0).clamp(0.0, 1.0).squeeze(0).detach().cpu()

    @staticmethod
    def _is_known_mps_float64_error(error: Exception) -> bool:
        message = str(error).lower()
        return "mps" in message and "float64" in message and ("cannot convert" in message or "doesn't support" in message)

    def _switch_official_runtime_to_cpu(self):
        """
        Switch official DIRE runtime from MPS to CPU after a known MPS dtype failure.
        """
        if self._official_cpu_fallback_applied:
            return

        self._official_cpu_fallback_applied = True
        self.device = "cpu"
        warnings.warn(
            "DIRE official backend fell back to CPU after an MPS dtype issue.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._classifier = self._load_classifier_for_current_device()
        self._official_model, self._official_diffusion = self._load_official_components_for_current_device()

    def _compute_dire_tensor_with_auto_fallback(self, image: Any) -> torch.Tensor:
        """
        Compute DIRE tensor and transparently retry on CPU once if MPS float64 issues occur.
        """
        if self.backend != "adm-ddim-official":
            return self._compute_dire_sdxl_proxy(image)

        try:
            return self._compute_dire_official(image)
        except Exception as error:
            if (
                self.device.startswith("mps")
                and self._is_known_mps_float64_error(error)
                and not self._official_cpu_fallback_applied
            ):
                self._switch_official_runtime_to_cpu()
                return self._compute_dire_official(image)
            raise

    def _setup_sdxl_proxy_pipeline(self, model_id: str):
        from diffusers import AutoPipelineForImage2Image

        use_fp16 = self.device.startswith("cuda") or self.device.startswith("mps")
        torch_dtype = torch.float16 if use_fp16 else torch.float32

        if use_fp16:
            try:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    variant="fp16",
                )
            except Exception:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                )
        else:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
            )
        pipe = pipe.to(self.device)
        return pipe

    def _compute_dire_sdxl_proxy(self, image: Any) -> torch.Tensor:
        """
        Compute DIRE proxy via SDXL-Turbo img2img denoising (experimental, non-paper-faithful).
        """
        source = _as_rgb_pil_image(image)
        source = self._center_crop_square(source)
        source_512 = source.resize((512, 512), resample=Image.Resampling.BICUBIC)
        recon = self._sdxl_pipeline(
            prompt="",
            image=source_512,
            guidance_scale=0.0,
            num_inference_steps=self.sdxl_num_inference_steps,
            strength=self.sdxl_strength,
        ).images[0]
        recon = _as_rgb_pil_image(recon)

        source_256 = source_512.resize((256, 256), resample=Image.Resampling.BICUBIC)
        recon_256 = recon.resize((256, 256), resample=Image.Resampling.BICUBIC)
        source_tensor = self._to_tensor(source_256)
        recon_tensor = self._to_tensor(recon_256)
        return torch.abs(source_tensor - recon_tensor).clamp(0.0, 1.0).cpu()

    def _score_dire_tensor(self, dire_tensor: torch.Tensor) -> float:
        """
        Classify a DIRE image and return human-likeliness percent in [0, 100].
        """
        human_score, _, _ = self._score_dire_tensor_with_details(dire_tensor, source_format=None)
        return human_score

    @staticmethod
    def _infer_source_format(image: Any) -> str | None:
        """
        Infer source format from PIL metadata or filename extension.
        """
        if not isinstance(image, Image.Image):
            return None
        fmt = (image.format or "").strip().lower()
        if fmt:
            return fmt
        # PIL drops `format` after convert("RGB"), but JPEG keeps JFIF markers in `info`.
        info = getattr(image, "info", {}) or {}
        if any(key.lower().startswith("jfif") for key in info.keys()):
            return "jpeg"
        filename = str(getattr(image, "filename", "") or "")
        if filename:
            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            if ext:
                return ext
        return None

    @staticmethod
    def _encode_decode_jpeg(image: Image.Image, quality: int = 95) -> Image.Image:
        """
        Reproduce JPEG codec artifacts used when DIRE images are saved as .jpg.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")

    def _apply_official_codec_mode(self, dire_image: Image.Image, source_format: str | None) -> Image.Image:
        """
        Apply official dataset-like codec behavior before classifier preprocessing.
        """
        if self.backend != "adm-ddim-official":
            return dire_image
        if self.official_codec_mode == "none":
            return dire_image
        if source_format in ("jpg", "jpeg", "jfif"):
            return self._encode_decode_jpeg(dire_image, quality=95)
        return dire_image

    def _warn_if_codec_domain_mismatch(self, source_format: str | None):
        """
        Emit one-time guidance when released lsun_adm defaults are applied to PNG/unknown sources.
        """
        if self._codec_domain_warning_emitted:
            return
        if self.backend != "adm-ddim-official":
            return
        if self.official_codec_mode != "match-source-format":
            return
        checkpoint_name = os.path.basename(self._classifier_checkpoint_path).lower()
        if checkpoint_name != "lsun_adm.pth":
            return
        if source_format in ("png", None):
            warnings.warn(
                "DIRE official lsun_adm checkpoint was released on LSUN data where real samples are mostly JPEG "
                "and synthetic samples are PNG. PNG-only or unknown-format inputs may be biased toward synthetic "
                "scores; prefer domain-matched checkpoints/datasets when available.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._codec_domain_warning_emitted = True

    def _score_dire_tensor_with_details(
        self,
        dire_tensor: torch.Tensor,
        source_format: str | None,
    ) -> tuple[float, float, float]:
        """
        Classify a DIRE image and return (human_score, synthetic_probability, logit).
        """
        self._warn_if_codec_domain_mismatch(source_format=source_format)
        dire_image = to_pil_image(dire_tensor.clamp(0.0, 1.0))
        dire_image = self._apply_official_codec_mode(dire_image=dire_image, source_format=source_format)
        input_tensor = self._classifier_preprocess(dire_image).unsqueeze(0).to(self.device)
        logit = self._classifier(input_tensor).ravel()[0]
        logit_value = float(logit.detach().cpu().item())
        # Keep probability computation in float64 to avoid precision saturation at large |logit|.
        if logit_value >= 0:
            exp_term = np.exp(-logit_value)
            synthetic_probability = float(1.0 / (1.0 + exp_term))
        else:
            exp_term = np.exp(logit_value)
            synthetic_probability = float(exp_term / (1.0 + exp_term))
        human_score = float((1.0 - synthetic_probability) * 100.0)
        return human_score, synthetic_probability, logit_value

    @torch.no_grad()
    def evaluate_image_with_diagnostics(self, image: Any) -> Dict[str, Any]:
        """
        Evaluate one image and return score diagnostics for analysis and notebook reporting.
        """
        source_format = self._infer_source_format(image)
        dire_tensor = self._compute_dire_tensor_with_auto_fallback(image)
        human_score, synthetic_probability, logit_value = self._score_dire_tensor_with_details(
            dire_tensor=dire_tensor,
            source_format=source_format,
        )
        return {
            "backend": self.backend,
            "requested_device": self.requested_device,
            "runtime_device": self.device,
            "source_format": source_format,
            "used_cpu_fallback": self._official_cpu_fallback_applied,
            "human_score": human_score,
            "synthetic_probability": synthetic_probability,
            "logit": logit_value,
            "dire_min": float(dire_tensor.min().item()),
            "dire_max": float(dire_tensor.max().item()),
            "dire_mean": float(dire_tensor.mean().item()),
        }

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        """
        Return mean human-likeliness percent (0..100) for a batch of images.
        """
        scores = []
        for image in result.images:
            source_format = self._infer_source_format(image)
            dire_tensor = self._compute_dire_tensor_with_auto_fallback(image)
            human_score, _, _ = self._score_dire_tensor_with_details(
                dire_tensor=dire_tensor,
                source_format=source_format,
            )
            scores.append(human_score)
        return float(np.mean(scores)) if scores else 0.0


class CLIPScoreEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluates the CLIP score of an image.
    https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html
    Supported CLIP models are defined in `SupportedCLIPModels`.
    """

    SupportedCLIPModels = Literal["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14"]

    def _setup_model(self, model_name_or_path: SupportedCLIPModels):
        return CLIPScore(model_name_or_path=model_name_or_path)

    def __init__(self, prompt: str, clip_model: SupportedCLIPModels = "openai/clip-vit-base-patch16"):
        self._prompt = prompt
        self._model = get_or_create_model(f"CLIPScoreEvaluator_{clip_model}",
                                          lambda: self._setup_model(clip_model))

    @staticmethod
    def _feature_tensor(features: Any) -> torch.Tensor:
        """
        Return the projected CLIP feature tensor from TorchMetrics/Transformers outputs.

        Transformers 5 returns a BaseModelOutputWithPooling from
        `CLIPModel.get_*_features`, with the projected features in
        `pooler_output`. Older versions returned the tensor directly.
        """
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        if isinstance(features, tuple) and features and isinstance(features[0], torch.Tensor):
            return features[0]
        raise TypeError(f"Unsupported CLIP feature output type: {type(features)!r}")

    @torch.no_grad()
    def _score_images(self, images: List[Image.Image]) -> List[float]:
        """
        Evaluate images against the configured prompt with CLIPScore semantics.
        """
        if not images:
            return []

        metric = self._model
        model = metric.model
        processor = metric.processor
        device = next(model.parameters()).device
        image_tensors = [pil_to_tensor(_as_rgb_pil_image(image)).cpu() for image in images]

        image_inputs = processor(images=image_tensors, return_tensors="pt", padding=True)
        text_inputs = processor(text=[self._prompt] * len(images), return_tensors="pt", padding=True)

        image_features = self._feature_tensor(
            model.get_image_features(image_inputs["pixel_values"].to(device))
        )
        text_features = self._feature_tensor(
            model.get_text_features(
                text_inputs["input_ids"].to(device),
                text_inputs["attention_mask"].to(device),
            )
        )

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        score = 100 * (image_features * text_features).sum(dim=-1)
        return torch.clamp(score, min=0).detach().cpu().tolist()

    @torch.no_grad()
    def evaluate_scores(self, result: ImageSolutionData) -> List[float]:
        """
        Return one CLIP score per image in the provided result.
        """
        return self._score_images(result.images)

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        scores = self.evaluate_scores(result)
        return np.mean(scores) if scores else 0.0


class MultiCLIPIQAEvaluator(Evaluator[ImageSolutionData, MultiObjectiveFitness]):
    """
    Evaluates the CLIP Image Quality Assessment score of images for multiple objectives.
    For single-objective evaluation, use the SingleCLIPIQAEvaluator.
    https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html
    """

    SupportedCLIPModels = Literal['clip_iqa', 'openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32',
                                  'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14']

    SupportedMetricLiteral = Literal[
        "quality", "brightness", "noisiness", "colorfullness", "colorfulness",
        "sharpness", "contrast", "complexity", "natural",
        "happy", "scary", "new", "warm", "real",
        "beautiful", "lonely", "relaxing"
    ]

    SupportedMetrics = Union[SupportedMetricLiteral, Tuple[str, str]]
    """
    Metrics are either predefined or a tuple of two strings (positive, negative) for custom prompts.
    """

    _METRIC_ALIASES = {
        # TorchMetrics CLIP-IQA uses "colorfullness" (double-l).
        # Keep both spellings for backwards compatibility.
        "colorfulness": "colorfullness",
    }

    def _normalize_metric(self, metric: SupportedMetrics) -> SupportedMetrics:
        if isinstance(metric, tuple):
            return metric
        return self._METRIC_ALIASES.get(metric, metric)

    def _normalize_metrics(self, metrics: Tuple[SupportedMetrics, ...]) -> Tuple[SupportedMetrics, ...]:
        return tuple(self._normalize_metric(metric) for metric in metrics)

    def _setup_model(self, model: SupportedCLIPModels, metrics: Tuple[SupportedMetrics, ...]):
        # noinspection PyTypeChecker
        return CLIPImageQualityAssessment(model_name_or_path=model, prompts=metrics)  # type is correct

    def __init__(self, metrics: Tuple[SupportedMetrics, ...],
                 clip_model: SupportedCLIPModels = "openai/clip-vit-base-patch16"):
        normalized_metrics = self._normalize_metrics(metrics)
        self._metrics = normalized_metrics
        self._model = get_or_create_model(f"MultiCLIPIQAEvaluator_{clip_model}_{normalized_metrics}",
                                          lambda: self._setup_model(clip_model, normalized_metrics))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> MultiObjectiveFitness:
        scores = [[] for _ in range(len(self._metrics))]  # Initialize empty lists for each metric

        for img in result.images:
            t = pil_to_tensor(img).unsqueeze(0)
            score_dict = self._model(t)

            if isinstance(score_dict, dict):
                for i, score in enumerate(score_dict.values()):
                    scores[i].append(score.item())
            else:  # Only single tensor value
                scores[0].append(score_dict.item())

        return [np.mean(scores[i]) for i in range(len(self._metrics))]


class SingleCLIPIQAEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluates the CLIP Image Quality Assessment score of images based on a single metric.
    For multi-objective evaluation, use the MultiCLIPIQAEvaluator.
    https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html
    """

    def _setup_model(self, metric: MultiCLIPIQAEvaluator.SupportedMetrics,
                     clip_model: MultiCLIPIQAEvaluator.SupportedCLIPModels):
        return MultiCLIPIQAEvaluator(metrics=(metric,), clip_model=clip_model)

    def __init__(self, metric: MultiCLIPIQAEvaluator.SupportedMetrics,
                 clip_model: MultiCLIPIQAEvaluator.SupportedCLIPModels = "openai/clip-vit-base-patch16"):
        # Initialize the MultiCLIPIQAEvaluator with a single metric
        self._multi_evaluator = get_or_create_model(f"SingleCLIPIQAEvaluator_{clip_model}_{metric}",
                                                    lambda: self._setup_model(metric, clip_model))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> float:
        scores = self._multi_evaluator.evaluate(result)
        return scores[0]  # Return the score for the single metric


class AestheticPredictorV25ImageEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluates the aesthetic quality of images using the aesthetic_predictor_v2_5 model.
    This model is based on SigLIP.
    """

    def _setup_model(self):
        # Load model and preprocessor
        model, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True, # todo test if still works when disabled
        )

        # Move model to appropriate device and convert to bfloat16 if supported
        if self.device == 'cuda' and torch.cuda.is_available():
            model = model.to(torch.bfloat16).cuda()
        else:
            model = model.to(self.device)

        return model, preprocessor

    def __init__(self, device: torch.device = auto_device()):
        self.device = device
        self.model, self.preprocessor = get_or_create_model("AestheticPredictorV25ImageEvaluator",
                                                           lambda: self._setup_model())

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        scores = []
        for img in result.images:
            # Preprocess image
            pixel_values = self.preprocessor(images=img, return_tensors="pt").pixel_values

            # Convert to appropriate format and device
            if self.device == 'cuda' and torch.cuda.is_available():
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
            else:
                pixel_values = pixel_values.to(self.device)

            # Predict aesthetic score
            score = self.model(pixel_values).logits.squeeze().float().cpu().numpy()
            scores.append(float(score))

        return np.mean(scores) if scores else 0.0
