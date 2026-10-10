"""CPU-only official-source/model preparation, independent of GPU ownership."""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .config import MODEL_REVISION
from .data import download

SOURCES = {
    "safe": (
        "https://github.com/Ouxiang-Li/SAFE",
        "4e998724651b227def64f5be0cd60c0aa1552c35",
    ),
    "universal_fake_detect": (
        "https://github.com/Yuheng-Li/UniversalFakeDetect",
        "030495aea3300a8b54c0ec37ec7fe1dd7e63c619",
    ),
    "ssp": (
        "https://github.com/bcmi/SSP-AI-Generated-Image-Detection",
        "38e9ab30f423cf6572bebd481f2b14c16ae8b890",
    ),
    "bfree": (
        "https://github.com/grip-unina/B-Free",
        "c6a9f898782fb466b29af01f21960b67415afb0e",
    ),
    "aeroblade": (
        "https://github.com/jonasricker/aeroblade",
        "50fc2b22c929fdc481757adc4bc81bf98d36f9fc",
    ),
}

# The original SD2 repository is unavailable even to authenticated users.
# Two separately hosted archives expose identical VAE blobs. Preserve the
# original model identity while recording the actual download provenance.
SD2_ARCHIVE = (
    "Manojb/stable-diffusion-2-base",
    "64bf7b4f10eee35494b38d55c06c0c78cf8b44d0",
)
SD2_VAE_SHA256 = "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a"
SD2_CONFIG_BLOB = "3586c43eb38b5ca71bf790d4bae61f1794143a38"


def extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if entry.filename == "/" and entry.is_dir():
                continue
            target = destination / entry.filename
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Unsafe archive entry")
            if not entry.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)


def prepare_models(root, source_root=None):
    from huggingface_hub import HfApi, snapshot_download

    root = Path(root).resolve()
    source_root = Path(source_root).resolve() if source_root else root / "sources"
    root.mkdir(parents=True, exist_ok=True)
    lock = {"hf_cache": str(root / "hf_cache"), "detectors": {}, "complete": False}
    for detector, (url, revision) in SOURCES.items():
        source = source_root / detector
        if not source.exists():
            subprocess.run(
                ["git", "clone", "--no-checkout", url, str(source)], check=True
            )
            subprocess.run(
                ["git", "-C", str(source), "checkout", "--detach", revision], check=True
            )
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise ValueError(f"Require pinned source {detector}: {revision}")
        lock["detectors"][detector] = {
            "source": str(source),
            "url": url,
            "commit": revision,
            "files": {},
        }
    checks = {
        "safe": source_root / "safe/checkpoint/checkpoint-best.pth",
        "universal_fake_detect": source_root
        / "universal_fake_detect/pretrained_weights/fc_weights.pth",
    }
    archive = root / "bfree.zip"
    if not archive.exists():
        download(
            "https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip",
            archive,
        )
    if hashes(archive)["md5"] != "f3f53fa647848b16cf81c913f148a198":
        raise ValueError("B-Free release MD5 mismatch")
    extract(archive, root / "bfree")
    checks["bfree"] = root / "bfree/BFREE_dino2reg4/model_epoch_best.pth"
    lock["detectors"]["bfree"]["model_config"] = str(
        root / "bfree/BFREE_dino2reg4/config.yaml"
    )
    archive = root / "ssp.zip"
    if not archive.exists():
        download(
            "https://www.dropbox.com/scl/fo/2xdvtew4rjrrsl6cseq30/AElVHGO84W1DSmlImMg1ruM?rlkey=e1a2hnzh62wuuxrnbkfv6f90v&dl=1",
            archive,
        )
    # Dropbox regenerates ZIP container metadata. Pin the released sd4 file,
    # rather than mistaking a container timestamp change for a weight change.
    extract(archive, root / "ssp")
    matches = list((root / "ssp").rglob("sd4/Net_epoch_best.pth"))
    if len(matches) != 1:
        raise ValueError("Require exactly one SSP sd4 checkpoint")
    checks["ssp"] = matches[0]
    if (
        hashes(matches[0])["sha256"]
        != "47df39bfa24c49ec786acde8443818f08da0ef0ea1973071f6a783aa89a431f7"
    ):
        raise ValueError("SSP sd4 checkpoint SHA-256 mismatch")
    for detector, path in checks.items():
        item = lock["detectors"][detector]
        item["checkpoint"] = str(path)
        item["files"][str(path)] = hashes(path)["sha256"]
    bfree_config = lock["detectors"]["bfree"]["model_config"]
    lock["detectors"]["bfree"]["files"][bfree_config] = hashes(bfree_config)["sha256"]
    clip_path = root / "ViT-L-14.pt"
    if not clip_path.exists():
        download(
            "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            clip_path,
        )
    if (
        hashes(clip_path)["sha256"]
        != "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
    ):
        raise ValueError("CLIP checkpoint SHA mismatch")
    lock["detectors"]["universal_fake_detect"]["clip_checkpoint"] = str(clip_path)
    lock["detectors"]["universal_fake_detect"]["files"][str(clip_path)] = hashes(
        clip_path
    )["sha256"]
    vgg = root / "torch_hub/hub/checkpoints/vgg16-397923af.pth"
    if not vgg.exists():
        download("https://download.pytorch.org/models/vgg16-397923af.pth", vgg)
    if not hashes(vgg)["sha256"].startswith("397923af"):
        raise ValueError("VGG16 foundation checkpoint hash mismatch")
    lock["torch_home"] = str(root / "torch_hub")
    lock["detectors"]["aeroblade"]["files"][str(vgg)] = hashes(vgg)["sha256"]
    atomic_json(root / "assets.partial.json", lock)
    allow = [
        "*.json",
        "*.txt",
        "*.model",
        "*.merges",
        "*.vocab",
        "*/*.fp16.safetensors",
    ]
    generation = snapshot_download(
        "stabilityai/sdxl-turbo",
        revision=MODEL_REVISION,
        cache_dir=lock["hf_cache"],
        allow_patterns=allow,
    )
    lock["generation"] = {
        "revision": MODEL_REVISION,
        "files": {
            str(p): hashes(p)["sha256"]
            for p in Path(generation).rglob("*")
            if p.is_file()
        },
    }
    atomic_json(root / "assets.partial.json", lock)
    autoencoders = []
    failures = []
    for repo in (
        "CompVis/stable-diffusion-v1-1",
        "stabilityai/stable-diffusion-2-base",
        "kandinsky-community/kandinsky-2-1",
    ):
        try:
            download_repo = repo
            provenance = {}
            if repo == "stabilityai/stable-diffusion-2-base":
                download_repo, revision = SD2_ARCHIVE
                witnesses = []
                for witness, pinned in (
                    SD2_ARCHIVE,
                    (
                        "sd2-community/stable-diffusion-2-base",
                        "f5bc1bd97485577aa0b946fa8a9004e2ec147402",
                    ),
                ):
                    metadata = HfApi().model_info(
                        witness, revision=pinned, files_metadata=True
                    )
                    siblings = {s.rfilename: s for s in metadata.siblings}
                    if (
                        siblings[
                            "vae/diffusion_pytorch_model.fp16.safetensors"
                        ].lfs.sha256
                        != SD2_VAE_SHA256
                        or siblings["vae/config.json"].blob_id != SD2_CONFIG_BLOB
                    ):
                        raise ValueError("SD2 archived autoencoder identity differs")
                    witnesses.append({"repo": witness, "revision": pinned})
                provenance = {
                    "download_repo": download_repo,
                    "archive_witnesses": witnesses,
                    "vae_sha256": SD2_VAE_SHA256,
                    "original_repository_unavailable": True,
                    "identity_limit": "Matching archived blobs; original repository cannot be queried.",
                }
            else:
                revision = HfApi().model_info(repo).sha
            patterns = (
                allow
                if "kandinsky" not in repo
                else [
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.merges",
                    "*.vocab",
                    "*/*.safetensors",
                ]
            )
            snapshot = snapshot_download(
                download_repo,
                revision=revision,
                cache_dir=lock["hf_cache"],
                allow_patterns=patterns,
            )
            # Official code distinguishes Kandinsky by its repository path.
            destination = root / "autoencoders" / repo.split("/")[-1]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copytree(snapshot, destination)
            if (
                provenance
                and hashes(
                    destination / "vae/diffusion_pytorch_model.fp16.safetensors"
                )["sha256"]
                != SD2_VAE_SHA256
            ):
                raise ValueError("Downloaded SD2 archive differs from frozen VAE")
            files = {
                str(p): hashes(p)["sha256"]
                for p in destination.rglob("*")
                if p.is_file()
            }
            lock["detectors"]["aeroblade"]["files"].update(files)
            autoencoders.append(
                {
                    "name": repo.split("/")[-1],
                    "repo": repo,
                    "revision": revision,
                    "snapshot": str(destination),
                    **provenance,
                }
            )
        except Exception as error:  # noqa: BLE001 -- Preserve failures before rejecting incomplete preparation.
            failures.append(
                {
                    "repo": repo,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        lock["detectors"]["aeroblade"]["autoencoders"] = autoencoders
        atomic_json(root / "assets.partial.json", lock)
    if failures:
        atomic_json(root / "asset_failures.json", failures)
        raise RuntimeError(
            "Official model preparation incomplete; see asset_failures.json"
        )
    lock["complete"] = True
    atomic_json(root / "assets.json", lock)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources")
    args = parser.parse_args()
    prepare_models(args.output, args.sources)


if __name__ == "__main__":
    main()
