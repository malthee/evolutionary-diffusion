import os
import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, TypeVar, Generic, Union, get_args, Optional, Callable
from enum import Enum
from PIL import Image
from tensorboard.plugins import projector

from evolutionary_prompt_embedding.argument_types import PromptEmbedData, PooledPromptEmbedData

# Define generic type variables.
EmbedType = TypeVar("EmbedType", bound=PromptEmbedData)
LabelType = TypeVar("LabelType", str, List[str])

METADATA_FILE = "metadata.tsv"
EMBEDDING_CHECKPOINT = "embedding.ckpt"
SPRITE_IMAGE = "sprite.png"
SPRITE_MAX_DIM = 8192  # TensorBoard supports sprites up to 8192x8192 pixels.
DEFAULT_OUTPUT_FOLDER = "vis"


class EmbeddingVariant(Enum):
    NORMAL = "embedding_a_normal"
    POOLED = "embedding_b_pooled"
    COMBINED_AVG = "embedding_c_combined_avg"
    COMBINED_APPEND = "embedding_d_combined_append"


class TensorboardEmbedVisualizer(Generic[EmbedType, LabelType]):
    """
    Visualizes embeddings in TensorBoard with custom identifiers.
    The user must provide a metadata header (as a str or list of str) which will appear as the first line in the metadata file.
    Warning as for now TensorBoard is limited to 2GB of data in the embeddings checkpoint because of Protobuf limitations.
    This already had an RFC but did not get implemented yet, and is gated by an is_oss flag https://github.com/tensorflow/community/blob/master/rfcs/20230720-unbound-saved-model.md.
    """
    def __init__(self,
                 metadata_header: LabelType,
                 output_folder: str = DEFAULT_OUTPUT_FOLDER) -> None:
        self._metadata_header: List[str] = (
            [metadata_header] if isinstance(metadata_header, str) else metadata_header
        )
        self._output_folder = output_folder
        self._embeddings: List[EmbedType] = []
        self._labels: List[LabelType] = []
        self._image_paths: List[str] = []

    def _compute_variants(self, include_variants: Iterable[EmbeddingVariant], embeddings: List[EmbedType]) -> Dict[str, torch.Tensor]:
        normal_variants: List[torch.Tensor] = []
        pooled_variants: List[torch.Tensor] = []
        combined_avg_variants: List[torch.Tensor] = []
        combined_append_variants: List[torch.Tensor] = []
        has_pooled = any(isinstance(emb, PooledPromptEmbedData) for emb in embeddings)

        # Compute the embeddings for each variant.
        for emb in embeddings:
            p_embed = emb.prompt_embeds
            normal_flat = p_embed.flatten()
            if EmbeddingVariant.NORMAL in include_variants:
                normal_variants.append(normal_flat)
            if has_pooled:
                pooled_embed = emb.pooled_prompt_embeds
                pooled_flat = pooled_embed.flatten()
                if EmbeddingVariant.POOLED in include_variants:
                    pooled_variants.append(pooled_flat)
                if EmbeddingVariant.COMBINED_AVG in include_variants:
                    avg_token = p_embed.mean(dim=1)
                    combined_avg = torch.cat([avg_token, pooled_embed], dim=-1).flatten()
                    combined_avg_variants.append(combined_avg)
                if EmbeddingVariant.COMBINED_APPEND in include_variants:
                    combined_append = torch.cat([normal_flat, pooled_flat], dim=0)
                    combined_append_variants.append(combined_append)

        # Create a dictionary of embedding variants to include in the checkpoint
        # Stacked to a single tensor
        variants: Dict[str, torch.Tensor] = {EmbeddingVariant.NORMAL.value: torch.stack(normal_variants, dim=0)}
        if EmbeddingVariant.NORMAL in include_variants:
            print("Including default embeddings.")
            variants[EmbeddingVariant.NORMAL.value] = torch.stack(normal_variants, dim=0)

        if has_pooled:
            if EmbeddingVariant.POOLED in include_variants:
                print("Including pooled embeddings.")
                variants[EmbeddingVariant.POOLED.value] = torch.stack(pooled_variants, dim=0)

            if EmbeddingVariant.COMBINED_AVG in include_variants:
                print("Including combined average embeddings.")
                variants[EmbeddingVariant.COMBINED_AVG.value] = torch.stack(combined_avg_variants, dim=0)

            if EmbeddingVariant.COMBINED_APPEND in include_variants:
                print("Including combined appended embeddings.")
                variants[EmbeddingVariant.COMBINED_APPEND.value] = torch.stack(combined_append_variants, dim=0)
        return variants

    def _save_checkpoint(self, variants: Dict[str, torch.Tensor]) -> Dict[str, tf.Variable]:
        tf_vars: Dict[str, tf.Variable] = {}
        for key, tensor in variants.items():
            np_array = tensor.cpu().numpy()
            tf_vars[key] = tf.Variable(np_array, name=key)
        checkpoint = tf.train.Checkpoint(**tf_vars)
        checkpoint_path = os.path.join(self._output_folder, EMBEDDING_CHECKPOINT)
        checkpoint.save(checkpoint_path)
        total_size = sum(f.stat().st_size for f in Path(self._output_folder).glob(f"{EMBEDDING_CHECKPOINT}*"))
        size_gb = total_size / (1024 ** 3)
        print(f"Total saved checkpoint size: {size_gb:.2f} GB")
        if total_size > 2 * 1024 ** 3:
            print("Warning: Checkpoint size exceeds 2GB! This checkpoint is unusable in TensorBoard embedding visualization. Consider filtering embeddings.")
        return tf_vars

    def _save_metadata(self, labels: List[LabelType]) -> None:
        metadata_file = os.path.join(self._output_folder, METADATA_FILE)
        with open(metadata_file, "w") as f:
            # Write the header line first.
            f.write("\t".join(self._metadata_header) + "\n")
            for label in labels:
                if isinstance(label, list):
                    safe_label = "\t".join(str(l).replace("\t", " ").replace("\n", " ").strip() for l in label)
                else:
                    safe_label = str(label).replace("\t", " ").replace("\n", " ").strip()
                f.write(f"{safe_label}\n")

    def _save_sprite(self, image_paths: List[str], sprite_single_image_dim: Optional[Tuple[int, int]]) -> None:
        if len(image_paths) == 0:
            return
        # Sprite conversion inspired by https://medium.com/@juanabascal78/exploratory-image-analysis-part-2-embeddings-on-tensorboard-a13a5d4f98b0
        # needs to be done in order to work with TensorBoard
        # Convert each PIL image to a numpy array after resizing.
        data = []
        for img_path in image_paths:
            img_resized = Image.open(img_path).resize(sprite_single_image_dim)
            arr = np.array(img_resized)
            data.append(arr)
        data = np.array(data)  # shape: (N, H, W, C) or (N, H, W)

        # If data has shape (N, H, W) (grayscale), convert to (N, H, W, 3)
        if data.ndim == 3:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

        data = data.astype(np.float32)

        # Normalize each image individually.
        mins = np.min(data.reshape(data.shape[0], -1), axis=1)
        data = (data.transpose(1, 2, 3, 0) - mins).transpose(3, 0, 1, 2)
        maxs = np.max(data.reshape(data.shape[0], -1), axis=1)
        data = (data.transpose(1, 2, 3, 0) / maxs).transpose(3, 0, 1, 2)

        n = int(np.ceil(np.sqrt(data.shape[0])))
        # Pad data so that number of images is a perfect square.
        padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=0)

        # Arrange images into a grid.
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        sprite = (data * 255).astype(np.uint8)
        sprite_img = Image.fromarray(sprite)

        if sprite_img.width > SPRITE_MAX_DIM or sprite_img.height > SPRITE_MAX_DIM:
            print(f"Warning: Generated sprite image size ({sprite_img.width}x{sprite_img.height}) "
                  f"exceeds the maximum supported size of {SPRITE_MAX_DIM}x{SPRITE_MAX_DIM} for TensorBoard Embedding Projector. "
                  f"Consider tuning the sprite_single_image_dim parameter.")
        sprite_path = os.path.join(self._output_folder, SPRITE_IMAGE)
        sprite_img.save(sprite_path)

    def _create_projector_config(self, tf_vars: Dict[str, tf.Variable], sprite_single_image_dim: Optional[Tuple[int, int]]) -> projector.ProjectorConfig:
        config = projector.ProjectorConfig()

        for key in tf_vars:
            emb_config = config.embeddings.add()
            emb_config.tensor_name = f"{key}/.ATTRIBUTES/VARIABLE_VALUE" # Naming required
            emb_config.metadata_path = METADATA_FILE
            if len(self._image_paths) > 0 and sprite_single_image_dim is not None:
                emb_config.sprite.image_path = SPRITE_IMAGE
                emb_config.sprite.single_image_dim.extend(sprite_single_image_dim)
        return config

    @property
    def output_folder(self) -> str:
        return self._output_folder

    def add_embedding(self, embedding: EmbedType, label: LabelType, image_path: Optional[str]) -> None:
        self._embeddings.append(embedding)
        self._labels.append(label)
        if image_path: self._image_paths.append(image_path)

    def add_embeddings(self, items: Iterable[Tuple[EmbedType, LabelType, Optional[str]]]) -> None:
        for item in items:
            if len(item) == 2:
                embedding, label = item
                image_path = None
            elif len(item) == 3:
                embedding, label, image_path = item
            else:
                raise ValueError(f"Invalid number of items ({len(item)}) provided, expected 2 or 3.")

            self.add_embedding(embedding, label, image_path)

    def generate_visualization(self, include_variants: Optional[Iterable[EmbeddingVariant]] = None,
                                sprite_single_image_dim: Optional[Tuple[int, int]] = None,
                                filter_predicate: Optional[Callable[[EmbedType, LabelType, str], bool]] = None) -> None:
        """
        Generates the visualization for the embeddings in TensorBoard. Saving the embeddings, metadata, and sprite image in the output folder.
        Warning: The embedding checkpoint shall not exceed 2GB or else TensorBoard is not able to load it! As a rule of thumb if all
        variants are included around 3000 embeddings can be visualized.
        Optionally variants of embeddings can be specifically included in the visualization. By default, all are included.
        Optionally embeddings can be filtered using a predicate function that takes the embedding, label, and image path as arguments.
        Optionally takes a tuple (width, height) for the images in the sprite file. If not provided the images are not used.
        """
        if not self._embeddings:
            raise ValueError("No embeddings have been added.")
        # Check if lengths match or throw an error
        if len(self._embeddings) != len(self._labels):
            raise ValueError("Number of embeddings and labels do not match.")
        if sprite_single_image_dim is not None and len(self._embeddings) != len(self._image_paths):
            raise ValueError("Number of embeddings and image paths do not match. There are entries without images.")

        if include_variants is None:
            include_variants = list(EmbeddingVariant) # Default to all variants

        filtered_embeddings: List[EmbedType] = []
        filtered_labels: List[LabelType] = []
        filtered_image_paths: List[str] = []
        if filter_predicate is not None:
            for emb, lab, img in zip(self._embeddings, self._labels, self._image_paths):
                if filter_predicate(emb, lab, img):
                    filtered_embeddings.append(emb)
                    filtered_labels.append(lab)
                    filtered_image_paths.append(img)
        else:
            filtered_embeddings = self._embeddings
            filtered_labels = self._labels
            filtered_image_paths = self._image_paths

        os.makedirs(self._output_folder, exist_ok=True)  # Create folder at visualization time.
        variants = self._compute_variants(embeddings=filtered_embeddings, include_variants=include_variants)
        tf_vars = self._save_checkpoint(variants)
        self._save_metadata(labels=filtered_labels)
        if sprite_single_image_dim is not None: self._save_sprite(image_paths=filtered_image_paths, sprite_single_image_dim=sprite_single_image_dim)
        config = self._create_projector_config(tf_vars=tf_vars, sprite_single_image_dim=sprite_single_image_dim)
        projector.visualize_embeddings(logdir=self._output_folder, config=config)
        print(f"Run 'tensorboard --logdir={self._output_folder}' to visualize your embeddings.")