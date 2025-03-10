import os
from typing import List, Dict, Tuple, Iterable, TypeVar, Generic, Union, get_args, Optional

import torch
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorboard.plugins import projector

from evolutionary_prompt_embedding.argument_types import PromptEmbedData, PooledPromptEmbedData

# Define generic type variables.
EmbedType = TypeVar("EmbedType", bound=PromptEmbedData)
LabelType = TypeVar("LabelType", str, List[str])

METADATA_FILE = "metadata.tsv"
EMBEDDING_CHECKPOINT = "embedding.ckpt"
SPRITE_IMAGE = "sprite.png"

class TensorboardEmbedVisualizer(Generic[EmbedType, LabelType]):
    """
    Visualizes embeddings in TensorBoard with custom identifiers.
    The user must provide a metadata header (as a str or list of str) which will appear as the first line in the metadata file.
    Part of the utils suite.
    """
    def __init__(self,
                 metadata_header: LabelType,
                 output_folder: str = "vis") -> None:
        self._metadata_header: List[str] = (
            [metadata_header] if isinstance(metadata_header, str) else metadata_header
        )
        self._output_folder = output_folder
        self._embeddings: List[EmbedType] = []
        self._labels: List[LabelType] = []
        self._image_paths: List[str] = []

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

    def _compute_variants(self) -> Dict[str, torch.Tensor]:
        normal_variants: List[torch.Tensor] = []
        pooled_variants: List[torch.Tensor] = []
        combined_avg_variants: List[torch.Tensor] = []
        combined_append_variants: List[torch.Tensor] = []
        has_pooled = any(isinstance(emb, PooledPromptEmbedData) for emb in self._embeddings)

        # Compute the embeddings for each variant.
        for emb in self._embeddings:
            p_embed = emb.prompt_embeds
            normal_flat = p_embed.flatten()
            normal_variants.append(normal_flat)
            if has_pooled:
                pooled_embed = emb.pooled_prompt_embeds
                pooled_flat = pooled_embed.flatten()
                pooled_variants.append(pooled_flat)
                avg_token = p_embed.mean(dim=1)
                combined_avg = torch.cat([avg_token, pooled_embed], dim=-1).flatten()
                combined_avg_variants.append(combined_avg)
                combined_append = torch.cat([normal_flat, pooled_flat], dim=0)
                combined_append_variants.append(combined_append)

        # Variants prefixed with abcd for ordering
        variants: Dict[str, torch.Tensor] = {"embedding_a_normal": torch.stack(normal_variants, dim=0)}
        if has_pooled and pooled_variants:
            variants["embedding_b_pooled"] = torch.stack(pooled_variants, dim=0)
            variants["embedding_c_combined_avg"] = torch.stack(combined_avg_variants, dim=0)
            variants["embedding_d_combined_append"] = torch.stack(combined_append_variants, dim=0)
        return variants

    def _save_checkpoint(self, variants: Dict[str, torch.Tensor]) -> Dict[str, tf.Variable]:
        tf_vars: Dict[str, tf.Variable] = {}
        for key, tensor in variants.items():
            np_array = tensor.cpu().numpy()
            tf_vars[key] = tf.Variable(np_array, name=key)
        checkpoint = tf.train.Checkpoint(**tf_vars)
        checkpoint_path = os.path.join(self._output_folder, EMBEDDING_CHECKPOINT)
        checkpoint.save(checkpoint_path)
        return tf_vars

    def _save_metadata(self) -> None:
        metadata_file = os.path.join(self._output_folder, METADATA_FILE)
        with open(metadata_file, "w") as f:
            # Write the header line first.
            f.write("\t".join(self._metadata_header) + "\n")
            for label in self._labels:
                if isinstance(label, list):
                    safe_label = "\t".join(str(l).replace("\t", " ").replace("\n", " ").strip() for l in label)
                else:
                    safe_label = str(label).replace("\t", " ").replace("\n", " ").strip()
                f.write(f"{safe_label}\n")

    def _has_images(self) -> bool:
        return len(self._image_paths or []) > 0

    def _save_sprite(self, sprite_single_image_dim: Optional[Tuple[int, int]]) -> None:
        if not self._has_images():
            return
        # Sprite conversion inspired by https://medium.com/@juanabascal78/exploratory-image-analysis-part-2-embeddings-on-tensorboard-a13a5d4f98b0
        # needs to be done in order to work with TensorBoard
        # Convert each PIL image to a numpy array after resizing.
        data = []
        for img_path in self._image_paths:
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
        sprite_path = os.path.join(self._output_folder, SPRITE_IMAGE)
        sprite_img.save(sprite_path)

    def _create_projector_config(self, tf_vars: Dict[str, tf.Variable], sprite_single_image_dim: Optional[Tuple[int, int]]) -> projector.ProjectorConfig:
        config = projector.ProjectorConfig()

        for key in tf_vars:
            emb_config = config.embeddings.add()
            emb_config.tensor_name = f"{key}/.ATTRIBUTES/VARIABLE_VALUE" # Naming required
            emb_config.metadata_path = METADATA_FILE
            if self._has_images():
                emb_config.sprite.image_path = SPRITE_IMAGE
                emb_config.sprite.single_image_dim.extend(sprite_single_image_dim)
        return config

    def generate_visualization(self, sprite_single_image_dim: Optional[Tuple[int, int]]) -> None:
        """
        Generates the visualization for the embeddings in TensorBoard.
        Optionally takes a tuple (width, height) for the images in the sprite file. If not provided the images are not used.
        """
        if not self._embeddings:
            raise ValueError("No embeddings have been added.")
        # Check if lengths match or throw an error
        if len(self._embeddings) != len(self._labels):
            raise ValueError("Number of embeddings and labels do not match.")
        if sprite_single_image_dim is not None and self._has_images() and len(self._embeddings) != len(self._image_paths):
            raise ValueError("Number of embeddings and image paths do not match. There are entries without images.")

        os.makedirs(self._output_folder, exist_ok=True)  # Create folder at visualization time.
        variants = self._compute_variants()
        tf_vars = self._save_checkpoint(variants)
        self._save_metadata()
        if sprite_single_image_dim is not None: self._save_sprite(sprite_single_image_dim)
        print("Checkpoint and metadata saved in", self._output_folder)
        config = self._create_projector_config(tf_vars, sprite_single_image_dim)
        projector.visualize_embeddings(self._output_folder, config)
        print(f"Run 'tensorboard --logdir={self._output_folder}' to visualize your embeddings.")