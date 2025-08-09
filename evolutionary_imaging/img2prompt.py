"""
This module provides functionality to generate prompts from images using vision-language models.
It can be used to reverse-engineer prompts from existing images for use in diffusion models.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from evolutionary_model_helpers.auto_device import auto_device


class ImageToPromptGenerator:
    """
    A class that generates prompts from images using a vision-language model.
    It can produce both positive and negative prompts.
    """

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, device: Optional[str] = None, debug: bool = False):
        """
        Initialize the ImageToPromptGenerator with the specified device.

        :param device: The device to use for inference. If None, the best available device is used.
        :param debug: Whether to print debug information, including raw model outputs.
        """
        self.device = device if device is not None else auto_device()
        self.model = None
        self.processor = None
        self.debug = debug
        self._setup_model()

    def _setup_model(self):
        """
        Set up the vision-language model and processor.
        """
        print(f"Loading {self.MODEL_ID} on {self.device}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID, 
            torch_dtype="auto", 
            device_map=None
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            # 512×512 works fine with these; adjust if you pass larger images later:
            min_pixels=14*28*28, 
            max_pixels=48*28*28
        )

    def generate_prompt(self, image: Union[Image.Image, str], 
                        generate_negative: bool = False,
                        use_single_call: bool = True) -> Union[str, Tuple[str, str]]:
        """
        Generate a prompt from an image.

        :param image: The image to generate a prompt from. Can be a PIL Image or a path to an image.
        :param generate_negative: Whether to generate a negative prompt as well.
        :param use_single_call: Whether to use a single model call for both positive and negative prompts.
                               If True and generate_negative is True, uses a single call.
                               If False, uses separate calls for positive and negative prompts.
        :return: If generate_negative is False, returns the positive prompt.
                 If generate_negative is True, returns a tuple of (positive_prompt, negative_prompt).
        """
        # Handle image input
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        # Generate prompt(s) based on parameters
        if generate_negative:
            if use_single_call:
                # Use a single model call to generate both positive and negative prompts
                return self._generate_prompt(img, mode="posneg")
            else:
                # Use separate model calls for positive and negative prompts (legacy behavior)
                positive_prompt = self._generate_single_prompt(img, is_negative=False)
                negative_prompt = self._generate_single_prompt(img, is_negative=True)
                return positive_prompt, negative_prompt
        else:
            # Only generate positive prompt
            return self._generate_prompt(img, mode="single")

    def generate_prompts_from_directory(self, directory_path: str, generate_negative: bool = False, use_single_call: bool = True) -> List[dict]:
        """
        Generate prompts for all images in a directory.

        :param directory_path: Path to the directory containing images.
        :param generate_negative: Whether to generate negative prompts as well.
        :param use_single_call: Whether to use a single model call for both positive and negative prompts.
                               If True and generate_negative is True, uses a single call.
                               If False, uses separate calls for positive and negative prompts.
        :return: A list of dictionaries, each containing the image path and generated prompt(s).
        """
        results = []

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_path = os.path.join(directory_path, filename)
                try:
                    prompt_result = self.generate_prompt(image_path, generate_negative, use_single_call)

                    if generate_negative:
                        positive_prompt, negative_prompt = prompt_result
                        results.append({
                            'image_path': image_path,
                            'positive_prompt': positive_prompt,
                            'negative_prompt': negative_prompt
                        })
                    else:
                        results.append({
                            'image_path': image_path,
                            'prompt': prompt_result
                        })
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")

        return results

    def _parse_pos_neg(self, text: str) -> Tuple[str, str]:
        """
        Robustly extract Positive/Negative lines from the model output.

        :param text: The text output from the model.
        :return: A tuple of (positive_prompt, negative_prompt).
        """
        if self.debug:
            print("DEBUG: Parsing positive and negative prompts from:")
            print(text)
            print("-"*50)

        # Normalize text: replace various dashes, quotes, and newlines for consistency
        t = text.lower()
        t = re.sub(r"[–—]", "-", t)
        t = re.sub(r"[\r\n]+", "\n", t)

        # Try multiple patterns to extract positive prompt
        positive_patterns = [
            # Standard format: "Positive: prompt"
            r"(?:^|\n)\s*positive\s*[:\-]\s*(.+?)(?:\n|$)",
            # Format with "Positive prompt: prompt"
            r"(?:^|\n)\s*positive\s*prompt\s*[:\-]\s*(.+?)(?:\n|$)",
            # Format with just "prompt:" at the beginning
            r"^(?:prompt|description)\s*[:\-]\s*(.+?)(?:\n|$)",
            # Format with section headers
            r"(?:^|\n)\s*\*\*positive\*\*\s*[:\-]?\s*(.+?)(?:\n|$)",
        ]

        # Try multiple patterns to extract negative prompt
        negative_patterns = [
            # Standard format: "Negative: prompt"
            r"(?:^|\n)\s*negative\s*[:\-]\s*(.+?)(?:\n|$)",
            # Format with "Negative prompt: prompt"
            r"(?:^|\n)\s*negative\s*prompt\s*[:\-]\s*(.+?)(?:\n|$)",
            # Format with section headers
            r"(?:^|\n)\s*\*\*negative\*\*\s*[:\-]?\s*(.+?)(?:\n|$)",
            # Format with "to avoid" or similar phrases
            r"(?:^|\n)\s*(?:to avoid|avoid|unwanted)\s*[:\-]\s*(.+?)(?:\n|$)",
        ]

        # Try to find positive prompt
        pos_line = ""
        for pattern in positive_patterns:
            pos_match = re.search(pattern, t, re.IGNORECASE | re.MULTILINE)
            if pos_match:
                pos_line = pos_match.group(1).strip()
                break

        # Try to find negative prompt
        neg_line = ""
        for pattern in negative_patterns:
            neg_match = re.search(pattern, t, re.IGNORECASE | re.MULTILINE)
            if neg_match:
                neg_line = neg_match.group(1).strip()
                break

        # Clean up the extracted prompts
        pos_line = pos_line.strip('"\'').strip()
        neg_line = neg_line.strip('"\'').strip()

        # Fallbacks if extraction failed
        if not pos_line:
            # Use the first non-empty line as positive prompt
            for line in text.splitlines():
                line = line.strip()
                if line and not line.lower().startswith(("positive", "negative")):
                    pos_line = line.strip('"\'').strip()
                    break

        if not neg_line:
            # Use a comprehensive default negative prompt
            neg_line = "blurry, low quality, jpeg artifacts, watermark, text, signature, deformed, distorted, disfigured, bad anatomy, out of frame, cropped"

        # Restore original case
        pos_line = pos_line.strip()
        neg_line = neg_line.strip()

        if self.debug:
            print("DEBUG: Extracted prompts:")
            print(f"Positive: {pos_line}")
            print(f"Negative: {neg_line}")
            print("-"*50)

        return pos_line, neg_line

    def _generate_prompt(self, img: Image.Image, mode: Literal["single", "posneg"] = "single") -> Union[str, Tuple[str, str]]:
        """
        Generate a prompt from an image.

        :param img: The PIL Image to generate a prompt from.
        :param mode: The mode to use for generation. "single" returns one SD-style prompt line (no labels).
                    "posneg" returns a (positive, negative) tuple.
        :return: Either a single prompt string or a tuple of (positive_prompt, negative_prompt).
        """
        # Build instruction based on the mode
        if mode == "posneg":
            instruction = (
                "Analyze this image for Stable Diffusion prompt generation. "
                "Return exactly two clearly labeled sections:\n\n"
                "Positive: Detailed, comma-separated descriptors covering subject, medium, style, lighting, composition, colors, camera settings, mood, and any distinctive features. Be specific and descriptive.\n\n"
                "Negative: Comprehensive list of comma-separated defects and unwanted elements to avoid, including technical issues (blurry, low quality, jpeg artifacts), unwanted elements (watermark, text, signature), and stylistic problems (oversaturation, poor composition)."
            )
        else:
            instruction = (
                "Create a detailed Stable Diffusion prompt for this image. "
                "Provide a single line of comma-separated descriptors that precisely capture: "
                "main subject, artistic medium, specific style, lighting conditions, composition technique, color palette, camera perspective, and any distinctive visual elements. "
                "Be detailed and specific, focusing on visual aspects rather than conceptual ones. "
                "Do not include 'Positive' or 'Negative' labels."
            )

        # Create a chat message with the image and instruction
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(img.filename).absolute().as_uri()) if hasattr(img, 'filename') else "image"},
                {"type": "text", "text": instruction},
            ],
        }]

        # Process the input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt").to(self.device)

        # Print debug information if requested
        if self.debug:
            print("\n" + "="*50)
            print("DEBUG: Input instruction:")
            print(instruction)
            print("-"*50)

        # Generate the prompt
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, 
                max_new_tokens=180,  # Increased for more detailed outputs
                temperature=0.2, 
                do_sample=False
            )

        # Decode the generated text
        gen = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        # Print debug information if requested
        if self.debug:
            print("DEBUG: Raw model output:")
            print(gen)
            print("="*50 + "\n")

        if mode == "posneg":
            return self._parse_pos_neg(gen)
        else:
            # For single-line mode, strip any accidental labels the model might add
            line = gen.splitlines()[0]
            for lab in ("Positive:", "positive:", "POSITIVE:"):
                if line.startswith(lab):
                    line = line[len(lab):].strip()
            return line
