"""
MolmoPoint model wrapper for the FiftyOne Model Zoo.

Supports both allenai/MolmoPoint-8B (general) and
allenai/MolmoPoint-Img-8B (UI / screenshot tasks).
"""
import contextlib
import logging
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.labels import Keypoint, Keypoints
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import GetItem

from transformers import AutoProcessor, AutoModelForImageTextToText

logger = logging.getLogger(__name__)


def get_device():
    """Return the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# GetItem – runs in DataLoader workers (I/O only, no model code here)
# ---------------------------------------------------------------------------

class MolmoPointGetItem(GetItem):
    """Loads a PIL Image (and optional per-sample prompt) from a sample.

    When the user sets ``model.needs_fields = {"prompt_field": "my_field"}``,
    FiftyOne adds ``"prompt_field"`` to the field mapping so that
    ``sample_dict`` contains the per-sample value. If no mapping is set,
    ``sample_dict.get("prompt_field")`` returns ``None`` and the model falls
    back to the global ``model.prompt``.
    """

    @property
    def required_keys(self):
        # Only real dataset fields that always exist.
        # "prompt_field" enters field_mapping only when the user explicitly
        # maps it via model.needs_fields, so it must NOT be listed here.
        return ["filepath"]

    def __call__(self, sample_dict):
        image = Image.open(sample_dict["filepath"]).convert("RGB")
        return {
            "image": image,
            "prompt": sample_dict.get("prompt_field"),  # None if not mapped
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MolmoPointModel(Model, SupportsGetItem, TorchModelMixin):
    """FiftyOne model wrapping MolmoPoint for keypoint prediction.

    Given one or more object descriptions, the model returns a
    ``fo.Keypoints`` label for each sample containing one ``fo.Keypoint``
    per located instance, with normalized [0, 1] coordinates.

    Args:
        model_path: Local directory or HuggingFace repo ID.
        prompt: Object(s) to point at (global, applied to all samples).
            Accepts a comma-separated string (``"boat, person"``) or a list
            of strings (``["boat", "person"]``). Can also be set after
            loading via ``model.prompt = ...``. For per-sample prompts, use
            ``model.needs_fields = {"prompt_field": "your_field_name"}``
            instead.
        **kwargs: Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        model_path: str,
        prompt: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        SupportsGetItem.__init__(self)
        self._preprocess = False
        self._fields = {}
        self.model_path = model_path
        self.prompt = prompt  # normalised via property setter

        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading MolmoPoint processor from {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        model_kwargs = {"trust_remote_code": True}
        if self.device == "cuda":
            capability = torch.cuda.get_device_capability(0)
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if capability[0] >= 8 else torch.float16
            )
            model_kwargs["device_map"] = self.device
        else:
            model_kwargs["torch_dtype"] = torch.float32

        logger.info(f"Loading MolmoPoint model from {model_path}")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, **model_kwargs
        )
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # prompt property
    # ------------------------------------------------------------------

    @property
    def prompt(self) -> List[str]:
        """List of object descriptions to locate (global, applied to all samples).

        Can be set as a comma-separated string (``"boat, person"``) or a list
        (``["boat", "person"]``). When ``needs_fields`` maps a dataset field to
        ``"prompt_field"``, the per-sample value overrides this global prompt
        for that sample.
        """
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        if value is None:
            self._prompt = []
        elif isinstance(value, list):
            self._prompt = [str(v).strip() for v in value if str(v).strip()]
        else:
            self._prompt = [s.strip() for s in str(value).split(",") if s.strip()]

    # ------------------------------------------------------------------
    # needs_fields – optional per-sample field mapping
    # ------------------------------------------------------------------

    @property
    def needs_fields(self):
        """Dict mapping logical keys to dataset field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    # ------------------------------------------------------------------
    # Required properties from Model
    # ------------------------------------------------------------------

    @property
    def media_type(self):
        return "image"

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value):
        self._preprocess = value

    @property
    def ragged_batches(self):
        return False

    # ------------------------------------------------------------------
    # TorchModelMixin – custom collation for variable-size images
    # ------------------------------------------------------------------

    @property
    def has_collate_fn(self):
        return True

    @property
    def collate_fn(self):
        def identity_collate(batch):
            return batch
        return identity_collate

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

    # ------------------------------------------------------------------
    # SupportsGetItem
    # ------------------------------------------------------------------

    def build_get_item(self, field_mapping=None):
        return MolmoPointGetItem(field_mapping=field_mapping)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _autocast_ctx(self):
        """Return appropriate autocast context for the current device."""
        if self.device == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability(0)[0] >= 8
                else torch.float16
            )
            return torch.autocast("cuda", dtype=dtype)
        return contextlib.nullcontext()

    def _run_batch_for_object(
        self, batch: List[Image.Image], obj: str
    ) -> List[list]:
        """Run one generation pass for *obj* over all images in *batch*.

        Args:
            batch: List of PIL Images.
            obj: Single object description (e.g. ``"boat"``).

        Returns:
            List of raw point lists (one per image), where each element is
            a list of ``[object_id, image_num, x, y]`` entries with absolute
            pixel coordinates.
        """
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Point to the {obj}"},
                        {"type": "image", "image": img},
                    ],
                }
            ]
            for img in batch
        ]

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            return_pointing_metadata=True,
        )
        metadata = inputs.pop("metadata")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode(), self._autocast_ctx():
            output = self._model.generate(
                **inputs,
                logits_processor=self._model.build_logit_processor_from_inputs(
                    inputs
                ),
                max_new_tokens=200,
            )

        generated_tokens = output[:, inputs["input_ids"].size(1):]
        generated_texts = self.processor.post_process_image_text_to_text(
            generated_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        batch_points = []
        for i, gen_text in enumerate(generated_texts):
            points = self._model.extract_image_points(
                gen_text,
                metadata["token_pooling"][i],
                metadata["subpatch_mapping"][i],
                metadata["image_sizes"][i],
            )
            batch_points.append(points)

        return batch_points

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _resolve_objects(self, sample_prompt) -> List[str]:
        """Return the object list to use for a single sample.

        Per-sample prompt (from a dataset field) takes priority over the
        global ``model.prompt``. The value is normalised the same way as
        the ``prompt`` setter.
        """
        if sample_prompt is not None:
            if isinstance(sample_prompt, list):
                return [str(v).strip() for v in sample_prompt if str(v).strip()]
            return [s.strip() for s in str(sample_prompt).split(",") if s.strip()]
        return self._prompt

    def predict_all(
        self, batch: List[dict], preprocess=None
    ) -> List[Keypoints]:
        """Run batched inference for all configured object prompts.

        Each item in *batch* is a dict with keys ``"image"`` (PIL Image) and
        ``"prompt"`` (per-sample object string / list, or ``None`` to use the
        global ``model.prompt``).

        When all samples share the same prompt, a single batched forward pass
        is made per object string. When prompts differ across samples, each
        sample is processed individually.

        Args:
            batch: List of dicts from ``MolmoPointGetItem``.
            preprocess: Unused; preprocessing is handled by GetItem.

        Returns:
            List of ``fo.Keypoints``, one per item in *batch*.
        """
        images = [item["image"] for item in batch]
        sample_prompts = [item.get("prompt") for item in batch]

        # Resolve the effective object list for each sample
        per_sample_objects = [self._resolve_objects(p) for p in sample_prompts]

        if not any(per_sample_objects):
            raise ValueError(
                "No objects specified. Set model.prompt = ['boat', 'person'], "
                "or pass prompt=... when loading the model, "
                "or use model.needs_fields = {'prompt_field': 'your_field'}."
            )

        accumulated: List[List[Keypoint]] = [[] for _ in batch]

        # Fast path: all samples share the same prompt → true batch inference
        if len(set(tuple(o) for o in per_sample_objects)) == 1:
            objects = per_sample_objects[0]
            for obj in objects:
                batch_points = self._run_batch_for_object(images, obj)
                for i, (img, points) in enumerate(zip(images, batch_points)):
                    width, height = img.size
                    for point in points:
                        _obj_id, _img_num, x, y = point
                        accumulated[i].append(
                            Keypoint(
                                label=obj,
                                points=[[float(x) / width, float(y) / height]],
                            )
                        )
        else:
            # Slow path: per-sample prompts differ → process each individually
            for i, (img, objects) in enumerate(zip(images, per_sample_objects)):
                for obj in objects:
                    points_list = self._run_batch_for_object([img], obj)
                    width, height = img.size
                    for point in points_list[0]:
                        _obj_id, _img_num, x, y = point
                        accumulated[i].append(
                            Keypoint(
                                label=obj,
                                points=[[float(x) / width, float(y) / height]],
                            )
                        )

        return [Keypoints(keypoints=kps) for kps in accumulated]

    def predict(self, image: np.ndarray, sample=None) -> Keypoints:
        """Run inference on a single image.

        Args:
            image: H x W x C uint8 numpy array (RGB).
            sample: Unused; included for API compatibility.

        Returns:
            ``fo.Keypoints``
        """
        pil_image = Image.fromarray(image)
        return self.predict_all([{"image": pil_image, "prompt": None}])[0]
