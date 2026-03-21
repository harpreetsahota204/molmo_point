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
    """Loads a PIL Image from a sample's filepath."""

    @property
    def required_keys(self):
        return ["filepath"]

    def __call__(self, sample_dict):
        return Image.open(sample_dict["filepath"]).convert("RGB")


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
        objects: Object(s) to point at. Accepts a comma-separated string
            (``"boat, person"``) or a list of strings
            (``["boat", "person"]``).
        **kwargs: Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        model_path: str,
        objects: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        SupportsGetItem.__init__(self)
        self._preprocess = False
        self._fields = {}
        self.model_path = model_path
        self.objects = objects  # normalised via property setter

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
    # objects property
    # ------------------------------------------------------------------

    @property
    def objects(self) -> List[str]:
        """List of object descriptions to locate."""
        return self._objects

    @objects.setter
    def objects(self, value):
        if value is None:
            self._objects = []
        elif isinstance(value, list):
            self._objects = [str(v).strip() for v in value if str(v).strip()]
        else:
            self._objects = [s.strip() for s in str(value).split(",") if s.strip()]

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

    def predict_all(
        self, batch: List[Image.Image], preprocess=None
    ) -> List[Keypoints]:
        """Run batched inference for all configured object prompts.

        One forward pass is made per object string; results are accumulated
        across prompts and returned as one ``fo.Keypoints`` per image.

        Args:
            batch: List of PIL Images supplied by the DataLoader.
            preprocess: Unused; preprocessing is handled by GetItem.

        Returns:
            List of ``fo.Keypoints``, one per image in *batch*.
        """
        if not self._objects:
            raise ValueError(
                "No objects specified. Set model.objects = ['boat', 'person'] "
                "or pass objects=... when loading the model."
            )

        accumulated: List[List[Keypoint]] = [[] for _ in batch]

        for obj in self._objects:
            batch_points = self._run_batch_for_object(batch, obj)
            for i, (img, points) in enumerate(zip(batch, batch_points)):
                width, height = img.size
                for point in points:
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
        return self.predict_all([pil_image])[0]
