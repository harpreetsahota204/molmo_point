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
import fiftyone.core.models as fom
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
    """Loads a PIL Image from a sample's filepath.

    Runs in DataLoader worker processes — I/O only, no model code.
    Per-sample prompts are read from the sample object in ``predict_all``
    via the ``samples`` parameter, not here.
    """

    @property
    def required_keys(self):
        return ["filepath"]

    def __call__(self, sample_dict):
        return Image.open(sample_dict["filepath"]).convert("RGB")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MolmoPointModel(Model, fom.SamplesMixin, SupportsGetItem, TorchModelMixin):
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
        fom.SamplesMixin.__init__(self)
        SupportsGetItem.__init__(self)
        self._preprocess = False
        self._fields = {}
        self.model_path = model_path
        self.prompt = prompt  # normalised via property setter

        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Lazy-loaded on first call to predict_all
        self._model = None
        self.processor = None

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
        self._prompt = self._normalize_prompt(value)

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

    def _get_field(self):
        """Return the dataset field name to use for per-sample prompts, or None."""
        if "prompt_field" in self._fields:
            return self._fields["prompt_field"]
        return next(iter(self._fields.values()), None)

    def _get_field(self):
        """Get the field name to use for prompt extraction."""
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
        return prompt_field
        
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
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the processor and model weights (called lazily on first use)."""
        logger.info(f"Loading MolmoPoint processor from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        model_kwargs = {"trust_remote_code": True}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = self._cuda_dtype()
            model_kwargs["device_map"] = self.device
        else:
            model_kwargs["torch_dtype"] = torch.float32

        logger.info(f"Loading MolmoPoint model from {self.model_path}")
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, **model_kwargs
        )
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_prompt(value) -> List[str]:
        """Normalise a raw prompt value into a list of stripped, non-empty strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [s.strip() for s in str(value).split(",") if s.strip()]

    @staticmethod
    def _cuda_dtype() -> torch.dtype:
        """Return the appropriate float dtype for the current CUDA device.

        Uses bfloat16 on compute capability >= 8 (Ampere+), float16 otherwise.
        """
        return (
            torch.bfloat16
            if torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )

    def _autocast_ctx(self):
        """Return appropriate autocast context for the current device."""
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=self._cuda_dtype())
        return contextlib.nullcontext()

    def _run_single_for_object(self, img: Image.Image, obj: str) -> list:
        """Run one generation pass for *obj* on a single image.

        MolmoPoint's custom ``forward`` method uses Python's ``and`` operator
        on multi-element tensors, which raises a ``RuntimeError`` for
        batch_size > 1. Images must therefore be processed one at a time at
        the model level.

        Args:
            img: A single PIL Image.
            obj: Object description (e.g. ``"boat"``).

        Returns:
            List of ``[object_id, image_num, x, y]`` entries with absolute
            pixel coordinates for this image.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Point to the {obj}"},
                    {"type": "image", "image": img},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
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
        generated_text = self.processor.post_process_image_text_to_text(
            generated_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._model.extract_image_points(
            generated_text,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )

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
            return self._normalize_prompt(sample_prompt)
        return self._prompt

    def predict_all(
        self, batch: List[Image.Image], preprocess=None, samples=None
    ) -> List[Keypoints]:
        """Run inference over a batch of images.

        *batch* is a list of PIL Images loaded by ``MolmoPointGetItem``.

        Per-sample prompts are resolved in priority order:
        1. Value read from the sample's ``prompt_field`` dataset field
           (when ``apply_model(..., prompt_field="my_field")`` is used, or
           ``model.needs_fields = {"prompt_field": "my_field"}`` is set).
        2. Global ``model.prompt``.

        MolmoPoint's model code does not support batch_size > 1 in its
        forward pass, so images are processed one at a time. The DataLoader
        workers still load images in parallel, so ``num_workers`` and
        ``batch_size`` passed to ``apply_model`` continue to improve
        throughput via I/O parallelism.

        Args:
            batch: List of PIL Images from ``MolmoPointGetItem``.
            preprocess: Unused; preprocessing is handled by GetItem.
            samples: Optional list of FiftyOne samples, used to read
                per-sample prompt fields.

        Returns:
            List of ``fo.Keypoints``, one per item in *batch*.
        """
        if self._model is None:
            self._load_model()

        accumulated: List[List[Keypoint]] = [[] for _ in batch]
        field_name = self._get_field()

        for i, img in enumerate(batch):
            # Resolve per-sample prompt from sample field, fall back to global
            sample_prompt = None
            if field_name is not None and samples is not None and i < len(samples):
                sample = samples[i]
                if sample.has_field(field_name):
                    sample_prompt = sample.get_field(field_name)

            objects = self._resolve_objects(sample_prompt)

            if not objects:
                raise ValueError(
                    "No objects specified. Set model.prompt = ['boat', 'person'], "
                    "pass prompt_field= to apply_model(), "
                    "or use model.needs_fields = {'prompt_field': 'your_field'}."
                )

            width, height = img.size
            for obj in objects:
                points = self._run_single_for_object(img, obj)
                for point in points:
                    _obj_id, _img_num, x, y = point
                    accumulated[i].append(
                        Keypoint(
                            label=obj,
                            points=[[float(x) / width, float(y) / height]],
                        )
                    )

        return [Keypoints(keypoints=kps) for kps in accumulated]

    def predict(self, arg, sample=None) -> Keypoints:
        """Run inference on a single image.

        Accepts a PIL Image, a numpy array, or any object with a filepath
        attribute. When a ``sample`` is provided and ``needs_fields`` (or
        ``prompt_field=`` in ``apply_model``) maps a field, the per-sample
        value is used as the prompt.

        Args:
            arg: PIL Image, H x W x C uint8 numpy array (RGB), or filepath.
            sample: Optional FiftyOne sample for per-sample prompt resolution.

        Returns:
            ``fo.Keypoints``
        """
        if isinstance(arg, Image.Image):
            pil_image = arg
        else:
            pil_image = Image.fromarray(arg)

        return self.predict_all(
            [pil_image],
            samples=[sample] if sample is not None else None,
        )[0]
