"""
MolmoPoint model wrapper for the FiftyOne Model Zoo.

Supports both allenai/MolmoPoint-8B (general) and
allenai/MolmoPoint-Img-8B (UI / screenshot tasks).

Supports image pointing and video pointing / tracking.

Class hierarchy:
    MolmoPointImageGetItem          — DataLoader worker transform (images)
    MolmoPointVideoGetItem          — DataLoader worker transform (videos)
    MolmoPointBaseModel             — shared model plumbing
        MolmoPointImageModel        — media_type="image"
        MolmoPointVideoModel        — media_type="video"
"""
import contextlib
import logging
import math
from typing import Dict, List, Optional, Union

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

_TRACKING_MAX_FPS = 10
_POINTING_MAX_FPS = 2


def get_device():
    """Return the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# GetItem – image variant
# ---------------------------------------------------------------------------

class MolmoPointImageGetItem(GetItem):
    """Opens a PIL Image from a sample filepath.

    Runs in DataLoader worker processes — I/O only, no model code.
    Per-sample prompts are resolved in ``predict_all`` via the ``samples``
    parameter, not here.
    """

    @property
    def required_keys(self):
        return ["filepath"]

    def __call__(self, sample_dict):
        return Image.open(sample_dict["filepath"]).convert("RGB")


# ---------------------------------------------------------------------------
# GetItem – video variant
# ---------------------------------------------------------------------------

class MolmoPointVideoGetItem(GetItem):
    """Passes a video sample's filepath and optional per-sample prompt through.

    ``"prompt_field"`` is a logical key: when the user passes
    ``apply_model(..., prompt_field="my_field")``, FiftyOne builds a
    ``field_mapping`` that maps this key to the actual dataset field, so
    ``sample_dict.get("prompt_field")`` contains the per-sample value.
    When no mapping is provided the key is absent and ``.get()`` returns
    ``None``, which ``predict_all`` treats as "use the global model.prompt".
    """

    @property
    def required_keys(self):
        return ["filepath", "prompt_field"]

    def __call__(self, sample_dict):
        return {
            "filepath": sample_dict["filepath"],
            "prompt": sample_dict.get("prompt_field"),
        }


# ---------------------------------------------------------------------------
# Base model – shared plumbing for image and video variants
# ---------------------------------------------------------------------------

class MolmoPointBaseModel(Model, fom.SamplesMixin, SupportsGetItem, TorchModelMixin):
    """Shared base class for MolmoPointImageModel and MolmoPointVideoModel.

    Handles model loading, prompt management, FiftyOne batching boilerplate,
    and device/dtype selection. Subclasses implement ``media_type``,
    ``build_get_item``, and ``predict_all``.
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
        self.prompt = prompt

        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        self._model = None
        self.processor = None
        self._load_model()

    # ------------------------------------------------------------------
    # prompt property
    # ------------------------------------------------------------------

    @property
    def prompt(self) -> List[str]:
        """List of object descriptions to locate.

        Can be set as a comma-separated string (``"boat, person"``) or a list
        (``["boat", "person"]``). Per-sample values from a dataset field
        override this global prompt when ``needs_fields`` is configured.
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
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    def _get_field(self):
        if "prompt_field" in self._fields:
            return self._fields["prompt_field"]
        return next(iter(self._fields.values()), None)

    # ------------------------------------------------------------------
    # Required Model properties
    # ------------------------------------------------------------------

    @property
    def media_type(self):
        raise NotImplementedError

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
    # TorchModelMixin – custom collation for variable-size inputs
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
    # SupportsGetItem – subclasses override
    # ------------------------------------------------------------------

    def build_get_item(self, field_mapping=None):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the processor and model weights onto the target device."""
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
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_prompt(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [s.strip() for s in str(value).split(",") if s.strip()]

    @staticmethod
    def _cuda_dtype() -> torch.dtype:
        return (
            torch.bfloat16
            if torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )

    def _autocast_ctx(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=self._cuda_dtype())
        return contextlib.nullcontext()

    def _resolve_objects(self, sample_prompt) -> List[str]:
        if sample_prompt is not None:
            return self._normalize_prompt(sample_prompt)
        return self._prompt


# ---------------------------------------------------------------------------
# Image model
# ---------------------------------------------------------------------------

class MolmoPointImageModel(MolmoPointBaseModel):
    """MolmoPoint model for image pointing.

    Given one or more object descriptions, returns a ``fo.Keypoints`` label
    per sample with normalized [0, 1] coordinates.

    Args:
        model_path: Local directory or HuggingFace repo ID.
        prompt: Object(s) to point at. Accepts a comma-separated string
            (``"boat, person"``) or a list (``["boat", "person"]``).
        **kwargs: Additional keyword arguments (ignored).
    """

    @property
    def media_type(self):
        return "image"

    def build_get_item(self, field_mapping=None):
        return MolmoPointImageGetItem(field_mapping=field_mapping)

    def _run_single_for_object(self, img: Image.Image, obj: str) -> list:
        """Run one generation pass for *obj* on a single image.

        MolmoPoint's custom ``forward`` uses Python's ``and`` operator on
        multi-element tensors, which raises a ``RuntimeError`` for
        batch_size > 1. Images must therefore be processed one at a time.

        Returns:
            List of ``[object_id, image_num, x, y]`` with absolute pixel
            coordinates.
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

    def predict_all(
        self, batch: List[Image.Image], preprocess=None, samples=None
    ) -> List[Keypoints]:
        """Run inference over a batch of images.

        Args:
            batch: List of PIL Images from ``MolmoPointImageGetItem``.
            preprocess: Unused; preprocessing is handled by GetItem.
            samples: Optional FiftyOne samples for per-sample prompt resolution.

        Returns:
            List of ``fo.Keypoints``, one per item in *batch*.
        """
        accumulated: List[List[Keypoint]] = [[] for _ in batch]
        field_name = self._get_field()

        for i, img in enumerate(batch):
            sample_prompt = None
            if field_name is not None and samples is not None and i < len(samples):
                sample = samples[i]
                if sample.has_field(field_name):
                    sample_prompt = sample.get_field(field_name)

            objects = self._resolve_objects(sample_prompt)

            if not objects:
                logger.debug("No objects resolved for sample %d — skipping.", i)
                continue

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

        Args:
            arg: PIL Image, H×W×C uint8 numpy array (RGB), or filepath.
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


# ---------------------------------------------------------------------------
# Video model
# ---------------------------------------------------------------------------

class MolmoPointVideoModel(MolmoPointBaseModel):
    """MolmoPoint model for video pointing and tracking.

    Two operations are supported:

    - ``"tracking"`` — follows objects across frames at ``max_fps=10``.
      Prompt: ``"Track the {obj}."``.
      Returns frame-level ``{frame_num: fo.Keypoints}``.
      Call ``dataset.compute_metadata()`` first for accurate frame numbers.

    - ``"pointing"`` — identifies objects on sparse frames at ``max_fps=2``.
      Prompt: ``"Point to the {obj}."``.
      Returns frame-level ``{frame_num: fo.Keypoints}`` (sparser coverage).

    Both operations return ``fo.Keypoint`` with:
    - ``label``: the object name you prompted with
    - ``index``: integer object ID from the model (for re-identification)
    - ``points``: normalized ``[[x, y]]`` coordinates in [0, 1]

    Args:
        model_path: Local directory or HuggingFace repo ID.
        prompt: Object(s) to locate. Accepts a comma-separated string or list.
        operation: ``"tracking"`` (default) or ``"pointing"``.
        max_fps: Override default frame sampling rate. Defaults to 10 for
            tracking, 2 for pointing.
        **kwargs: Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        model_path: str,
        prompt: Optional[Union[str, List[str]]] = None,
        operation: str = "tracking",
        max_fps: Optional[int] = None,
        **kwargs,
    ):
        if operation not in ("tracking", "pointing"):
            raise ValueError(
                f"Invalid operation '{operation}'. Must be 'tracking' or 'pointing'."
            )
        self.operation = operation
        self._max_fps = (
            max_fps
            if max_fps is not None
            else (_TRACKING_MAX_FPS if operation == "tracking" else _POINTING_MAX_FPS)
        )
        super().__init__(model_path=model_path, prompt=prompt, **kwargs)

    @property
    def media_type(self):
        return "video"

    def build_get_item(self, field_mapping=None):
        return MolmoPointVideoGetItem(field_mapping=field_mapping)

    def _load_model(self):
        """Load the processor and model.

        Uses ``dtype="auto"`` on CUDA so the model selects its preferred
        dtype from config rather than forcing bfloat16/float16.
        """
        logger.info(f"Loading MolmoPoint-Vid processor from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        model_kwargs = {"trust_remote_code": True}
        if self.device == "cuda":
            model_kwargs["dtype"] = "auto"
            model_kwargs["device_map"] = self.device
        else:
            model_kwargs["torch_dtype"] = torch.float32

        logger.info(f"Loading MolmoPoint-Vid model from {self.model_path}")
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, **model_kwargs
        )
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, obj: str) -> str:
        if self.operation == "tracking":
            return f"Track the {obj}."
        return f"Point to the {obj}."

    @staticmethod
    def _extract_video_path(item: dict) -> str:
        """Extract the filepath from a ``MolmoPointVideoGetItem`` dict."""
        return item["filepath"]

    @staticmethod
    def _get_fps(sample) -> Optional[float]:
        """Return ``frame_rate`` from a FiftyOne sample's metadata, or None."""
        if sample is not None and sample.metadata is not None:
            fps = getattr(sample.metadata, "frame_rate", None)
            if fps:
                return float(fps)
        return None

    def _run_video_inference_for_object(self, video_path: str, obj: str) -> tuple:
        """Run one generation pass for *obj* on a single video.

        Uses ``apply_chat_template`` with ``type="video"`` directly — no
        external ``process_vision_info`` utility needed.

        Args:
            video_path: Absolute path to the video file.
            obj: Object description (e.g. ``"swimmer"``).

        Returns:
            Tuple of ``(points, video_size)`` where:
            - ``points``: list of ``[point_id, timestamp_s, x_px, y_px]``
            - ``video_size``: ``(width, height)`` for coordinate normalisation
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_prompt(obj)},
                    {"type": "video", "video": video_path, "max_fps": self._max_fps},
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
                max_new_tokens=2048,
            )

        generated_tokens = output[:, inputs["input_ids"].size(1):]
        generated_text = self.processor.post_process_image_text_to_text(
            generated_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

        points = self._model.extract_video_points(
            generated_text,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["timestamps"],
            metadata["video_size"],
        )
        return points, metadata["video_size"]

    def _build_frame_dict(
        self,
        video_path: str,
        objects: List[str],
        fps: float,
    ) -> Dict:
        """Run inference for all objects and accumulate frame-level keypoints.

        Returns:
            ``{frame_num: fo.Keypoints}`` with 1-indexed frame numbers.
        """
        frame_kp_lists: Dict[int, List[Keypoint]] = {}

        for obj in objects:
            try:
                points, video_size = self._run_video_inference_for_object(
                    video_path, obj
                )
            except Exception:
                logger.exception(
                    "Inference failed for object '%s' on '%s'", obj, video_path
                )
                continue

            width = float(video_size[0])
            height = float(video_size[1])

            for point in points:
                point_id, timestamp, x_px, y_px = point
                frame_num = max(1, round(float(timestamp) * fps))
                kp = Keypoint(
                    label=obj,
                    index=int(point_id),
                    points=[[
                        max(0.0, min(1.0, float(x_px) / width)),
                        max(0.0, min(1.0, float(y_px) / height)),
                    ]],
                )
                frame_kp_lists.setdefault(frame_num, []).append(kp)

        return {fn: Keypoints(keypoints=kps) for fn, kps in frame_kp_lists.items()}

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict_all(self, batch, preprocess=None, samples=None) -> List[Dict]:
        """Run inference over a batch of videos.

        Each video is processed one object at a time (sequential ``generate``
        calls). The DataLoader workers still accelerate I/O.

        Returns frame-level ``{frame_num: fo.Keypoints}`` dicts. FiftyOne
        treats integer-keyed dicts as frame-level predictions and writes them
        to ``sample.frames``.

        Timestamp → frame: ``max(1, round(timestamp * fps))``.
        Requires ``dataset.compute_metadata()`` for accurate fps; falls back
        to 30 fps with a warning otherwise.

        Args:
            batch: List of dicts (from ``MolmoPointVideoGetItem``) or video
                reader objects (when called via FiftyOne's direct ``predict``
                pathway).
            preprocess: Unused.
            samples: Optional FiftyOne samples for prompt and fps resolution.

        Returns:
            List of ``{frame_num: fo.Keypoints}`` dicts, one per video.
        """
        results = []
        field_name = self._get_field()

        for i, item in enumerate(batch):
            video_path = self._extract_video_path(item)
            sample = (
                samples[i]
                if samples is not None and i < len(samples)
                else None
            )

            # Prompt resolution: GetItem dict value → sample field → global
            sample_prompt = item.get("prompt")
            if sample_prompt is None and field_name is not None and sample is not None:
                if sample.has_field(field_name):
                    sample_prompt = sample.get_field(field_name)

            objects = self._resolve_objects(sample_prompt)

            if not objects:
                logger.warning(
                    "No objects resolved for sample %d ('%s'). "
                    "Set model.prompt or pass prompt_field= to apply_model().",
                    i,
                    video_path,
                )
                results.append({})
                continue

            fps = self._get_fps(sample)
            if fps is None:
                logger.warning(
                    "No frame_rate in metadata for '%s' — defaulting to 30 fps. "
                    "Run dataset.compute_metadata() for accurate frame numbers.",
                    video_path,
                )
                fps = 30.0

            results.append(self._build_frame_dict(video_path, objects, fps))

        return results

    def predict(self, arg, sample=None) -> Dict:
        """Run inference on a single video.

        Args:
            arg: Dict with ``"filepath"`` key, as returned by
                ``MolmoPointVideoGetItem``.
            sample: Optional FiftyOne sample for prompt and fps resolution.

        Returns:
            ``{frame_num: fo.Keypoints}`` dict.
        """
        return self.predict_all(
            [arg],
            samples=[sample] if sample is not None else None,
        )[0]
