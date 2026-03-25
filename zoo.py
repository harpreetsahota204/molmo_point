"""
MolmoPoint model wrapper for the FiftyOne Model Zoo.

Supports both allenai/MolmoPoint-8B (general) and
allenai/MolmoPoint-Img-8B (UI / screenshot tasks).

Supports image pointing and video pointing / tracking.

Class hierarchy:
    MolmoPointGetItem               — shared DataLoader worker transform
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

try:
    from molmo_utils import process_vision_info as _molmo_process_vision_info
except ImportError:
    _molmo_process_vision_info = None

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
# GetItem – runs in DataLoader workers (I/O only, no model code here)
# ---------------------------------------------------------------------------

class MolmoPointGetItem(GetItem):
    """Loads data from a sample dict for use in the DataLoader.

    For images: opens and returns a PIL Image in RGB.
    For videos: returns a lightweight dict with filepath and metadata —
    video frames are extracted during inference to avoid loading full video
    data in worker processes.
    """

    def __init__(self, media_type="image", **kwargs):
        super().__init__(**kwargs)
        self._media_type = media_type

    @property
    def required_keys(self):
        # "metadata" is included so frame_rate is available for tracking's
        # timestamp → frame number conversion without a separate DB lookup.
        return ["filepath", "metadata"]

    def __call__(self, sample_dict):
        if self._media_type == "video":
            return {
                "filepath": sample_dict["filepath"],
                "metadata": sample_dict.get("metadata"),
            }
        return Image.open(sample_dict["filepath"]).convert("RGB")


# ---------------------------------------------------------------------------
# Base model – shared plumbing for image and video variants
# ---------------------------------------------------------------------------

class MolmoPointBaseModel(Model, fom.SamplesMixin, SupportsGetItem, TorchModelMixin):
    """Shared base class for MolmoPointImageModel and MolmoPointVideoModel.

    Handles model loading, prompt management, FiftyOne batching boilerplate,
    and device/dtype selection. Subclasses implement ``media_type`` and
    ``predict_all``.
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
    # SupportsGetItem
    # ------------------------------------------------------------------

    def build_get_item(self, field_mapping=None):
        return MolmoPointGetItem(
            media_type=self.media_type, field_mapping=field_mapping
        )

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
            batch: List of PIL Images from ``MolmoPointGetItem``.
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
      Returns frame-level ``fo.Keypoints`` keyed by 1-indexed frame number.
      Requires ``dataset.compute_metadata()`` for accurate frame numbers.

    - ``"pointing"`` — identifies objects on sparse frames at ``max_fps=2``.
      Returns sample-level ``fo.Keypoints``.

    Both operations return ``fo.Keypoint`` with ``label`` set to the object
    name and ``index`` set to the integer ``object_id`` emitted by the model.
    Coordinates are normalized to ``[0, 1]``.

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
        if _molmo_process_vision_info is None:
            raise ImportError(
                "molmo-utils is required for video inference. "
                "Install it with: pip install molmo-utils"
            )
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

    def _build_prompt(self, obj: str) -> str:
        if self.operation == "tracking":
            return f"Track the {obj}."
        return f"Point to the {obj}."

    def _run_video_inference_for_object(self, video_path: str, obj: str):
        """Run one generation pass for *obj* on a single video.

        Args:
            video_path: Path to the video file.
            obj: Object description (e.g. ``"swimmer"``).

        Returns:
            points: List of ``(object_id, timestamp_seconds, x, y)`` tuples
                with absolute pixel coordinates in ``video_size`` space.
            video_size: ``(width, height)`` of the video as processed by
                the model, used for coordinate normalization.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_prompt(obj)},
                    {"type": "video", "video": video_path, "max_fps": self._max_fps},
                ],
            }
        ]

        _, videos, video_kwargs = _molmo_process_vision_info(messages)
        videos_list, video_metadatas = zip(*videos)
        videos_list = list(videos_list)
        video_metadatas = list(video_metadatas)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            videos=videos_list,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            return_pointing_metadata=True,
            **video_kwargs,
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

        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        points = self._model.extract_video_points(
            generated_text,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["timestamps"],
            metadata["video_size"],
        )

        return points, metadata["video_size"]

    def _predict_tracking(self, video_path: str, objects: List[str], item_metadata) -> Dict:
        """Frame-level keypoints for tracking mode.

        Requires sample metadata with ``frame_rate`` for accurate
        timestamp-to-frame-number conversion. Falls back to 30 fps with a
        warning if not available — call ``dataset.compute_metadata()`` first.

        Returns:
            Dict mapping 1-indexed frame numbers to
            ``{"keypoints": fo.Keypoints}``.
        """
        fps = 30.0
        frame_rate = None
        if item_metadata is not None:
            # item_metadata is a FiftyOne VideoMetadata object
            frame_rate = getattr(item_metadata, "frame_rate", None)
        if frame_rate:
            fps = float(frame_rate)
        else:
            logger.warning(
                "No frame_rate in sample metadata for %s — falling back to "
                "30 fps. Run dataset.compute_metadata() for accurate frame "
                "numbers.",
                video_path,
            )

        frame_dict: Dict = {}

        for obj in objects:
            points, video_size = self._run_video_inference_for_object(video_path, obj)
            vw = float(video_size[0])
            vh = float(video_size[1])

            for obj_id, ts, x, y in points:
                frame_num = math.floor(float(ts) * fps) + 1
                x_norm = max(0.0, min(1.0, float(x) / vw))
                y_norm = max(0.0, min(1.0, float(y) / vh))

                if frame_num not in frame_dict:
                    frame_dict[frame_num] = {"keypoints": Keypoints(keypoints=[])}

                frame_dict[frame_num]["keypoints"].keypoints.append(
                    Keypoint(
                        label=obj,
                        index=int(obj_id),
                        points=[[x_norm, y_norm]],
                    )
                )

        return frame_dict

    def _predict_pointing(self, video_path: str, objects: List[str]) -> Keypoints:
        """Sample-level keypoints for pointing mode.

        Returns:
            ``fo.Keypoints`` with one entry per detected instance.
        """
        all_keypoints: List[Keypoint] = []

        for obj in objects:
            points, video_size = self._run_video_inference_for_object(video_path, obj)
            vw = float(video_size[0])
            vh = float(video_size[1])

            for obj_id, ts, x, y in points:
                all_keypoints.append(
                    Keypoint(
                        label=obj,
                        index=int(obj_id),
                        points=[[
                            max(0.0, min(1.0, float(x) / vw)),
                            max(0.0, min(1.0, float(y) / vh)),
                        ]],
                    )
                )

        return Keypoints(keypoints=all_keypoints)

    def predict_all(self, batch: List[str], preprocess=None, samples=None):
        """Run inference over a batch of video filepaths.

        Each video is processed sequentially — one ``model.generate()`` call
        per object per video. The DataLoader workers still accelerate I/O.

        Args:
            batch: List of video filepaths from ``MolmoPointGetItem``.
            preprocess: Unused.
            samples: Optional FiftyOne samples for per-sample prompt resolution
                and metadata (frame_rate required for tracking).

        Returns:
            For ``"tracking"``: list of dicts mapping frame numbers to
            ``{"keypoints": fo.Keypoints}``.
            For ``"pointing"``: list of ``fo.Keypoints``.
        """
        results = []
        field_name = self._get_field()

        for i, item in enumerate(batch):
            # GetItem returns a dict {"filepath": ..., "metadata": ...} for video
            video_path = item["filepath"] if isinstance(item, dict) else item
            item_metadata = item.get("metadata") if isinstance(item, dict) else None

            sample = (
                samples[i]
                if samples is not None and i < len(samples)
                else None
            )

            sample_prompt = None
            if field_name is not None and sample is not None:
                if sample.has_field(field_name):
                    sample_prompt = sample.get_field(field_name)

            objects = self._resolve_objects(sample_prompt)

            if not objects:
                logger.warning(
                    "No objects resolved for sample %d ('%s') — skipping. "
                    "Set model.prompt or pass prompt_field= to apply_model().",
                    i,
                    video_path,
                )
                results.append(
                    {} if self.operation == "tracking" else Keypoints(keypoints=[])
                )
                continue

            if self.operation == "tracking":
                results.append(
                    self._predict_tracking(video_path, objects, item_metadata)
                )
            else:
                results.append(self._predict_pointing(video_path, objects))

        return results

    def predict(self, arg, sample=None):
        """Run inference on a single video filepath.

        Args:
            arg: Path to a video file.
            sample: Optional FiftyOne sample for per-sample prompt resolution.

        Returns:
            Frame-level dict (tracking) or ``fo.Keypoints`` (pointing).
        """
        return self.predict_all(
            [arg],
            samples=[sample] if sample is not None else None,
        )[0]
