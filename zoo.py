"""
MolmoPoint model wrapper for the FiftyOne Model Zoo.

Supports both allenai/MolmoPoint-8B (general) and
allenai/MolmoPoint-Img-8B (UI / screenshot tasks).

Supports image pointing and video pointing / tracking.

Class hierarchy:
    MolmoPointImageGetItem      — DataLoader worker transform (images)
    MolmoPointVideoGetItem      — DataLoader worker transform (videos)
    MolmoPointBaseModel         — shared model plumbing
        MolmoPointImageModel    — media_type="image"
        MolmoPointVideoModel    — media_type="video"

FiftyOne routing:
    Both models use SupportsGetItem + TorchModelMixin for the DataLoader
    pathway. The video model's predict() additionally normalises raw
    filepath strings and video-reader objects into the dict format that
    predict_all() expects, so it works when FiftyOne falls back to the
    direct predict() pathway as well.
"""
import contextlib
import logging
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

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
    """Opens a PIL Image from a sample filepath for DataLoader workers."""

    @property
    def required_keys(self):
        return ["filepath"]

    def __call__(self, sample_dict):
        return Image.open(sample_dict["filepath"]).convert("RGB")


# ---------------------------------------------------------------------------
# GetItem – video variant
# ---------------------------------------------------------------------------

class MolmoPointVideoGetItem(GetItem):
    """Returns a lightweight dict for each video sample.

    ``"prompt_field"`` is a logical key mapped by FiftyOne via
    ``field_mapping`` when the user passes
    ``apply_model(..., prompt_field="my_field")``.
    When no mapping is provided, ``.get("prompt_field")`` returns ``None``
    and ``predict_all`` falls back to ``model.prompt``.

    ``"metadata"`` carries ``VideoMetadata`` (frame_rate, etc.) into
    ``predict_all`` so it is available without an extra DB lookup.
    """

    @property
    def required_keys(self):
        return ["filepath", "prompt_field", "metadata"]

    def __call__(self, sample_dict):
        return {
            "filepath": sample_dict["filepath"],
            "prompt": sample_dict.get("prompt_field"),
            "metadata": sample_dict.get("metadata"),
        }


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class MolmoPointBaseModel(Model, fom.SamplesMixin, SupportsGetItem, TorchModelMixin):
    """Shared base for image and video MolmoPoint models."""

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

    @property
    def prompt(self) -> List[str]:
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = self._normalize_prompt(value)

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

    @property
    def has_collate_fn(self):
        return True

    @property
    def collate_fn(self):
        def identity_collate(batch):
            return batch
        return identity_collate

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

    def build_get_item(self, field_mapping=None):
        raise NotImplementedError

    def _load_model(self):
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

    Args:
        model_path: Local directory or HuggingFace repo ID.
        prompt: Object(s) to point at (comma-separated string or list).
        **kwargs: Ignored.
    """

    @property
    def media_type(self):
        return "image"

    def build_get_item(self, field_mapping=None):
        return MolmoPointImageGetItem(field_mapping=field_mapping)

    def _run_single_for_object(self, img: Image.Image, obj: str) -> list:
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
                for point in self._run_single_for_object(img, obj):
                    _obj_id, _img_num, x, y = point
                    accumulated[i].append(
                        Keypoint(
                            label=obj,
                            points=[[float(x) / width, float(y) / height]],
                        )
                    )

        return [Keypoints(keypoints=kps) for kps in accumulated]

    def predict(self, arg, sample=None) -> Keypoints:
        pil_image = arg if isinstance(arg, Image.Image) else Image.fromarray(arg)
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
    - ``"pointing"`` — identifies objects on sparse frames at ``max_fps=2``.
      Prompt: ``"Point to the {obj}."``.

    Both return ``{frame_num: fo.Keypoints}`` where each ``fo.Keypoint`` has:
    - ``label``: object name from the prompt
    - ``index``: integer object ID from the model
    - ``points``: normalized ``[[x, y]]`` in [0, 1]

    FiftyOne routing: uses the DataLoader pathway (SupportsGetItem +
    TorchModelMixin) when available. The ``predict()`` method also handles
    filepath strings and video-reader objects so the direct predict pathway
    works too.

    Args:
        model_path: Local directory or HuggingFace repo ID.
        prompt: Global object(s) to locate.
        operation: ``"tracking"`` (default) or ``"pointing"``.
        max_fps: Override default fps (10 for tracking, 2 for pointing).
        **kwargs: Ignored.
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
        logger.info(f"Loading MolmoPoint-Vid processor from {self.model_path}")
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
    def _get_fps(metadata) -> Optional[float]:
        """Return frame_rate from a VideoMetadata object, or None."""
        if metadata is not None:
            fps = getattr(metadata, "frame_rate", None)
            if fps:
                return float(fps)
        return None

    def _run_video_inference_for_object(self, video_path: str, obj: str) -> tuple:
        """Run one generation pass for *obj* on a single video.

        Uses the two-step native API: ``process_vision_info`` extracts frames
        and kwargs, then ``processor(videos=..., return_pointing_metadata=True)``
        builds inputs with ``metadata["timestamps"]`` and ``metadata["video_size"]``
        correctly populated.

        Returns:
            ``(points, video_size)`` — points is a list of
            ``[point_id, timestamp_s, x_px, y_px]``.
        """
        from molmo_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_prompt(obj)},
                    {"type": "video", "video": video_path, "max_fps": self._max_fps},
                ],
            }
        ]

        _, videos, video_kwargs = process_vision_info(messages)
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

    def _build_frame_dict(self, video_path: str, objects: List[str], fps: float) -> Dict:
        """Inference for all objects → ``{frame_num: fo.Keypoints}``."""
        frame_kp_lists: Dict[int, List[Keypoint]] = {}

        for obj in objects:
            points, video_size = self._run_video_inference_for_object(
                video_path, obj
            )

            width = float(video_size[0])
            height = float(video_size[1])

            for point in points:
                point_id, timestamp, x_px, y_px = point
                frame_num = max(1, round(float(timestamp) * fps))
                frame_kp_lists.setdefault(frame_num, []).append(
                    Keypoint(
                        label=obj,
                        index=int(point_id),
                        points=[[
                            max(0.0, min(1.0, float(x_px) / width)),
                            max(0.0, min(1.0, float(y_px) / height)),
                        ]],
                    )
                )

        return {fn: Keypoints(keypoints=kps) for fn, kps in frame_kp_lists.items()}

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict_all(self, batch, preprocess=None, samples=None) -> List[Dict]:
        """Batch inference over videos.

        Each item in *batch* is a dict from ``MolmoPointVideoGetItem`` with
        ``"filepath"``, ``"prompt"``, and ``"metadata"`` keys.

        Returns ``{frame_num: fo.Keypoints}`` dicts, one per video.
        FiftyOne writes integer-keyed dicts as frame-level labels.
        """
        results = []

        for i, item in enumerate(batch):
            video_path = item["filepath"]
            sample_prompt = item.get("prompt")
            item_metadata = item.get("metadata")

            # Fall back to sample object if metadata not in batch item
            if item_metadata is None and samples is not None and i < len(samples):
                item_metadata = getattr(samples[i], "metadata", None)

            # Prompt: batch item → sample field fallback → global
            if sample_prompt is None:
                field_name = self._get_field()
                sample = samples[i] if samples is not None and i < len(samples) else None
                if field_name is not None and sample is not None:
                    if sample.has_field(field_name):
                        sample_prompt = sample.get_field(field_name)

            objects = self._resolve_objects(sample_prompt)

            if not objects:
                logger.warning(
                    "No objects resolved for '%s'. "
                    "Set model.prompt or pass prompt_field= to apply_model().",
                    video_path,
                )
                results.append({})
                continue

            fps = self._get_fps(item_metadata)
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

        Normalises *arg* (filepath string, video-reader object, or dict from
        GetItem) into the dict format expected by ``predict_all``, then
        delegates. This makes the method work regardless of which pathway
        FiftyOne uses to call it.

        Args:
            arg: Filepath string, video-reader with ``.inpath``, or a dict
                with ``"filepath"`` (as returned by ``MolmoPointVideoGetItem``).
            sample: FiftyOne sample for per-sample prompts and metadata.

        Returns:
            ``{frame_num: fo.Keypoints}`` dict, or ``{}`` if no objects resolved.
        """
        if isinstance(arg, dict):
            batch_item = arg
        else:
            filepath = arg if isinstance(arg, str) else arg.inpath

            # Resolve prompt from sample via needs_fields
            sample_prompt = None
            field_name = self._get_field()
            if field_name is not None and sample is not None:
                if sample.has_field(field_name):
                    sample_prompt = sample.get_field(field_name)

            batch_item = {
                "filepath": filepath,
                "prompt": sample_prompt,
                "metadata": getattr(sample, "metadata", None) if sample else None,
            }

        return self.predict_all(
            [batch_item],
            samples=[sample] if sample is not None else None,
        )[0]
