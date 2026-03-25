import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import MolmoPointImageModel, MolmoPointVideoModel  # noqa: F401

logger = logging.getLogger(__name__)


def download_model(model_name, model_path, **kwargs):
    """Download MolmoPoint model weights from HuggingFace Hub.

    Args:
        model_name: HuggingFace repo ID declared in the manifest's
            ``base_name`` field (e.g. ``"allenai/MolmoPoint-8B"``).
        model_path: Local directory to download into.
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name, model_path, **kwargs):
    """Instantiate and return a MolmoPoint model.

    The ``media_type`` kwarg controls which model class is returned:

    - ``"image"`` (default): :class:`MolmoPointImageModel` — pointing on
      images. Returns sample-level ``fo.Keypoints``.
    - ``"video"``: :class:`MolmoPointVideoModel` — pointing or tracking on
      videos. Supports an additional ``operation`` kwarg:

        - ``"tracking"`` (default): follows objects frame-by-frame.
          Returns frame-level ``fo.Keypoints``. Requires
          ``dataset.compute_metadata()`` for accurate frame numbers.
        - ``"pointing"``: identifies objects on sparse frames.
          Returns sample-level ``fo.Keypoints``.

    Args:
        model_name: HuggingFace repo ID.
        model_path: Local directory containing the downloaded model weights.
        **kwargs: Forwarded to the model class. Common kwargs:

            - ``prompt`` (``str | list[str]``): Objects to locate.
            - ``media_type`` (``str``): ``"image"`` or ``"video"``.
            - ``operation`` (``str``, video only): ``"tracking"`` or
              ``"pointing"``.
            - ``max_fps`` (``int``, video only): Override default fps.

    Returns:
        :class:`MolmoPointImageModel` or :class:`MolmoPointVideoModel`
    """
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(
            f"Invalid model_path: '{model_path}'. "
            f"Ensure the model has been downloaded via "
            f"fiftyone.zoo.download_zoo_model('{model_name}')"
        )

    media_type = kwargs.pop("media_type", "image")
    logger.info(
        f"Loading MolmoPoint model from {model_path} (media_type={media_type})"
    )

    if media_type == "video":
        return MolmoPointVideoModel(model_path=model_path, **kwargs)
    return MolmoPointImageModel(model_path=model_path, **kwargs)


def resolve_input(model_name, ctx):
    """Define operator UI inputs for FiftyOne operators.

    Args:
        model_name: The model name.
        ctx: Operator ``ExecutionContext``.

    Returns:
        ``types.Property``
    """
    inputs = types.Object()

    media_type_choices = types.RadioGroup()
    media_type_choices.add_choice("image", label="Image")
    media_type_choices.add_choice("video", label="Video")

    inputs.enum(
        "media_type",
        media_type_choices.values(),
        label="Media Type",
        description="Whether to run on images or video.",
        default="image",
        view=media_type_choices,
    )

    inputs.str(
        "objects",
        label="Objects",
        description=(
            "Comma-separated list of objects to locate, "
            "e.g. 'boat, person, car'."
        ),
        required=True,
    )

    operation_choices = types.RadioGroup()
    operation_choices.add_choice("tracking", label="Tracking")
    operation_choices.add_choice("pointing", label="Pointing")

    inputs.enum(
        "operation",
        operation_choices.values(),
        label="Video Operation",
        description=(
            "Tracking follows objects across frames (frame-level keypoints, "
            "max_fps=10). Pointing identifies objects on sparse frames "
            "(sample-level keypoints, max_fps=2). Video only."
        ),
        default="tracking",
        view=operation_choices,
    )

    inputs.int(
        "max_fps",
        label="Max FPS (video only)",
        description=(
            "Maximum frames per second to sample from the video. "
            "Defaults: 10 for tracking, 2 for pointing."
        ),
        required=False,
    )

    inputs.str(
        "output_field",
        label="Output Field",
        description="Name of the field to store the predicted keypoints in.",
        required=True,
    )

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=(
            "Delegate this operation to a background service. "
            "Run `fiftyone delegated launch` in your terminal first."
        ),
        view=types.CheckboxView(),
    )

    inputs.view_target(ctx)

    return types.Property(inputs)
