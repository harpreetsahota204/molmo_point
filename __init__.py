import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import MolmoPointModel

logger = logging.getLogger(__name__)


def download_model(model_name, model_path, **kwargs):
    """Download MolmoPoint model weights from HuggingFace Hub.

    Args:
        model_name: HuggingFace repo ID declared in the manifest's
            ``base_name`` field (e.g. ``"allenai/MolmoPoint-8B"``).
        model_path: Local directory to download into, declared in the
            manifest's ``base_filename`` field.
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name, model_path, **kwargs):
    """Instantiate and return a :class:`MolmoPointModel`.

    Args:
        model_name: HuggingFace repo ID.
        model_path: Local directory containing the downloaded model weights.
        **kwargs: Forwarded to :class:`MolmoPointModel`. Useful kwargs:

            - ``prompt`` (``str | list[str]``): Objects to point at, e.g.
              ``["boat", "person"]`` or ``"boat, person"``. Can also be set
              after loading via ``model.prompt = ...``.

    Returns:
        :class:`MolmoPointModel`
    """
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(
            f"Invalid model_path: '{model_path}'. "
            f"Ensure the model has been downloaded via "
            f"fiftyone.zoo.download_zoo_model('{model_name}')"
        )

    logger.info(f"Loading MolmoPoint model from {model_path}")
    return MolmoPointModel(model_path=model_path, **kwargs)


def resolve_input(model_name, ctx):
    """Define operator UI inputs for FiftyOne operators.

    Args:
        model_name: The model name.
        ctx: Operator ``ExecutionContext``.

    Returns:
        ``types.Property``
    """
    inputs = types.Object()

    inputs.str(
        "objects",
        label="Objects to point at",
        description=(
            "Comma-separated list of objects to locate, "
            "e.g. 'boat, person, car'. "
            "The model will return a keypoint for each instance found."
        ),
        required=True,
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
