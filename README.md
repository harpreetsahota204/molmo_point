# MolmoPoint — FiftyOne Remote Zoo Model

![image/png](molmo_point_image.gif)

MolmoPoint is a vision-language model from the Allen Institute for AI that locates and tracks objects in images and videos by **pointing** — returning precise pixel coordinates — rather than generating bounding boxes. Given a natural language description like `"Point to the boats"`, MolmoPoint finds every matching instance and returns a set of keypoints, one per object.

## What makes it different

Most grounding models predict coordinates as text output (e.g. `"[412, 308]"`), which forces the model to memorise an arbitrary coordinate system and uses many tokens per point. MolmoPoint instead emits special **grounding tokens** that directly attend to the image's visual tokens and select the patch that contains the target, then refines the prediction sub-patch by sub-patch down to ~5 pixel precision. This gives it:

- **Higher accuracy** — state-of-the-art on PointBench (70.7%) and PixMo-Points, beating much larger models including Gemini 2.5 Pro
- **Faster inference** — 3 tokens per point instead of ~8, meaning faster decoding especially when many objects are present
- **Consistent resolution** — ~5 pixel precision regardless of input image size, including high-resolution images

## Available models

| Model | Best for |
|---|---|
| `allenai/MolmoPoint-8B` | General-purpose pointing in natural images and videos |
| `allenai/MolmoPoint-Img-8B` | UI elements and interactive components in screenshots and GUIs |
| `allenai/MolmoPoint-Vid-4B` | Lightweight 4B model optimised for video pointing and tracking |

## When to use MolmoPoint

MolmoPoint is a strong choice when you need to **locate objects without labelled bounding boxes**. Good use cases include:

- **Zero-shot object localization** — find any object described in natural language, with no prior annotation or fine-tuning
- **Counting via pointing** — the model returns one point per instance, so the count is simply the number of returned keypoints
- **Referring expressions** — point to objects described by relationship or attribute, e.g. `"the red car on the left"`, `"the person holding an umbrella"`
- **Weak supervision bootstrapping** — use the returned keypoints as rough center-point annotations to seed a downstream detector or segmentation model
- **GUI interaction & automation** — `MolmoPoint-Img-8B` finds buttons, fields, and other interactive elements in screenshots by their natural-language description

MolmoPoint is **not** a detection model — it returns center points, not bounding boxes. If you need tight boxes, consider using the keypoints as seeds for a downstream model.

## Installation

```bash
pip install fiftyone "transformers<5.0" torch pillow huggingface-hub molmo-utils
```

> **Note:** MolmoPoint requires `transformers<5.0`. It was developed and tested against `transformers==4.57.1`. Installing `transformers>=5.0` will likely cause errors during model loading or inference.
>
> `molmo-utils` is required for video inference. It handles frame extraction and the two-step video preprocessing pipeline the model expects.

## Quickstart

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Register the model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/molmo_point",
    overwrite=True
)

# Load a dataset
dataset = foz.load_zoo_dataset("quickstart")

# Load the model (weights are downloaded on first use)
model = foz.load_zoo_model("allenai/MolmoPoint-8B")

# Tell the model what to point at
model.prompt = ["person", "animal", "drink", "food", "vehicle"]

# Run on the dataset
dataset.apply_model(
    model,
    label_field="molmo_points",
    batch_size=4,
    num_workers=2,
)

session = fo.launch_app(dataset)
```

## What gets added to your dataset

`apply_model` stores a [`fo.Keypoints`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Keypoints) label on each sample at the field name you specify (e.g. `"molmo_points"`). Each `Keypoints` object contains one [`fo.Keypoint`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Keypoint) per located instance, with:

- **`label`** — the object description that produced this point (e.g. `"person"`)
- **`points`** — a single `[[x, y]]` coordinate pair, normalized to `[0, 1]` relative to the image dimensions

If no instances of a prompted object are found in an image, no keypoint is added for that object on that sample.

## Setting the prompt

**Global prompt (same for all samples):**

```python
# As a list
model.prompt = ["boat", "person", "life jacket"]

# Or as a comma-separated string
model.prompt = "boat, person, life jacket"
```

**Per-sample prompt from a dataset field:**

If your dataset already has ground-truth labels, you can derive a per-image object list from them and pass it straight to the model via `prompt_field`.

```python

import fiftyone.zoo as foz

# Load a dataset
dataset = foz.load_zoo_dataset("quickstart")

# Derive unique object labels per sample from existing ground-truth detections
unique_objects_per_sample = [list(set(labels)) for labels in dataset.values("ground_truth.detections.label")]

dataset.set_values("unique_objects_per_sample", unique_objects_per_sample)

model = foz.load_zoo_model("allenai/MolmoPoint-8B")

dataset.apply_model(
    model,
    prompt_field="unique_objects_per_sample",
    label_field="molmo_points",
    batch_size=4,
    num_workers=2,
)
```

This is useful for verifying or augmenting existing annotations. Each image is prompted only with the object classes that actually appear in it.

## Loading the GUI model

For screenshots and UI tasks, swap in `MolmoPoint-Img-8B`:

```python
model = foz.load_zoo_model("allenai/MolmoPoint-Img-8B")
model.prompt = ["submit button", "search bar", "navigation menu"]

dataset.apply_model(model, label_field="ui_points")
```

## Video tracking and pointing

MolmoPoint supports two video operations, controlled by the `operation` parameter:

| Operation | Prompt pattern | Default `max_fps` | Output |
|---|---|---|---|
| `"tracking"` | `"Track the {obj}."` | 10 | Frame-level `fo.Keypoints` with object IDs, interpolated between detections |
| `"pointing"` | `"Point to the {obj}."` | 2 | Frame-level `fo.Keypoints` on sparse frames only |

### Important: call `compute_metadata()` first

The model converts the timestamps it returns into FiftyOne frame numbers using the video's frame rate. Without metadata, it falls back to 30 fps with a warning:

```python
dataset.compute_metadata()
```

### Video tracking quickstart

```python
import fiftyone as fo
import fiftyone.zoo as foz

foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/molmo_point",
    overwrite=True,
)

dataset = foz.load_zoo_dataset("quickstart-video")
dataset.compute_metadata()

model = foz.load_zoo_model("allenai/MolmoPoint-8B", media_type="video")
model.operation = "tracking"
model.prompt = ["person", "car", "dog"]

dataset.apply_model(
    model,
    label_field="tracking_keypoints",
    batch_size=1,
    num_workers=2,
)

session = fo.launch_app(dataset)
```

### What gets written to the dataset

For both operations, `apply_model` writes a `fo.Keypoints` to **each frame** that has at least one detection. Access them at `sample.frames[n]["<label_field>"]`.

Each `fo.Keypoint` contains:

- **`label`** — the object name from your prompt (e.g. `"person"`)
- **`index`** — integer object ID from the model — the same object keeps the same ID across frames, making it useful for tracking identity over time
- **`points`** — a single `[[x, y]]` coordinate pair, normalized to `[0, 1]`

In **tracking mode**, the model emits detections at up to `max_fps` frames per second, and the wrapper linearly interpolates positions between consecutive detections of the same object, so every frame between the first and last detection is filled. Gaps larger than one second are left empty to avoid bridging scene cuts or long occlusions.

### Video pointing

Pointing samples the video sparsely and is useful when you just want to confirm that an object is present somewhere in the video without dense per-frame tracking:

```python
model.operation = "pointing"
model.prompt = ["parked car", "pedestrian", "traffic light"]

dataset.apply_model(
    model,
    label_field="pointing_keypoints",
    batch_size=1,
    num_workers=2,
)
```

### Switching operations without reloading

The model stays on the GPU. You can freely change `operation`, `max_fps`, and `interp_max_gap` between runs:

```python
# Switch to pointing — max_fps automatically updates to 2
model.operation = "pointing"

# Explicitly pin max_fps — won't change when you switch operation
model.max_fps = 5

# Reset to automatic default for the current operation
model.max_fps = None

# Widen the interpolation gap limit for tracking (default is 1 second = fps frames)
model.interp_max_gap = 60  # bridge gaps up to 60 frames
```

### Per-sample prompts (video)

Works the same as images — store a list of object names on each sample and pass the field name via `prompt_field`:

```python
# Derive objects from existing ground-truth labels
sample_objects = [dataset.distinct("frames.detections.detections.label")] * len(dataset)
dataset.set_values("sample_objects", sample_objects)

model = foz.load_zoo_model("allenai/MolmoPoint-8B", media_type="video")
model.operation = "tracking"

dataset.apply_model(
    model,
    prompt_field="sample_objects",
    label_field="tracking_keypoints",
    batch_size=1,
    num_workers=2,
)
```

### Using the lightweight 4B video model

`MolmoPoint-Vid-4B` is a smaller model optimised specifically for video. Swap it in by changing the model name — everything else is identical:

```python
model = foz.load_zoo_model("allenai/MolmoPoint-Vid-4B", media_type="video")
model.operation = "tracking"
model.prompt = ["swimmer"]
```

## Citation

```bibtex
@article{clark2025molmopoint,
  title={MolmoPoint: Better Pointing for VLMs with Grounding Tokens},
  author={Clark, Christopher and Yang, Yue and Park, Jae Sung and Ma, Zixian and
          Zhang, Jieyu and Tripathi, Rohun and Salehi, Mohammadreza and Lee, Sangho and
          Anderson, Taira and Han, Winson and Krishna, Ranjay},
  year={2025}
}
```
