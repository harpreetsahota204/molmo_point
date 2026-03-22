# MolmoPoint — FiftyOne Remote Zoo Model

![image/png](molmo_point_image.gif)

MolmoPoint is a vision-language model from the Allen Institute for AI that locates objects in images by **pointing** — returning precise pixel coordinates — rather than generating bounding boxes. Given a natural language description like `"Point to the boats"`, MolmoPoint finds every matching instance in the image and returns a set of keypoints, one per object.

## What makes it different

Most grounding models predict coordinates as text output (e.g. `"[412, 308]"`), which forces the model to memorise an arbitrary coordinate system and uses many tokens per point. MolmoPoint instead emits special **grounding tokens** that directly attend to the image's visual tokens and select the patch that contains the target, then refines the prediction sub-patch by sub-patch down to ~5 pixel precision. This gives it:

- **Higher accuracy** — state-of-the-art on PointBench (70.7%) and PixMo-Points, beating much larger models including Gemini 2.5 Pro
- **Faster inference** — 3 tokens per point instead of ~8, meaning faster decoding especially when many objects are present
- **Consistent resolution** — ~5 pixel precision regardless of input image size, including high-resolution images

## Available models

| Model | Best for |
|---|---|
| `allenai/MolmoPoint-8B` | General-purpose pointing in natural images — objects, animals, people, scene elements |
| `allenai/MolmoPoint-Img-8B` | UI elements and interactive components in screenshots and GUIs |

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
pip install fiftyone "transformers<5.0" torch pillow huggingface-hub
```

> **Note:** MolmoPoint requires `transformers<5.0`. It was developed and tested against `transformers==4.57.1`. Installing `transformers>=5.0` will likely cause errors during model loading or inference.

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

If your dataset already has ground-truth labels, you can derive a per-image object list from them and pass it straight to the model via `prompt_field`. No manual `needs_fields` setup required.

```python

import fiftyone.zoo as foz

# Load a dataset
dataset = foz.load_zoo_dataset("quickstart")

# Derive unique object labels per sample from existing ground-truth detections
uniqe_objects_per_sample = [list(set(labels)) for labels in dataset.values("ground_truth.detections.label")]

dataset.set_values(
    "uniqe_objects_per_sample",
    uniqe_objects_per_sample
)

model = foz.load_zoo_model("allenai/MolmoPoint-8B")

dataset.apply_model(
    model,
    prompt_field="uniqe_objects_per_sample",   # MolmoPoint reads per-sample objects from this field
    label_field="molmo_points",
    batch_size=4,
    num_workers=2,
)
```

This is useful for verifying or augmenting existing annotations — each image is prompted only with the object classes that actually appear in it.

## Loading the GUI model

For screenshots and UI tasks, swap in `MolmoPoint-Img-8B`:

```python
model = foz.load_zoo_model("allenai/MolmoPoint-Img-8B")
model.prompt = ["submit button", "search bar", "navigation menu"]

dataset.apply_model(model, label_field="ui_points")
```

## Performance notes

MolmoPoint's custom forward pass processes **one image at a time** internally (a constraint of the current model implementation). The `batch_size` and `num_workers` arguments to `apply_model` still control how many images are loaded in parallel by the DataLoader workers, which significantly improves throughput for I/O-bound workflows.

```python
dataset.apply_model(
    model,
    label_field="molmo_points",
    num_workers=4,   # parallel image loading
)
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
