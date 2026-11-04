# image_ohem_selector

Select image samples with the highest user-defined per-sample loss. This is an
offline hard-example selector: it scores the complete dataset and retains the
highest-loss `top_ratio` or `topk` samples.

The selector performs these steps:

1. Load the user model by calling `model_factory(**model_kwargs)`.
2. Load each sample's images and pass batches to `score_fn`.
3. Require exactly one finite loss value for every sample.
4. Store the loss in `__dj__stats__.image_ohem_loss` by default.
5. Rank all samples by loss in descending order and retain the requested amount.

> [!IMPORTANT]
> This Beta selector currently supports the local dataset execution path only.
> It scores batches serially at dataset level and is not supported by the Ray
> executor. It implements offline hard-example selection, not OHEM inside a
> model training loop.

Type 算子类型: **selector**

Tags 标签: image, gpu

## Input data

The selector does not prescribe a label schema. The user-defined `score_fn`
reads task-specific labels, bounding boxes, masks, or other fields directly from
each sample. A classification sample may look like:

```json
{"images":["/data/images/cat.jpg"],"label":282}
```

`images` is normally a list. A sample may contain multiple images, but
`score_fn` must combine them into one sample-level loss.

## User functions

`score_file` must be a Python file containing a score function and, optionally,
a model factory:

```python
def model_factory(**model_kwargs):
    return model


def score_fn(model, samples, images, device, **score_kwargs):
    return per_sample_losses
```

The arguments passed to `score_fn` are:

- `model`: the object returned by `model_factory`, or `None` if no factory is
  provided.
- `samples`: a list of the original sample dictionaries in the current batch.
- `images`: a list of image lists. `images[i]` contains the loaded PIL images
  for `samples[i]`.
- `device`: the configured device, such as `cpu` or `cuda`.
- `score_kwargs`: additional values from the operator configuration.

The return value must contain exactly one finite numeric loss per sample. A
PyTorch tensor with shape `[batch_size]` or a list of floats is accepted. A
single batch-mean loss is not accepted.

## Classification example

The following `/data/ohem_functions.py` example uses a pretrained ImageNet
ResNet-18. The input `label` values must use the ImageNet class indices from 0 to
999.

```python
import torch
from torchvision.models import ResNet18_Weights, resnet18


weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()


def model_factory():
    return resnet18(weights=weights)


def score_fn(model, samples, images, device):
    image_tensors = [transform(sample_images[0]) for sample_images in images]
    inputs = torch.stack(image_tensors).to(device)
    labels = torch.tensor(
        [sample["label"] for sample in samples],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits = model(inputs)
        return torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
        )
```

Use it in a Data Juicer recipe:

```yaml
process:
  - image_ohem_selector:
      score_file: /data/ohem_functions.py
      model_function: model_factory
      score_function: score_fn
      top_ratio: 0.3
      batch_size: 32
      device: cuda
      image_key: images
      image_bytes_key: image_bytes
      loss_field: image_ohem_loss
```

This configuration retains the 30% of samples with the highest loss. To retain
a fixed number instead, use:

```yaml
top_ratio: null
topk: 10000
```

When both `top_ratio` and `topk` are specified, the smaller resulting sample
count is used.

## Image loading

For every image position, the selector first checks `image_bytes_key`. Valid
bytes are loaded directly; a missing, invalid, or `None` entry falls back to the
corresponding path in `image_key`.

```json
{
  "images": ["unavailable-a.jpg", "/data/images/b.jpg"],
  "image_bytes": ["<bytes for image A>", null]
}
```

In this example, the first image is loaded from bytes and the second image is
loaded from `/data/images/b.jpg`.

## Output data

The selected samples retain their computed loss:

```json
{
  "images": ["/data/images/cat.jpg"],
  "label": 282,
  "__dj__stats__": {
    "image_ohem_loss": 1.37
  }
}
```

## Parameter configuration

| Parameter | Default | Description |
|---|---:|---|
| `score_file` | `""` | Python file containing the user functions. |
| `score_function` | `score_fn` | Per-sample loss function name. |
| `model_function` | `model_factory` | Optional model factory name. |
| `top_ratio` | `None` | Fraction of highest-loss samples to retain. |
| `topk` | `None` | Maximum number of highest-loss samples to retain. |
| `batch_size` | `8` | Number of samples passed to `score_fn` at once. |
| `image_key` | `images` | Field containing image paths. |
| `image_bytes_key` | `image_bytes` | Image bytes field; missing entries fall back to paths. |
| `loss_field` | `image_ohem_loss` | Loss key inside `__dj__stats__`. |
| `device` | `auto` | Device passed to user functions. |
| `model_kwargs` | `{}` | Keyword arguments for `model_factory`. |
| `score_kwargs` | `{}` | Extra keyword arguments for `score_fn`. |

## Common errors

- Returning one batch-mean loss instead of one loss per sample.
- Returning NaN or infinite loss values.
- Using labels that do not match the model's class indices.
- Returning a loss list whose length differs from the batch size.
- Referencing a `score_file`, checkpoint, or image path unavailable on the
  machine running Data Juicer.

## Related links

- [Source code](../../../data_juicer/ops/selector/image_ohem_selector.py)
- [Unit test](../../../tests/ops/selector/test_image_ohem_selector.py)
- [Operator list](../../Operators.md)
