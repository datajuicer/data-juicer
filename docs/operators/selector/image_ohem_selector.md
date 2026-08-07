# image_ohem_selector

Select image samples with the highest user-defined per-sample loss. This is an
offline OHEM selector: it scores the complete dataset and retains the highest-loss
`top_ratio` or `topk` samples.

The user Python file may define:

```python
def model_factory(checkpoint):
    model = load_your_model(checkpoint)
    return model


def score_fn(model, samples, images, device, **kwargs):
    # images is a list of image lists, one list for each sample.
    # Return exactly one numeric loss for every sample.
    return compute_per_sample_losses(model, samples, images, device)
```

Example configuration:

```yaml
process:
  - image_ohem_selector:
      score_file: /path/to/ohem_functions.py
      model_function: model_factory
      score_function: score_fn
      model_kwargs:
        checkpoint: /path/to/model.pt
      top_ratio: 0.3
      batch_size: 32
      device: cuda
```

When both `top_ratio` and `topk` are specified, the smaller resulting sample count
is used. The computed loss is stored in `__dj__stats__.image_ohem_loss` by default.

Type 算子类型: **selector**

Tags 标签: image, gpu

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `score_file` | `""` | Python file containing the user functions. |
| `score_function` | `score_fn` | Per-sample loss function name. |
| `model_function` | `model_factory` | Optional model factory name. |
| `top_ratio` | `None` | Fraction of highest-loss samples to retain. |
| `topk` | `None` | Maximum number of highest-loss samples to retain. |
| `batch_size` | `8` | Number of samples passed to `score_fn` at once. |
| `loss_field` | `image_ohem_loss` | Loss key inside `__dj__stats__`. |
| `device` | `auto` | Device string passed to the user functions. |
| `model_kwargs` | `{}` | Keyword arguments for `model_factory`. |
| `score_kwargs` | `{}` | Extra keyword arguments for `score_fn`. |
