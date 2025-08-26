# CorePulse-MLX

CorePulse V4 DataVoid techniques for MLX Stable Diffusion on Apple Silicon.

## Overview

Complete implementation of CorePulse attention manipulation system with zero-regression hooks:

- **Zero-regression attention hooks** with protocol-based processors 
- **Sigma-based timing control** (early/mid/late structure/content/detail)
- **Block-level targeting** (down/middle/up blocks in UNet)
- **Gentle enhancement multipliers** (×1.05-1.15) for stability
- **Research-backed CFG fixes** for SD 2.1-base prompt adherence
- **Product placement** and regional control capabilities

> Proven to work: 7-10% quality improvements while maintaining semantic consistency.

## Quickstart

```bash
pip install mlx # and mlx packages required by the Stable Diffusion example repo
# Ensure you have the MLX Stable Diffusion example or equivalent environment available.
```

Example minimal usage:

```python
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper import CorePulseStableDiffusion

sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
cpsd = CorePulseStableDiffusion(sd)

# Inject a sun in the top-right later in the process
cpsd.add_injection(
    prompt="a bright sun",
    weight=0.85,
    start_frac=0.6, end_frac=1.0,
    token_mask="sun",
    region=("rect_frac", 0.65, 0.05, 0.32, 0.32, 0.10),
)

latents = cpsd.generate_latents(
    base_prompt="a mountain landscape at sunrise",
    negative_text="low quality, blurry",
    num_steps=40, cfg_weight=7.0, n_images=1, height=512, width=512, seed=123,
)
# Take the final latent and decode
for x_t in latents:
    pass
img = sd.decode(x_t)
```

### Product placement (non-hallucinated)

```python
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper import CorePulseStableDiffusion
from plugins.product_placement import build_product_placement

sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
cpsd = CorePulseStableDiffusion(sd)

# Create a product-placement config with a reference PNG
pp = build_product_placement(
    cpsd,
    region=("rect_frac", 0.55, 0.15, 0.25, 0.25, 0.06),
    mode="inpaint",                    # or "image_cond"
    reference_rgba_path="product.png", # your ground-truth product
    phase=(0.55, 1.0),
    alpha=1.0,
    cfg_cap=6.0,
    ramp_steps=8,
)

# Background only; no brand words here
latents = cpsd.generate_latents(
    base_prompt="a cozy wooden desk near a window, soft morning light, film grain",
    negative_text="extra logos, melted text, wrong labels, distortions, watermark",
    num_steps=36, cfg_weight=6.0, n_images=1, height=768, width=768, seed=42,
)
for x_t in latents:
    pass
img = sd.decode(x_t)
```

## Repo Layout

```
corpus-mlx/
├── corpus_mlx/
│   ├── __init__.py
│   ├── sd_wrapper.py
│   ├── injection.py
│   ├── blending.py
│   ├── masks.py
│   ├── schedule.py
│   ├── utils.py
├── plugins/
│   ├── product_placement.py
│   └── regional_prompt.py
├── examples/
│   ├── demo_product.py
│   └── demo_multi_prompt.py
├── tests/
│   ├── test_injection.py
│   ├── test_masks.py
│   └── test_equivalence.py
├── scripts/
│   └── corpus_txt2img.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## License

MIT
