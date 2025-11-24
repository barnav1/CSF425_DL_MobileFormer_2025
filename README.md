# Mobile-Former: PyTorch Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Context
This repository contains a PyTorch implementation of **Mobile-Former**, developed as part of the group project for the course **CS F425: Deep Learning**.

We have selected the paper **"Mobile-Former: Bridging MobileNet and Transformer" (CVPR 2022)**. This architecture parallelizes MobileNet and Transformer blocks with a bidirectional bridge, allowing for efficient global-local feature interaction.

**Target Configuration:**
* **Model Variant:** Mobile-Former 26M (26 Million FLOPs)
* **Dataset:** Mini-ImageNet (100 Classes)

---

## Directory Structure

```text
.
├── mobile_former/
│   ├── blocks.py           # Core components: Mobile-Former Block, Bridges, Dynamic ReLU
│   ├── mobile_former.py    # Main model architecture & config for 26M variant
│   └── utils.py            # Utilities: DropPath, ChannelShuffle, Helper functions
├── main.py                 # Main training script (Training, Validation, Testing)
├── optimize_hyp_params.py  # Hyperparameter optimization using Optuna
├── requirements.txt        # Project dependencies
└── README.md
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    The project relies on `torch`, `torchvision`, `datasets`, and `optuna`.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Training the Model
To train the Mobile-Former 26M model on Mini-ImageNet, run the `main.py` script. This script handles:
* Loading the **Mini-ImageNet** dataset via Hugging Face.
* Initializing the model with the 26M configuration.
* Adapting the classifier head for 100 classes.
* Training for 225 epochs and saving checkpoints to `./saved_models/`.

```bash
python main.py
```

### 2. Hyperparameter Optimization
To find the optimal learning rate, weight decay, and dropout values, run the optimization script. This uses **Optuna** to prune unpromising trials early.

```bash
python optimize_hyp_params.py
```

---

## Implementation Details

Our implementation faithfully reconstructs the parallel design of Mobile-Former using the following components:

### The Mobile-Former Block
Located in `mobile_former/blocks.py`, this block orchestrates the parallel flow of data:
1.  **Mobile Sub-block:** Takes local feature maps (images) as input.
2.  **Former Sub-block:** Takes global tokens as input.
3.  **Bridges:**
    * **Mobile $\to$ Former:** The `LocalToGlobalInterface` allows tokens to gather context from the image features.
    * **Former $\to$ Mobile:** The `GlobalToLocalInterface` allows the image features to be modulated by the global tokens.

### Dynamic Parametrization
We utilize `DynamicActivator` and `TokenParamGenerator` to implement **Dynamic ReLU (DyReLU)**. Instead of static weights, the activation parameters are generated dynamically from the global tokens, allowing the model to adapt its non-linearity based on the global context.

### Model Configuration (26M)
The architecture is defined in `mobile_former/mobile_former.py`. We use the specific configuration for the 26M FLOPs variant:
* **Tokens:** 6 learnable global tokens.
* **Token Dimension:** 192.
* **Backbone:** 11 layers with varying expansion ratios and resolutions.

---

## Dataset

We utilize the **Mini-ImageNet** dataset (via `timm/mini-imagenet` on Hugging Face).
* **Classes:** 100
* **Input Size:** Resized to $224 \times 224$
* **Normalization:** Standard ImageNet mean/std.

---

## Team Members

* Ananya Veeraraghavan
* Arnav Adivi
* Prakhar Gupta
* Ria Arora
* Yash Chaphekar

---

## References

1.  **Paper:** Chen, Y., Dai, X., et al. "Mobile-Former: Bridging MobileNet and Transformer." *CVPR 2022*.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.