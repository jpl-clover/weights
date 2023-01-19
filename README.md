# weights
Repository for publicly sharing model weights related to CLOVER publications.

This repo stores neural network weights using Git Large File Storage (LFS), which allows files up to 100MB to be stored. Currently, ```.tar``` files are tracked for storage with Git LFS in this repo.

## File structure

self-supervised-distillation-for-computer-vision-onboard-planetary-robots
* ```resnet18_distilled_from_r101_1x_sk0_finetuned_on_100pctMSL.tar```. The best performing ResNet-18 model as benchmarked on the MSL v2.1 classification dataset, distilled from a r101_1x_sk0, with a fully connected layer optimized for classifying the MSL v2.1 dataset.
* ```resnet18_encoder.layer4_deeplab_distilled-from-r152_1x_sk0.tar```. The best performing ResNet-18 model as benchmarked on the AI4Mars segmentation dataset, distilled from a r151_1x_sk0, with a head optimized for segmenting the AI4Mars dataset.

## How to load these weights?

See ```example_usage.py```.

## Classification example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jpl-clover/weights/blob/devel/classification_demo.ipynb)