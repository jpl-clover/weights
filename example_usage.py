import os
import torch
import torchvision.models

# Weights can be loaded locally
dirname = os.path.dirname(__file__)
relative_path_to_checkpoint = "./self-supervised-distillation-for-computer-vision-onboard-planetary-robots/resnet18_distilled_from_r101_1x_sk0_finetuned_on_100pctMSL.tar"
checkpoint = os.path.join(dirname, relative_path_to_checkpoint)
state_dict = torch.load(checkpoint)
model = torchvision.models.resnet18().load_state_dict(state_dict)

# Weights can also be loaded from a URL using torch hub, see hubconf.py for list of supported models
model = torch.hub.load('jpl-clover/weights:main', 'resnet18_distilled_from_r101_1x_sk0_finetuned_on_100pctMSL', force_reload=True)

