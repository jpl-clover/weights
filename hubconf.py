import torch
import torchvision.models


def resnet18_distilled_from_r101_1x_sk0_finetuned_on_100pctMSL():
    checkpoint_url = "https://storage.googleapis.com/seed-aeroconf/resnet18_distilled_from_r101_1x_sk0_finetuned_on_100pctMSL.tar"
    model = torchvision.models.resnet18()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
    )
    return model


def resnet18_encoder_layer4_deeplab_distilled_from_r152_1x_sk0():
    checkpoint_url = "https://storage.googleapis.com/seed-aeroconf/resnet18_encoder.layer4_deeplab_distilled-from-r152_1x_sk0.tar"
    model = torchvision.models.resnet18()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
    )
    return model
