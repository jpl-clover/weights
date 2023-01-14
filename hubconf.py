import torch 
import torchvision.models

def resnet18_distilled_from_r101_1x_sk0_finetuned_on_100pctMSL():
    checkpoint_url = "https://jpl-clover.s3.amazonaws.com/weights/MSL+N5920+E0-100+mobilenetv3_large+LR1e-04+B64+Cdistilled_students_100k-Teacher_simclr_T-Epoch_200_Student_mobilenetv3_large_distill-Epoch_r101_2x_sk1-checkpoint_0199.pth.tar+2023-01-12_192328_checkpoint.tar"
    model = torchvision.models.resnet18()
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, progress=True))
    return model

def resnet18_encoder_layer4_deeplab_distilled_from_r152_1x_sk0():
    checkpoint_url = "https://github.com/jpl-clover/weights/blob/main/self-supervised-distillation-for-computer-vision-onboard-planetary-robots/resnet18_encoder.layer4_deeplab_distilled-from-r152_1x_sk0.tar"
    model = torchvision.models.resnet18()
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, progress=True))
    return model
