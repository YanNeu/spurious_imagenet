
def get_last_layer(model_id, model, multi_gpu=False, return_gradcam=False):
    last_layer = None

    if multi_gpu:
        if model_id != 'resnet_salient_imagenet' and model_id != 'robust_resnet':
            model = model.module
        

    if model_id == "robust_resnet":
        last_layer = model.model.model.fc
        gradcam_layers = [model.model.model.layer4[-1]]
    elif model_id in ["resnet50", "resnet101", "ssl_resnet50", "swsl_resnet50"]:
        last_layer = model.model.fc
        gradcam_layers = [model.model.layer4[-1]]
    elif model_id == "resnet_salient_imagenet":
        last_layer = model.model.fc
        gradcam_layers = [model.model.layer4[-1]]
    elif "tf_efficientnet" in model_id:
        last_layer = model.model.classifier
        gradcam_layers = [model.model.conv_head]
    elif "resnetv2" in model_id:
        last_layer = model.model.head.fc
        gradcam_layers = None
    elif "convnext" in model_id:
        last_layer = model.model.head.fc
        gradcam_layers = [model.model.stages[3].blocks[2].conv_dw]
    elif "deit" in model_id:
        last_layer = model.model.head
        gradcam_layers = None
    elif "swin" in model_id:
        last_layer = model.model.head.fc
        gradcam_layers = None
    elif "beit" in model_id:
        last_layer = model.model.head
        gradcam_layers = None
    elif "eva" in model_id:
        last_layer = model.model.head
        gradcam_layers = None
    elif "volo" in model_id:
        last_layer = model.model.head
        gradcam_layers = None
    elif "vit" in model_id:
        last_layer = model.model.head
        gradcam_layers = None
    elif "resnext" in model_id:
        last_layer = model.model.fc
        gradcam_layers = None
    if return_gradcam:
        return last_layer, gradcam_layers
    
    return last_layer
