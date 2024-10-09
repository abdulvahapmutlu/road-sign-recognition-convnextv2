from timm import create_model

def get_model(pretrained=True):
    model = create_model('convnextv2_tiny', pretrained=pretrained, num_classes=9)
    return model
