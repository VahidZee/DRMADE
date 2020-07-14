def resnet_encoder_generator(resnet: str = 'resnet18', pretrained=False):
    def wrapper(encoder):
        import importlib
        from torch.nn import Linear, Conv2d
        model = getattr(importlib.import_module('torchvision.models'), resnet)(pretrained=pretrained)
        model.conv1 = Conv2d(
            in_channels=encoder.input_shape[0], out_channels=model.conv1.out_channels, stride=model.conv1.stride,
            bias=model.conv1.bias is not None, kernel_size=model.conv1.kernel_size, padding=model.conv1.padding,
            padding_mode=model.conv1.padding_mode, dilation=model.conv1.dilation)
        model.fc = Linear(model.fc.in_features, encoder.latent_size, encoder.bias)
        return model, '{}{}-{}'.format('[var]' if encoder.variational else '', resnet, encoder.latent_size)

    return wrapper
