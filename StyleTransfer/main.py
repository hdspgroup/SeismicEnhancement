import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torchvision.utils import save_image

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a, b, c * d)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers,
                               style_layers):
    normalization = Normalization(normalization_mean, normalization_std).to('cpu')
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps,
                       style_weight, content_weight):
    content_layers_default = ['conv_14']
    style_layers_default = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                     style_img, content_img, content_layers_default,
                                                                     style_layers_default)

    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    for step in range(num_steps):
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = sum([sl.loss for sl in style_losses])
            content_score = sum([cl.loss for cl in content_losses])

            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()

            if step % 50 == 0:
                print(f"Step {step}:")
                print(f"Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def preprocess_image(image_path, img_size):
    loader = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to('cpu', torch.float)


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer





def main(content_img,style_img):

    input_img = content_img.clone()
    cnn = models.vgg19(pretrained=True).features.to('cpu').eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to('cpu')
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to('cpu')
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img,
                                num_steps=100, style_weight=1e1, content_weight=1)
    print("Output Image Shape:", output.shape)  # Debug print to check shape
    save_image(output, 'x.jpg')
    return output
'''
if __name__ == '__main__':
    #sty = preprocess_image('../patch_58.jpg', 128)
    styl = preprocess_image('../patch_58.jpg', 128)
    #style_img = torch.stack([styl, sty], dim=0).squeeze()
    style_img = styl
    print(style_img.shape)
    ves = preprocess_image('../data/patch_9.jpg', 128).squeeze(0)
    ves1 = preprocess_image('../data/patch_9.jpg', 128).squeeze(0)

    content_img = torch.stack([ves, ves1], dim=0)
    print("Content Image Shape:", content_img.shape)  # Debug print to check shape
    main(content_img, style_img)
'''