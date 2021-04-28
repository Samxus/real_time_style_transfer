# @Time : 2021-03-29 20:30
# @Author : Xuanhan Liu
# @Site : 
# @File : main.py
# @Software: PyCharm

from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn as nn
from config import config
from torch.utils.data import DataLoader
import transformer_net
import PackedVgg
import utils
import torchnet as tnt
import tqdm


def train():
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')

    trans = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])

    dataset = torchvision.datasets.ImageFolder(config.data_root, transform=trans)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    transformer = transformer_net.TransformerNet()
    if config.model_path:
        transformer.load_state_dict(torch.load(config.model_path))
    transformer.to(device)

    vgg = PackedVgg.Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(transformer.parameters(), config.lr)

    style = utils.get_style_data(config.style_path)
    style = style.to(device)

    with torch.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]

    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(config.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = config.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)
            style_loss = 0.

            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= config.style_weight

            total_loss = style_loss + content_loss
            total_loss.backward()
            optimizer.step()

            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

        torch.save(transformer.state_dict(), 'checkpoints/moc_style.pth')


@torch.no_grad()
def stylize(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(config, k_, v_)
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')

    # input image preprocess
    content_image = torchvision.datasets.folder.default_loader(config.content_path)
    content_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # model setup
    style_model = transformer_net.TransformerNet()
    style_model.load_state_dict(torch.load(config.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    # style transfer and save output
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    torchvision.utils.save_image(((output_data / 255)).clamp(min=0, max=1), config.result_path)


if __name__ == '__main__':
    train()
