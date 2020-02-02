import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models


class HED(nn.Module):
    def __init__(self, crit, pretrained=False):
        super(HED, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=pretrained)
        self.crit = crit

        self.conv1 = self.extract_layer(vgg16, 1)
        self.conv2 = self.extract_layer(vgg16, 2)
        self.conv3 = self.extract_layer(vgg16, 3)
        self.conv4 = self.extract_layer(vgg16, 4)
        self.conv5 = self.extract_layer(vgg16, 5)

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

        self.other_layers = [self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5]

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.normal_(0, 0.01)

        for layer in self.other_layers:
            layer.apply(weights_init)

        self.fuse.weight.data.fill_(0.2)
        self.fuse.bias.data.fill_(0)

    def forward(self, x, y=None):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h, w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h, w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h, w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h, w))

        # dsn fusion output
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        if y:
            return self.crit([d1, d2, d3, d4, d5, fuse], y)
        else:
            return d1, d2, d3, d4, d5, fuse

    def extract_layer(self, model, ind):
        index_dict = {
            1: (0, 6),
            2: (6, 13),
            3: (13, 23),
            4: (23, 33),
            5: (33, 43)}
        start, end = index_dict[ind]
        modified_model = nn.Sequential(*list(model.features.children())[start:end])
        return modified_model