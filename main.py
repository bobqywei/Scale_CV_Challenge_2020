import numpy as np
import os
import torch
import torch.nn as nn
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, padding=1, bias=True, bn=True, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = nn.ReLU() if act else None

    def forward(self, x):
        out = self.conv(x)
        if not self.bn is None:
            out = self.bn(out)
        if not self.act is None:
            out = self.act(out)
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.down = nn.Sequential(
            Conv(1, 8),
            Conv(8, 16, stride=2),
            Conv(16, 32, stride=2),
            Conv(32, 64, stride=2),
            Conv(64, 96, stride=2),
            Conv(96, 96, stride=2),
            Conv(96, 96, stride=2)
        )
        self.linear = nn.Sequential(
            nn.Linear(4*4*96, 128),
            nn.Linear(128, 3))
        self.act = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        out = self.down(x).reshape(batch_size, -1)
        out = self.linear(out)
        return self.act(out)


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img

def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )

def find_circle(model, img, use_gpu=False):
    img = torch.from_numpy(img).to(torch.float)
    if use_gpu:
        img = img.cuda()
    detection = model(img.unsqueeze(0).unsqueeze(0))
    return int(detection[0,0].item() * 199), int(detection[0,1].item() * 199), int(detection[0,2].item() * 39 + 10)


def main():
    use_gpu = torch.cuda.is_available()
    results = []
    
    model = Model()
    model.load_state_dict(torch.load(os.path.join(os.curdir, "checkpoints/ckpt_312500"), map_location=torch.device('cuda:0' if use_gpu else 'cpu'))['state_dict'])
    if use_gpu:
        model = model.cuda()
    print("Model Parameters: {}".format(count_parameters(model)))
    model.eval()

    print("Start Evaluation")
    for i in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(model, img, use_gpu=use_gpu)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())


if __name__ == "__main__":
    main()
