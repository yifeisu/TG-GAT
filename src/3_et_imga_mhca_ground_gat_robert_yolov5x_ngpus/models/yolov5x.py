import torch
from models.common import *


class yolov5x_backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('0', Conv(3, 80, 6, 2, 2))
        self.add_module('1', Conv(80, 160, 3, 2))
        self.add_module('2', C3(160, 160, 4))
        self.add_module('3', Conv(160, 320, 3, 2))
        self.add_module('4', C3(320, 320, 8))
        self.add_module('5', Conv(320, 640, 3, 2))
        self.add_module('6', C3(640, 640, 12))
        self.add_module('7', Conv(640, 1280, 3, 2))
        self.add_module('8', C3(1280, 1280, 4))
        self.add_module('9', SPPF(1280, 1280, 5))

    def forward(self, x):
        x = getattr(self, '0')(x)
        x = getattr(self, '1')(x)
        x = getattr(self, '2')(x)
        x = getattr(self, '3')(x)
        x = getattr(self, '4')(x)
        x = getattr(self, '5')(x)
        x = getattr(self, '6')(x)
        x = getattr(self, '7')(x)
        x = getattr(self, '8')(x)
        x = getattr(self, '9')(x)

        return x


if __name__ == '__main__':
    inp = torch.rand([4, 3, 224, 224])
    backbone = yolov5x_backbone()

    model = torch.load('last.pt')
    new_state = model['model'].model.state_dict()
    filter_state_dict = {}
    for k, v in new_state.items():
        if int(k.split('.')[0]) <= 9:
            filter_state_dict[k] = v
    new_state = filter_state_dict

    msg = backbone.load_state_dict(new_state, strict=False)
    print(msg)

    out = backbone(inp)
