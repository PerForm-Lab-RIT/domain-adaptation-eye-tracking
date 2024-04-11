import torch
import torch.nn as nn

from .inceptionv4 import Inceptionv4


class SiameseNet2D(torch.nn.Module):
  def __init__(self, input_shape, n_classes=2) -> None:
    """

    Args:
        input_shape (_type_): should be (1, 240, 320)
        feature_channels (int, optional): number of channels of output features. Defaults to 32.
        n_classes (int, optional): number of domain. Defaults to 2.
    """
    super().__init__()

    C, H, W = input_shape

    self.feature_extractor = Inceptionv4(in_channels=C, classes=n_classes)


  def forward_once(self, img):
    return self.feature_extractor(img)

  def forward(self, img1, img2):
    out1 = self.forward_once(img1)
    out2 = self.forward_once(img2)
    return out1, out2

  def save(self, target_path):
    torch.save(self.state_dict(), target_path)
  
  def load(self, target_path):
    self.load_state_dict(torch.load(target_path))



class ContrastiveLoss(nn.Module):

    def __init__(self, margin, device=torch.device("cpu")):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * torch.nn.functional.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
