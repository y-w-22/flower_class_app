#必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
#学習時に使用したものと同じ学習済モデルをインポート
from torchvision.models import resnet18

#学習済みモデルに合わせた前処理を追加
transform=transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

        
class Net(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.feature = resnet18(pretrained=True)
    self.fc = nn.Linear(1000, 10)

  def forward(self,x):
    h = self.feature(x)
    h = self.fc(h)
    return h