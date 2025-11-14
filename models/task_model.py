import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch.functional import img_to_tensor
from typing import Optional, Tuple
from models.dpt import DPT_DINOv2

class Task_Network(nn.Module):
  def __init__(
      self,
      checkpoint_path: str,
      num_classes: int,
      encoder: str = "vitl",
      decoder: str = "dpt",
      pretrained_backbone: bool = False,
      norm_mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
      norm_std: Tuple[float, float, float]  = (58.395, 57.12, 57.375),
      input_crop: Optional[Tuple[int, int]] = None,   # e.g., (504, 504) or None
      device: Optional[torch.device] = 'cuda:0'
  ):
      super().__init__()

      # Build the multi-head model but we’ll only use the segmentation head
      head_configs = [
          {"name": "regression", "nclass": 1},         # present in checkpoints
          {"name": "segmentation", "nclass": num_classes},
      ]

      self.model = DPT_DINOv2(encoder=encoder, head_configs=head_configs, pretrained=pretrained_backbone)
      sd = torch.load(checkpoint_path, map_location=device)
      self.model.load_state_dict(sd, strict=True)
      
      # Freeze teacher params
      self.model.eval()
      for p in self.model.parameters():
          p.requires_grad = False

      self.register_buffer("mean", torch.tensor(norm_mean, dtype=torch.float32).view(1, 3, 1, 1))
      self.register_buffer("std",  torch.tensor(norm_std,  dtype=torch.float32).view(1, 3, 1, 1))
      self.input_crop = input_crop
      self._device = device

      if device is not None:
          self.model.to(device)

  def preprocess(self, x_minus1_1: torch.Tensor) -> torch.Tensor:
      """
      x_minus1_1: [N, 3, H, W], in [-1,1] (CycleGAN output)
      Convert -> [0,1] then apply SynRS3D mean/std (expects max_pixel_value=1 in your eval script)
      """
      x01 = (x_minus1_1 + 1.0) / 2.0
      x01_norm = (x01 - self.mean) / self.std
      if self.input_crop is not None:
          th, tw = self.input_crop
          _, _, H, W = x01_norm.shape
          ys = max((H - th) // 2, 0)
          xs = max((W - tw) // 2, 0)
          x01_norm = x01_norm[:, :, ys:ys+th, xs:xs+tw]
      return x01_norm
  def forward(self, x_minus1_1: torch.Tensor) -> torch.Tensor:
      x = self.preprocess(x_minus1_1)
      out = self.model(x)
      return out["segmentation"]











      
