from typing import Optional

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

DEFAULT_BIREFNET_MODEL_NAME = "ZhengPeng7/BiRefNet"
DEFAULT_BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
NON_COMMERCIAL_RMBG_MODEL_NAME = "briaai/RMBG-2.0"


class BiRefNet:
    def __init__(
        self,
        model_name: str = DEFAULT_BIREFNET_MODEL_NAME,
        revision: Optional[str] = DEFAULT_BIREFNET_REVISION,
        trust_remote_code: bool = True,
    ):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self._device = torch.device("cpu")
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self._device = torch.device(device)
        self.model.to(self._device)
        if self._device.type == "cpu":
            # Some published weights are fp16 by default; CPU inference expects fp32.
            self.model.float()
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        model_dtype = next(self.model.parameters()).dtype
        input_images = self.transform_image(image).unsqueeze(0).to(self._device, dtype=model_dtype)
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size, Image.Resampling.BILINEAR)
        image.putalpha(mask)
        return image
