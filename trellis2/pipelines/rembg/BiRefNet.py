from typing import Optional
import hashlib
import inspect
from pathlib import Path
import warnings

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

DEFAULT_BIREFNET_MODEL_NAME = "ZhengPeng7/BiRefNet"
DEFAULT_BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
NON_COMMERCIAL_RMBG_MODEL_NAME = "briaai/RMBG-2.0"
PINNED_REMOTE_CODE_SHA256 = {
    ("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"): {
        "BiRefNet_config.py": "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52",
        "birefnet.py": "208771ae626f653d64128fbf2d6ac9f8e645c5cc5e286258a73ec3322bbfe5ef",
    },
    ("ZhengPeng7/BiRefNet_dynamic", "280306042f57b7a33854319da62fd86aaa89ec4c"): {
        "BiRefNet_config.py": "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52",
        "birefnet.py": "2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831",
    },
    ("ZhengPeng7/BiRefNet_HR", "a7a562f6fd16021180f2f4348f4de003a2d3d1e1"): {
        "BiRefNet_config.py": "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52",
        "birefnet.py": "2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831",
    },
}


class BiRefNet:
    def __init__(
        self,
        model_name: str = DEFAULT_BIREFNET_MODEL_NAME,
        revision: Optional[str] = DEFAULT_BIREFNET_REVISION,
        trust_remote_code: bool = True,
        verify_remote_code_hashes: bool = True,
    ):
        if model_name == DEFAULT_BIREFNET_MODEL_NAME and revision is None:
            raise ValueError(
                "BiRefNet default model must be loaded with a pinned revision for reproducibility."
            )
        self.model_name = model_name
        self.revision = revision
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        if trust_remote_code and verify_remote_code_hashes and revision is not None:
            self._verify_remote_code_hashes()
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

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _verify_remote_code_hashes(self) -> None:
        key = (self.model_name, self.revision)
        expected_hashes = PINNED_REMOTE_CODE_SHA256.get(key)
        if expected_hashes is None:
            warnings.warn(
                f"No remote code hash manifest registered for {self.model_name}@{self.revision}. "
                "Model is pinned by revision but file hashes were not validated.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        source_file = inspect.getsourcefile(self.model.__class__)
        if source_file is None:
            raise RuntimeError(
                f"Unable to locate source file for remote model class {self.model.__class__.__module__}."
            )
        module_dir = Path(source_file).resolve().parent
        for file_name, expected_hash in expected_hashes.items():
            file_path = module_dir / file_name
            if not file_path.exists():
                raise RuntimeError(
                    f"Expected remote code file {file_name} missing in {module_dir} "
                    f"for {self.model_name}@{self.revision}."
                )
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Remote code hash mismatch for {file_name} in {self.model_name}@{self.revision}: "
                    f"expected {expected_hash}, got {actual_hash}."
                )
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        model_dtype = next(self.model.parameters()).dtype
        input_images = self.transform_image(image).unsqueeze(0).to(self.device, dtype=model_dtype)
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size, Image.Resampling.BILINEAR)
        image.putalpha(mask)
        return image
