import torch
from transformers import BlipProcessor, BlipForImageTextRetrieval

from rewards.base_reward import BaseRewardLoss

class BLIPLoss(BaseRewardLoss):
    """BLIP reward loss function for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-itm-base-coco",
            cache_dir=cache_dir
        )
        self.blip_model = BlipForImageTextRetrieval.from_pretrained(
            "Salesforce/blip-itm-base-coco",
            cache_dir=cache_dir
        )
        if memsave:
            import memsave_torch.nn
            self.blip_model = memsave_torch.nn.convert_to_memory_saving(self.blip_model)

        self.blip_model = self.blip_model.to(device, dtype=dtype)
        self.blip_model.eval()
        self.freeze_parameters(self.blip_model.parameters())
        super().__init__("BLIP", weighting)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        img_features = self.blip_model.get_image_features(image)
        return img_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        prompt_token = self.processor(
            text=[prompt], padding=True, return_tensors="pt"
        ).to("cuda")
        text_features = self.blip_model.get_text_features(**prompt_token)
        return text_features

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        blip_loss = (
            100
            - (image_features @ text_features.T).mean()
            * self.blip_model.logit_scale.exp()
        )
        return blip_loss
