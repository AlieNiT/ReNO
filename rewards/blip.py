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
        use_item_head: bool = False,
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
        self.use_itm_head = use_item_head
        super().__init__("BLIP", weighting)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def get_text_features(self, prompt: str) -> torch.Tensor:
        return prompt
    
    def process_features(self, features):
        return features

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        text_inputs = self.processor(text=text_features, return_tensors="pt")
        if self.use_itm_head:
            itm_score = self.blip_model(
                pixel_values=image_features, 
                **text_inputs, 
                use_itm_head=True
            )[0]
            target = torch.tensor([1]).to(self.blip_model.device)
            ce_loss = torch.nn.functional.cross_entropy(itm_score, target)
            return ce_loss * 50
        else:
            blip_loss = self.blip_model(
                pixel_values=image_features, 
                **text_inputs, 
                use_itm_head=False
            )[0]
            return 100 * (1 - blip_loss)
