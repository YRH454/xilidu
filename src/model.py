"""BLIP retrieval with explicit, narrow low-rank adapters."""
import math
import torch
from torch import nn
from transformers import BlipForImageTextRetrieval


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)}")
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.b(self.a(self.dropout(x))) * self.scaling


class BlipExperiment(nn.Module):
    def __init__(self, checkpoint, use_text, rank=8, alpha=16, dropout=0.05, last_layers=4):
        super().__init__()
        self.blip, info = BlipForImageTextRetrieval.from_pretrained(
            checkpoint, local_files_only=True, use_safetensors=True, output_loading_info=True)
        critical = ("vision_model.", "text_encoder.", "itm_head.")
        missing = [x for x in info["missing_keys"] if x.startswith(critical)]
        if missing or info.get("mismatched_keys"):
            raise RuntimeError(f"Incomplete BLIP checkpoint: {missing}, {info.get('mismatched_keys')}")
        self.blip.requires_grad_(False)
        vision_layers = self.blip.vision_model.encoder.layers
        text_layers = self.blip.text_encoder.encoder.layer
        if len(vision_layers) < last_layers or len(text_layers) < last_layers:
            raise ValueError("Fewer model layers than requested LoRA layers")
        self.targets = []
        for i in range(len(vision_layers) - last_layers, len(vision_layers)):
            attention = vision_layers[i].self_attn
            attention.qkv = LoRALinear(attention.qkv, rank, alpha, dropout)
            self.targets.append(f"vision_model.encoder.layers.{i}.self_attn.qkv")
        # Create the classifier before optional text adapters: A-D start identically.
        self.classifier = nn.Linear(self.blip.config.vision_config.hidden_size, 3)
        self.use_text = use_text
        if use_text:
            for i in range(len(text_layers) - last_layers, len(text_layers)):
                attention = text_layers[i].crossattention.self
                for part in ("query", "value"):
                    setattr(attention, part, LoRALinear(getattr(attention, part), rank, alpha, dropout))
                    self.targets.append(f"text_encoder.encoder.layer.{i}.crossattention.self.{part}")
            self.blip.itm_head.requires_grad_(True)
        if len(self.targets) != last_layers * (3 if use_text else 1):
            raise RuntimeError("Unexpected number of LoRA target layers")

    def forward(self, pixel_values, input_ids=None, attention_mask=None, text_image_indices=None):
        image = self.blip.vision_model(pixel_values=pixel_values).last_hidden_state
        result = {"class_logits": self.classifier(image[:, 0, :]), "itm_logits": None}
        if input_ids is not None:
            if not self.use_text or text_image_indices is None:
                raise ValueError("Text branch disabled or image indices missing")
            text_images = image.index_select(0, text_image_indices)
            image_mask = torch.ones(text_images.shape[:2], dtype=torch.long, device=image.device)
            fused = self.blip.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                           encoder_hidden_states=text_images,
                                           encoder_attention_mask=image_mask,
                                           return_dict=True).last_hidden_state
            result["itm_logits"] = self.blip.itm_head(fused[:, 0, :])
        return result

    def trainable_state(self):
        return {name: param.detach().cpu().clone() for name, param in self.named_parameters()
                if param.requires_grad}

    def load_trainable_state(self, state):
        params = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if set(params) != set(state):
            raise ValueError("Checkpoint trainable parameters do not match this model")
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))
