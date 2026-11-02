import unittest
from unittest.mock import patch

import torch
from transformers import BlipConfig, BlipForImageTextRetrieval

from src.data import TEMPLATES
from src.model import BlipExperiment, LoRALinear
from src.train import GROUPS, metrics, negative_texts


class BasicTests(unittest.TestCase):
    def test_lora_starts_as_base_and_freezes_base(self):
        torch.manual_seed(3)
        base = torch.nn.Linear(5, 12)
        x = torch.randn(2, 5)
        expected = base(x).detach().clone()
        layer = LoRALinear(base, rank=2, alpha=4, dropout=0)
        self.assertTrue(torch.equal(layer(x), expected))
        layer(x).sum().backward()
        self.assertIsNone(layer.base.weight.grad)
        self.assertIsNotNone(layer.b.weight.grad)

    def test_metrics(self):
        out = metrics([0, 1, 2], [0, 1, 2])
        self.assertEqual(out["accuracy"], 1)
        self.assertEqual(out["macro_f1"], 1)
        self.assertEqual(len(TEMPLATES), 3)
        self.assertEqual(GROUPS["D"], "caption")

    def test_negative_samples_different_class(self):
        class Fake:
            by_label = {0: [0], 1: [1], 2: [2]}
            def text(self, i):
                return str(i)
        import random
        labels = [0, 1, 2] * 10
        neg = negative_texts(Fake(), labels, random.Random(4))
        self.assertTrue(all(int(n) != label for n, label in zip(neg, labels)))

    def test_tiny_blip_forward_and_freeze(self):
        cfg = BlipConfig(
            vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                           "num_attention_heads": 4, "image_size": 32, "patch_size": 16},
            text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "vocab_size": 100, "is_decoder": True},
            image_text_hidden_size=16)
        raw = BlipForImageTextRetrieval(cfg)
        with patch("src.model.BlipForImageTextRetrieval.from_pretrained",
                   return_value=(raw, {"missing_keys": [], "mismatched_keys": []})):
            model = BlipExperiment("unused", use_text=True, last_layers=1, rank=2, alpha=4)
        self.assertEqual(len(model.targets), 3)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertFalse(any("base." in n for n in names))
        x = torch.randn(2, 3, 32, 32)
        ids = torch.randint(1, 99, (4, 5))
        result = model(x, ids, torch.ones_like(ids), torch.tensor([0, 1, 0, 1]))
        self.assertEqual(tuple(result["class_logits"].shape), (2, 3))
        self.assertEqual(tuple(result["itm_logits"].shape), (4, 2))
        (result["class_logits"].sum() + result["itm_logits"].sum()).backward()
        self.assertIsNotNone(model.blip.vision_model.encoder.layers[-1].self_attn.qkv.b.weight.grad)
        self.assertIsNotNone(model.blip.text_encoder.encoder.layer[-1].crossattention.self.value.b.weight.grad)
        state = model.trainable_state()
        model.load_trainable_state(state)
        self.assertIsNone(model(x)["itm_logits"])


if __name__ == "__main__":
    unittest.main()
