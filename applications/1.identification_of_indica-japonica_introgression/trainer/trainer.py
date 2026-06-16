import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from utils.trainer_utils import SupConLoss, compute_metrics

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, multi_label=False, lambda_contrastive=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_label = multi_label
        self.lambda_contrastive = lambda_contrastive
        
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs
    ):
        # transformers>=4.50 可能会额外传入 num_items_in_batch / 其他 kwargs
        del num_items_in_batch, kwargs
        device = next(model.parameters()).device
        if "labels" not in inputs:
            raise KeyError(f"'labels' not found in batch inputs. Available keys: {list(inputs.keys())}")
        labels = inputs["labels"].to(device)
        
        # 分支处理输入
        if "embeddings" in inputs:
            # 模式2: 直接使用预计算嵌入
            embeddings = inputs["embeddings"].to(device)
            z, logits = model(embeddings)
        else:
            # 模式1: 原始token输入
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            z, logits = model(input_ids, attention_mask)

        # 计算损失
        if self.multi_label:
            labels_float = labels.float()
            logits_float = logits.float()
            loss_cls = nn.BCEWithLogitsLoss()(logits_float, labels_float)
        else:
            if labels.ndim == 2:
                labels_cls = labels.argmax(dim=1)
            else:
                labels_cls = labels.long()
            loss_cls = F.cross_entropy(logits, labels_cls)

        supcon = SupConLoss()
        loss_con = supcon(z, labels)

        loss = (1 - self.lambda_contrastive) * loss_cls + self.lambda_contrastive * loss_con
        return (loss, {"logits": logits.float()}) if return_outputs else loss
