import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


# ===========================
# 1. Pooling Layer (可插拔)
# ===========================
class PoolingLayer(nn.Module):
    """Pooling 策略模块，支持多种池化方式"""
    def __init__(self, strategy="masked_mean"):
        super().__init__()
        self.strategy = strategy
        assert strategy in ["masked_mean", "cls_token", "attention_weighted"], \
            f"Unknown pooling strategy: {strategy}"
    
    def forward(self, hidden, attention_mask):
        """
        Args:
            hidden: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        if self.strategy == "masked_mean":
            # Masked mean pooling: ignore paddings
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden).float()
            sum_hidden = (hidden * attention_mask_expanded).sum(dim=1)
            sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask
        
        elif self.strategy == "cls_token":
            # Use [CLS] token (first token)
            return hidden[:, 0, :]
        
        elif self.strategy == "attention_weighted":
            # Attention-weighted pooling
            attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            attention_weights = attention_mask.float() / attention_mask.float().sum(dim=1, keepdim=True)
            return (hidden * attention_weights).sum(dim=1)


# ===========================
# 2. Backbone Module (可插拔)
# ===========================
class BackboneModule(nn.Module):
    """Backbone 模块，支持 LoRA 微调"""
    def __init__(self, model_path, token, lora_config=None):
        super().__init__()
        self.lora_enabled = lora_config is not None
        
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=token
        )
        
        if lora_config:
            peft_conf = LoraConfig(
                r=lora_config["r"],
                lora_alpha=lora_config["alpha"],
                lora_dropout=lora_config["dropout"],
                target_modules=lora_config["target_modules"],
                task_type="FEATURE_EXTRACTION"
            )
            self.model = get_peft_model(self.model, peft_conf)
            self.model.print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask):
        """
        Returns:
            hidden: [batch_size, seq_len, hidden_dim]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


# ===========================
# 3. Projection Head (原有)
# ===========================
class ProjectionHead(nn.Module):
    """投影头，用于对比学习"""
    def __init__(self, dims, is_bn=False):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if is_bn:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ===========================
# 4. Classification Head (可插拔)
# ===========================
class ClassificationHead(nn.Module):
    """分类头"""
    def __init__(self, input_dim, num_labels, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_labels)
    def forward(self, x):
        return self.fc(self.dropout(x))
    

# ===========================
# 5. Full Model (组装器)
# ===========================
class FullModel(nn.Module):
    """完整模型：组装 backbone、pooling、projection、classifier"""
    def __init__(
        self,
        model_path,
        proj_dims,
        num_labels,
        token,
        lora_config=None,
        pooling_strategy="masked_mean",
        dropout=0.1,
        is_bn=False
    ):
        super().__init__()
        
        # 各个模块
        self.backbone = BackboneModule(model_path, token, lora_config)
        self.pooling = PoolingLayer(strategy=pooling_strategy)
        self.proj = ProjectionHead(proj_dims, is_bn=is_bn)
        self.cls = ClassificationHead(proj_dims[-1], num_labels, dropout=dropout)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Returns:
            z: 对比学习的嵌入 [batch_size, proj_dims[-1]]
            logits: 分类器输出 [batch_size, num_labels]
        """
        del labels, kwargs
        hidden = self.backbone(input_ids, attention_mask)
        pooled = self.pooling(hidden, attention_mask)
        z = self.proj(pooled)
        logits = self.cls(z)
        return z, logits


# ===========================
# 6. Ablation Models (实验用)
# ===========================
class FeatureExtractor(nn.Module):
    """移除分类头，仅输出对比学习嵌入"""
    def __init__(
        self,
        model_path,
        proj_dims,
        token,
        lora_config=None,
        pooling_strategy="masked_mean"
    ):
        super().__init__()
        self.backbone = BackboneModule(model_path, token, lora_config)
        self.pooling = PoolingLayer(strategy=pooling_strategy)
        self.proj = ProjectionHead(proj_dims)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask)
        pooled = self.pooling(hidden, attention_mask)
        z = self.proj(pooled)
        return z


class SimpleClassifier(nn.Module):
    """移除投影头，直接分类"""
    def __init__(
        self,
        model_path,
        num_labels,
        token,
        lora_config=None,
        pooling_strategy="masked_mean",
        dropout=0.1
    ):
        super().__init__()
        self.backbone = BackboneModule(model_path, token, lora_config)
        self.pooling = PoolingLayer(strategy=pooling_strategy)
        self.cls = ClassificationHead(1024, num_labels, dropout=dropout)  # 1024 is backbone hidden dim

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask)
        pooled = self.pooling(hidden, attention_mask)
        logits = self.cls(pooled)
        return logits


# ===========================
# 7. Embedding Model (新)
# ===========================
class EmbeddingModel(nn.Module):
    """仅使用预计算嵌入的模型：Projection + Classifier"""
    def __init__(self, proj_dims, num_labels, is_bn=False, dropout=0.1):
        super().__init__()
        self.proj = ProjectionHead(proj_dims, is_bn=is_bn)
        self.cls = ClassificationHead(proj_dims[-1], num_labels, dropout=dropout)

    def forward(self, embeddings, labels=None, **kwargs):
        """
        Args:
            embeddings: [batch_size, hidden_dim] - 预计算的嵌入
        Returns:
            z: 对比学习的嵌入 [batch_size, proj_dims[-1]]
            logits: 分类器输出 [batch_size, num_labels]
        """
        del labels, kwargs
        z = self.proj(embeddings)
        logits = self.cls(z)
        return z, logits
