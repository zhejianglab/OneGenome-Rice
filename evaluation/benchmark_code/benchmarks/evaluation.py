import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import wandb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr, spearmanr
import numpy as np
from xgboost import XGBClassifier
torch.use_deterministic_algorithms(True)

def resume_metrics(y_pred, y_score, y_test, num_classes, dataset, layer, classifer_type, config):
    y_true = y_test.numpy()

    # y_score = np.asarray(y_score)
    # # === 关键修复：处理多 layer / 多模型 ===
    # if y_score.ndim == 3:
    #     # 多层输出 → soft ensemble
    #     y_score = y_score.mean(axis=0)
    # elif y_score.ndim != 2:
    #     raise ValueError(f"Unexpected y_score shape: {y_score.shape}")


    if config.dataset_info[dataset]["eval_task"] == "classification" or config.dataset_info[dataset]["eval_task"] == "labels":
        # task = f"{dataset_name}-{layer}layer"
        acc = accuracy_score(y_true, y_pred)

        # Handle ROC AUC calculation for different classification scenarios
        if num_classes == 2 and (
            config.dataset_info[dataset]["eval_task"] == "classification"
            or classifer_type == "RF"
        ):
            # Binary classification: use probability of positive class
            auc = roc_auc_score(y_true, y_score[:, 1])
        else:
            # Multi-class classification: use macro average
            auc = roc_auc_score(
                y_true, y_score, multi_class='ovr', average='macro')
        # 根据类别数选择评估方式
        if num_classes == 2 and config.dataset_info[dataset]["eval_task"] == "classification":
            # 二分类使用默认的binary设置
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            # 多分类使用macro平均
            precision = precision_score(
                y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(
                y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if config.dataset_info[dataset]["eval_task"] == "labels":
            mcc_scores = []
            for i in range(y_true.shape[1]):  # 遍历每个标签
                mcc_scores.append(matthews_corrcoef(
                    y_true[:, i], y_pred[:, i]))
            mcc = np.mean(mcc_scores)  # 使用所有标签的MCC平均值
        elif config.dataset_info[dataset]["eval_task"] == "classification":
            mcc = matthews_corrcoef(y_true, y_pred)
        print(
            f"[INFO] {dataset}-{layer}layer: {acc:.4f} {auc:.4f} {precision:.4f} {recall:.4f} {f1:.4f} {mcc:.4f}")
        return {
            "task": dataset,
            "classifer": classifer_type,
            "layer": layer,
            "accuracy": acc,
            "roc_auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc
        }
    elif config.dataset_info[dataset]["eval_task"] == "regression":
        if num_classes == 1:
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()

            mse = mean_squared_error(y_true_flat, y_pred_flat)
            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            r2 = r2_score(y_true_flat, y_pred_flat)
            pearson_corr = pearsonr(y_true_flat, y_pred_flat)[0]
            spearman_corr = spearmanr(y_true_flat, y_pred_flat)[0]
        else:
            # Multi-target regression - use macro averaging
            mse = mean_squared_error(
                y_true, y_pred, multioutput='uniform_average')
            mae = mean_absolute_error(
                y_true, y_pred, multioutput='uniform_average')
            r2 = r2_score(y_true, y_pred, multioutput='uniform_average')

            # Calculate correlations per target and average
            pearson_corrs = [pearsonr(y_true[:, i], y_pred[:, i])[
                0] for i in range(y_true.shape[1])]
            spearman_corrs = [spearmanr(y_true[:, i], y_pred[:, i])[
                0] for i in range(y_true.shape[1])]
            pearson_corr = sum(pearson_corrs) / len(pearson_corrs)
            spearman_corr = sum(spearman_corrs) / len(spearman_corrs)
            print(
                f"[INFO] {dataset}-{layer}layer: {mse:.4f} {mae:.4f} {r2:.4f} {pearson_corr:.4f} {spearman_corr:.4f}")
        return {
            "task": dataset,
            "classifer": classifer_type,
            "layer": layer,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "pearsonr": pearson_corr,
            "spearmanr": spearman_corr
        }


def catter(data, cat_dim):
    if len(data.shape) == 2:
        return data
    else:
        if cat_dim == 1:
            return torch.reshape(data, (data.shape[0], -1))
        else:
            return data.mean(dim=1)


def results_exist(result_path, layer_to_eval, num_hidden_layers, eval_task) -> list:
    if layer_to_eval is None:
        layer_to_eval = list(range(num_hidden_layers+1))
    else:
        for layer in layer_to_eval:
            assert layer <= num_hidden_layers, f"[ERROR] layer_to_eval is out of range: {layer} > {num_hidden_layers}"
    if os.path.exists(result_path):
        if eval_task == "classification" or eval_task == "labels":
            eval_items = ['task', 'layer', 'accuracy', 'roc_auc',
                          'precision', 'recall', 'f1', 'mcc']
        elif eval_task == "regression":
            eval_items = ['task', 'layer', 'mse', 'mae', 'r2',
                          'pearsonr', 'spearmanr']
        result = pd.read_csv(result_path, sep="\t",
                             index_col=False).to_dict(orient="list")
        layer_need_eval = []
        for item in eval_items:
            # 查看评测项目是否齐全
            if item not in result.keys():
                print(f"[eval] 📊 结果文件中缺少项目 {item}")
                return layer_to_eval
        # 查看是否每一层都有完整的评测结果
        for layer in layer_to_eval:
            if layer not in result["layer"]:
                layer_need_eval.append(layer)
        return layer_need_eval
    else:
        return layer_to_eval


def number_class_count(y_train, y_test, y_eval, dataset_name, config):
    if config.dataset_info[dataset_name]["eval_task"] == "classification":
        assert torch.unique(y_train).numel() == torch.unique(y_test).numel(
        ), f"[ERROR] y_train and y_test have different number of unique elements: {torch.unique(y_train).numel()} and {torch.unique(y_test).numel()}"
        assert torch.unique(y_train).numel() == torch.unique(y_eval).numel(
        ), f"[ERROR] y_train and y_eval have different number of unique elements: {torch.unique(y_train).numel()} and {torch.unique(y_eval).numel()}"
        return torch.unique(y_train).numel()
    elif config.dataset_info[dataset_name]["eval_task"] == "regression":
        assert len(y_train[0]) == len(y_test[0]) == len(
            y_eval[0]), f"[ERROR] y_train and y_test and y_eval have different number of elements: {len(y_train[0])} and {len(y_test[0])} and {len(y_eval[0])}"
        return len(y_train[0])
    elif config.dataset_info[dataset_name]["eval_task"] == "labels":
        assert len(y_train[0]) == len(y_test[0]) == len(
            y_eval[0]), f"[ERROR] y_train and y_test and y_eval have different number of elements: {len(y_train[0])} and {len(y_test[0])} and {len(y_eval[0])}"
        return len(y_train[0])

# -------------------------------
# XGBoost 分类器训练与预测函数
# -------------------------------


def train_xgboost_classifier(X_train, y_train, X_test, config, random_state=42):
    X_train_np = catter(
        X_train, config["pooled_embeddings_cat_dim"]).cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_test_np = catter(
        X_test, config["pooled_embeddings_cat_dim"]).cpu().numpy()

    """训练XGBoost分类器"""

    # print("Training XGBoost classifier...")
    # print("XGboost parameters: n_estimators=100, learning_rate=0.1, max_depth=6, random_state={}".format(random_state))
    xgb = XGBClassifier(
        n_estimators=config["xgb_n_estimators"],
        random_state=random_state,
        eval_metric=config["xgb_eval_metric"],
        learning_rate=config["xgb_learning_rate"],
        max_depth=config["xgb_max_depth"]
    )
    xgb.fit(X_train_np, y_train_np)
    print("XGBoost training completed")

    probs = xgb.predict_proba(X_test_np)
    preds = xgb.predict(X_test_np)
    return preds, probs
# -------------------------------
# 随机森林训练与预测函数
# -------------------------------


def train_rf_classifier(X_train, y_train, X_val, y_val, device, config, n_estimators=100, random_state=42):
    X_train_np = catter(
        X_train, config["pooled_embeddings_cat_dim"]).cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_val_np = catter(X_val, config["pooled_embeddings_cat_dim"]).cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    # print("train_rf_classifier")
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_np, y_train_np)
    # print("train_rf_classifier done")

    # 获取所有类别的概率
    probs = rf.predict_proba(X_val_np)
    # 直接使用predict获取预测类别
    preds = rf.predict(X_val_np)

    return preds, probs

# -------------------------------
# MLP 分类器
# -------------------------------


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()
        hidden_dim1 = input_dim//2
        hidden_dim2 = input_dim//4
        hidden_dim3 = input_dim//8
        if hidden_dim3 > 128:
            hidden_dim3 = 128

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 修改输出层：输出维度=类别数，使用线性激活
            nn.Linear(hidden_dim3, num_classes)
            # 注意：这里移除了Sigmoid，将在损失函数中处理
        )

    def forward(self, x):
        return self.layers(x)


# -------------------------------
# MLP 训练函数
# -------------------------------


def train_mlp_classifier(X_train, y_train, X_test, y_test, X_eval, y_eval, device, num_classes, gpu_id, config, dataset_name, layer,
                         dropout=0.2, batch_size=64,
                         lr=1e-3, epochs=50, random_state=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.manual_seed(random_state)
    # 标签转换为长整型 (CrossEntropyLoss要求)
    X_train = catter(X_train.to(
        device), config["pooled_embeddings_cat_dim"])
    X_test = catter(
        X_test.to(device), config["pooled_embeddings_cat_dim"])
    X_eval = catter(
        X_eval.to(device), config["pooled_embeddings_cat_dim"])

    X_train = X_train.float()
    X_test = X_test.float()
    X_eval = X_eval.float()

    if config.dataset_info[dataset_name]["eval_task"] == "classification":
        y_train = y_train.to(device).long()
        y_test = y_test.to(device).long()
        y_eval = y_eval.to(device).long()
    elif config.dataset_info[dataset_name]["eval_task"] == "regression":
        y_train = y_train.to(device)
        y_test = y_test.to(device)
        y_eval = y_eval.to(device)
    elif config.dataset_info[dataset_name]["eval_task"] == "labels":
        y_train = y_train.to(device).float()
        y_test = y_test.to(device).float()
        y_eval = y_eval.to(device).float()
    else:
        raise ValueError(
            f'Invalid eval task: {config.dataset_info[dataset_name]["eval_task"]}')

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # 添加num_classes参数
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim, num_classes,
                          dropout).to(device)
    if config['wandb_report']:
        wandb.watch(model)

    # 修改损失函数为CrossEntropyLoss（包含Softmax）
    if config.dataset_info[dataset_name]["eval_task"] == "classification":
        criterion = nn.CrossEntropyLoss()
    elif config.dataset_info[dataset_name]["eval_task"] == "regression":
        criterion = nn.HuberLoss(reduction='mean')
    elif config.dataset_info[dataset_name]["eval_task"] == "labels":
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=float(lr))

    best_val_acc = 0.0
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)  # 自动处理Softmax
            loss.backward()
            optimizer.step()

            # 计算准确率（使用argmax获取预测类别）
            if config.dataset_info[dataset_name]["eval_task"] == "classification":
                _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别索引
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            elif config.dataset_info[dataset_name]["eval_task"] == "regression":
                pass
            elif config.dataset_info[dataset_name]["eval_task"] == "labels":
                predicted = (outputs > 0.5).float()
                total += batch_y.numel()
                correct += (predicted == batch_y).sum().item()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_acc = None if config.dataset_info[dataset_name][
            "eval_task"] == "regression" else 100 * correct / total

        # 验证集评估
        model.eval()
        with torch.no_grad():
            eval_outputs = model(X_eval)
            eval_loss = criterion(eval_outputs, y_eval).item()
            if config.dataset_info[dataset_name]["eval_task"] == "classification":
                _, eval_predicted = torch.max(eval_outputs, 1)
                eval_correct = (eval_predicted == y_eval).sum().item()
                eval_acc = 100 * eval_correct / y_eval.size(0)
            elif config.dataset_info[dataset_name]["eval_task"] == "regression":
                eval_acc = None
            elif config.dataset_info[dataset_name]["eval_task"] == "labels":
                eval_predicted = (torch.sigmoid(eval_outputs) > 0.5).float()
                eval_correct = (eval_predicted == y_eval).sum().item()
                eval_acc = 100 * eval_correct / y_eval.numel()

        # 日志记录
        log_data = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_loss': eval_loss
        }
        if config.dataset_info[dataset_name]["eval_task"] == "classification" or config.dataset_info[dataset_name]["eval_task"] == "labels":
            log_data.update({
                'train_accuracy': train_acc,
                'val_accuracy': eval_acc
            })
        if config['wandb_report']:
            wandb.log(log_data)

    # 最终预测
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        if config.dataset_info[dataset_name]["eval_task"] == "classification":
            probs = torch.softmax(test_outputs, dim=1).cpu().numpy()  # 获取各类别概率
            preds = torch.argmax(test_outputs, dim=1).cpu().numpy()    # 获取预测类别
        elif config.dataset_info[dataset_name]["eval_task"] == "regression":
            probs = None
            preds = test_outputs.cpu().numpy()
        elif config.dataset_info[dataset_name]["eval_task"] == "labels":
            probs = torch.sigmoid(test_outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
    if config['save_last_epoch_model']:
        os.makedirs(
            f"{config['eval_result_path']}/{config['model_name']}/last_epoch_model", exist_ok=True)
        torch.save(model.state_dict(
        ), f"{config['eval_result_path']}/{config['model_name']}/last_epoch_model/{dataset_name}-{layer}layer-{epochs}epoch.pt")

    return preds, probs


def evaluate_model_on_dataset_layer(dataset_name, gpu_id, model_name, layer, config):
    if config['wandb_report']:
        wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], name=f"{model_name}_{dataset_name}_multi_layer", reinit=True, config={
            "model_type": "MLP", "task": dataset_name, "model_name": model_name, "model_layers": layer})
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dir = f"{config['embedding_output_dir']}/{config['model_name']}"

    data_train_path = f"{embedding_dir}/{dataset_name}-{layer}layer_train.pt"
    data_eval_path = f"{embedding_dir}/{dataset_name}-{layer}layer_eval.pt"
    data_test_path = f"{embedding_dir}/{dataset_name}-{layer}layer_test.pt"

    # 检查是否存在embedding文件，如果没有测试集或数据集其中之一，则测试集和数据集相同
    assert os.path.exists(
        data_train_path), f"[ERROR] data not found: {data_train_path}"
    assert os.path.exists(data_test_path) or os.path.exists(
        data_eval_path), f"{data_test_path} or {data_eval_path} must exist one"
    if not os.path.exists(data_test_path):
        data_test_path = data_eval_path
    elif not os.path.exists(data_eval_path):
        data_eval_path = data_test_path

    X_train = torch.load(data_train_path)["embeddings"]
    y_train = torch.load(data_train_path)["labels"]
    X_eval = torch.load(data_eval_path)["embeddings"]
    y_eval = torch.load(data_eval_path)["labels"]
    X_test = torch.load(data_test_path)["embeddings"]
    y_test = torch.load(data_test_path)["labels"]

    num_classes = number_class_count(
        y_train, y_test, y_eval, dataset_name, config)

    if "classifer_type" in config.dataset_info[dataset_name]:
        classifer_type = config.dataset_info[dataset_name]["classifer_type"]
    else:
        classifer_type = config["classifer_type"]
    if classifer_type == "MLP":
        y_pred, y_score = train_mlp_classifier(
            X_train, y_train, X_test, y_test, X_eval, y_eval, device, num_classes, gpu_id, config, dataset_name, layer, epochs=config['mlp_epochs'], dropout=config['mlp_dropout'], lr=config['mlp_lr'], random_state=config['seed'])
    elif classifer_type == "RF":
        y_pred, y_score = train_rf_classifier(
            X_train, y_train, X_eval, y_eval, device, config, n_estimators=config['rf_n_estimators'], random_state=config['seed'])
    elif classifer_type == "XGB":
        y_pred, y_score = train_xgboost_classifier(
            X_train, y_train, X_test, config, random_state=config['seed'])

    results = [resume_metrics(y_pred, y_score, y_test,
                              num_classes, dataset_name, layer, classifer_type, config)]
    return results

