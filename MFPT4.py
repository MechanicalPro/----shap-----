import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import shap
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体（可以根据需要更换）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 1. 数据加载器
class MFPTSegmentLoader(Dataset):
    def __init__(self, root_dir, important_features=None):
        self.root_dir = root_dir
        self.important_features = important_features
        self.label_map = {
            "Three Baseline Conditions.csv": 0,
            "ThreeOuterRaceFaultConditions.csv": 1,
            "SevenMoreOuterRaceFaultConditions.csv": 2,
            "SevenMoreInnerRaceFaultConditions.csv": 3
        }
        self.features = None
        self.labels = None
        self.scaler = StandardScaler()
        self.sample_metadata = []  # 新增：记录每个样本的元数据（文件名，行号）
        self._load_data()
        self._print_class_distribution()

    def _load_data(self):
        print(f"加载数据路径: {self.root_dir}")
        if not os.path.isdir(self.root_dir):
            print(f"错误：{self.root_dir} 不是一个有效的目录。")
            return

        all_data = []
        all_labels = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".csv") and filename in self.label_map:
                file_path = os.path.join(self.root_dir, filename)
                try:
                    label = self.label_map[filename]
                    data = pd.read_csv(file_path, header=None).values[:, :480]
                    if self.important_features is not None:
                        data = data[:, self.important_features]
                    # 记录每个样本的元数据（文件名，行号）
                    metadata = [(filename, idx) for idx in range(len(data))]
                    all_data.append(data)
                    all_labels.append(np.full(len(data), label))
                    self.sample_metadata.extend(metadata)
                    print(f"从 {filename} 加载了 {len(data)} 个样本")
                except Exception as e:
                    print(f"读取文件 {file_path} 时出现错误: {e}")

        if all_data:
            self.features = np.vstack(all_data)
            self.labels = np.hstack(all_labels)
            self.features = self.scaler.fit_transform(self.features)
        else:
            print("错误：未加载任何有效数据。")

    def _print_class_distribution(self):
        class_counts = np.bincount(self.labels, minlength=4)
        class_names = ["正常状态", "外圈故障", "变载荷外圈故障", "变载荷内圈故障"]
        print("\n数据集类别分布：")
        for cls, count in enumerate(class_counts):
            print(f"  {class_names[cls]}: {count} 个样本")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.sample_metadata[idx]  # 新增：返回元数据


# 2. CNN模型定义
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        fc_input_size = input_size // 8
        self.fc_input_size = 64 * fc_input_size
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(-1, self.fc_input_size)
        x = self.fc1_drop(self.fc1_relu(self.fc1(x)))
        x = self.fc2(x)
        return x

##第二种Resnet模型
class Residual(nn.Module):  
    def __init__(self, input_channels, num_channels, use_1x1conv=False, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.dropout = nn.Dropout(p=0.2)  # 增加 Dropout 概率

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        X = self.dropout(X)
        Y = Y+X
        return F.relu(Y)
    

# ...existing code...

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, downsample=True))
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)

class Resnet14(nn.Module):
    def __init__(self, input_size=450, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet_block(32, 32, 2, first_block=True)
        self.layer2 = resnet_block(32, 64, 2)
        self.layer3 = resnet_block(64, 128, 2)
        # 池化后自动计算全连接输入
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)  # 增加 Dropout 概率
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Resnet10(nn.Module):
    def __init__(self, input_size=450, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet_block(32, 32, 2, first_block=True)
        self.layer2 = resnet_block(32, 64, 2)
        
        # 池化后自动计算全连接输入
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)  # 增加 Dropout 概率
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# 3. 训练与评估函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda', model_save_path='best_mfpt_model.pth'):
    model.to(device)
    
    best_val_acc = 0.0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        model.train()
        
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels, _ in train_loader:  # 忽略元数据
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_train_loss, epoch_train_acc = running_loss / total, correct / total
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc)

        model.eval()
        
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:  # 忽略元数据
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.float()
                labels = labels.long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        epoch_val_loss, epoch_val_acc = running_loss / total, correct / total
        training_history['val_loss'].append(epoch_val_loss)
        training_history['val_acc'].append(epoch_val_acc)

        if epoch % 2 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs} | '
                  f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | '
                  f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            try:
                torch.save(model.state_dict(), model_save_path)
            except Exception as e:
                print(f"保存模型时出现错误: {e}")
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return training_history


def evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:  # 忽略元数据
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ["正常状态", "外圈故障", "变载荷外圈故障", "变载荷内圈故障"]
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(f'测试集准确率: {accuracy:.4f}')
    print('混淆矩阵:')
    print(conf_matrix)
    print('分类报告:')
    print(report)
    return y_true, y_pred, conf_matrix


# 4. SHAP可解释性分析
class DeepSHAP:
    def __init__(self, model, background_data, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        background = torch.tensor(background_data, dtype=torch.float32).to(device)
        self.explainer = shap.DeepExplainer(self.model, background)

    def explain(self, test_data, class_index=None):
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(self.device)
        shap_values = self.explainer.shap_values(test_tensor, check_additivity=False)  # 关闭additivity检查
        if isinstance(shap_values, list):
            return shap_values[class_index] if class_index is not None else shap_values
        else:
            return shap_values[:, :, class_index] if class_index is not None else shap_values

# 修改后的可视化函数，每个类别单独成图
def visualize_shap_values_by_class(shap_values, feature_names=None, class_names=None):
    num_classes = 4
    valid_shap_values = [shap for shap in shap_values if shap is not None]
    if not valid_shap_values:
        print("没有有效的SHAP值，无法进行可视化。")
        return

    # 假设所有有效的SHAP值具有相同的特征维度
    num_features = valid_shap_values[0].shape[1]
    if feature_names is None:
        feature_names = [f"频率特征-{i + 1}" for i in range(num_features)]
    if len(feature_names) != num_features:
        raise ValueError(f"特征名称数量({len(feature_names)})与SHAP特征数({num_features})不匹配")

    class_names = class_names or ["正常状态", "外圈故障", "变载荷外圈故障", "变载荷内圈故障"]

    for cls_idx in range(num_classes):
        cls_shap = shap_values[cls_idx]
        if cls_shap is None:
            print(f"类别 {class_names[cls_idx]} 没有有效的SHAP值，跳过可视化。")
            continue
        mean_abs_shap = np.abs(cls_shap).mean(axis=0)
        top_k = min(10, num_features)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        top_values = mean_abs_shap[top_indices]
        top_names = [feature_names[idx] for idx in top_indices]

        plt.figure(figsize=(8, 4))  # 单独创建每个类别的图
        plt.barh(range(top_k), top_values, color=plt.cm.viridis(np.linspace(0, 1, top_k)))
        plt.yticks(range(top_k), top_names, fontsize=8, rotation=0)
        plt.xlabel('平均|SHAP值|', fontsize=10)
        plt.title(f'{class_names[cls_idx]}的TOP{top_k}重要特征', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'shap_{class_names[cls_idx].replace(" ", "_")}_top10.png', dpi=300, bbox_inches='tight')
        plt.show()


# 新增：故障类别对比可视化
def visualize_fault_vs_normal(shap_values_dict, feature_names=None, class_names=None):
    if class_names is None:
        class_names = ["正常状态", "外圈故障", "变载荷外圈故障", "变载荷内圈故障"]

    base_class = 0  # 正常状态作为基准
    if shap_values_dict[base_class] is None:
        raise ValueError("正常状态类别无SHAP值，无法进行对比")

    base_shap = np.abs(shap_values_dict[base_class]).mean(axis=0)

    for fault_cls in [1, 2, 3]:
        if shap_values_dict[fault_cls] is None:
            print(f"跳过类别 {fault_cls}（无SHAP值）")
            continue

        fault_shap = np.abs(shap_values_dict[fault_cls]).mean(axis=0)
        diff = fault_shap - base_shap
        top_k = min(10, len(diff))
        top_indices = np.argsort(-np.abs(diff))[:top_k]
        top_features = [feature_names[i] for i in top_indices]
        top_diff = diff[top_indices]

        plt.figure(figsize=(12, 4))
        colors = ['#FF5733' if val < 0 else '#2ECC71' for val in top_diff]
        plt.barh(range(top_k), top_diff, color=colors, edgecolor='black')
        plt.yticks(range(top_k), top_features, fontsize=10)
        plt.xlabel('SHAP值差异（故障 - 正常）', fontsize=12)
        plt.title(f'{class_names[fault_cls]} vs {class_names[base_class]} 的TOP{top_k}差异特征', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'shap_diff_{class_names[fault_cls].replace(" ", "_")}_vs_normal.png', dpi=300)
        plt.show()


# 新增：训练曲线可视化函数

def plot_training_curves(training_history, num_epochs):
    epochs = np.arange(num_epochs)
    plt.figure(figsize=(12, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_history['train_loss'], label='训练损失')
    plt.plot(epochs, training_history['val_loss'], label='验证损失')
    plt.title('训练和验证损失曲线')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    loss_min = min(training_history['train_loss'] + training_history['val_loss'])
    loss_max = max(training_history['train_loss'] + training_history['val_loss'])
    plt.ylim(loss_min, loss_max * 1.05)
    loss_ticks = np.linspace(loss_min, loss_max * 1.05, 15)
    # 去掉与1.0非常接近的刻度，只保留真正的1.0
    loss_ticks = [tick for tick in loss_ticks if abs(tick - 1.01) > 1e-2]
    plt.yticks(sorted(set(np.append(loss_ticks, [1.0]))))
    plt.xticks(epochs)
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_history['train_acc'], label='训练准确率')
    plt.plot(epochs, training_history['val_acc'], label='验证准确率')
    plt.title('训练和验证准确率曲线')
    plt.xlabel('轮数')
    plt.ylabel('准确率')
    acc_min = min(training_history['train_acc'] + training_history['val_acc'])
    acc_max = max(training_history['train_acc'] + training_history['val_acc'])
    plt.ylim(acc_min, max(acc_max * 1.05, 1.0))
    acc_ticks = np.linspace(acc_min, max(acc_max * 1.05, 1.0), 15)
    acc_ticks = [tick for tick in acc_ticks if abs(tick - 1.01) > 1e-2]
    plt.yticks(sorted(set(np.append(acc_ticks, [1.0]))))
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout()
    plt.show()
   

# 新增：百分比混淆矩阵可视化函数
def plot_confusion_matrix_percentage(conf_matrix, class_names):
    norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(norm_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('百分比形式混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = norm_conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, '{:.2f}%'.format(norm_conf_matrix[i, j] * 100),
                     horizontalalignment="center",
                     color="white" if norm_conf_matrix[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()


# SHAP特征提取函数
def extract_key_features(class_shap_values, num_features_per_class=10):
    selected_features = set()
    for cls_idx in range(4):
        shap_values = class_shap_values[cls_idx]
        if shap_values is None:
            continue
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-num_features_per_class:][::-1]
        selected_features.update(top_indices)
    return sorted(selected_features)


# 可视化函数（新增对比版本）
def plot_training_comparison(history_original, history_selected, num_epochs):
    epochs = np.arange(num_epochs)
    plt.figure(figsize=(16, 6))

    # 损失对比
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_original['train_loss'], label='原始模型-训练损失', linestyle='-', color='tab:blue')
    plt.plot(epochs, history_original['val_loss'], label='原始模型-验证损失', linestyle='--', color='tab:blue')
    plt.plot(epochs, history_selected['train_loss'], label='特征模型-训练损失', linestyle='-', color='tab:orange')
    plt.plot(epochs, history_selected['val_loss'], label='特征模型-验证损失', linestyle='--', color='tab:orange')
    plt.title('损失曲线对比')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    all_loss = history_original['train_loss'] + history_original['val_loss'] + \
               history_selected['train_loss'] + history_selected['val_loss']
    loss_min = min(all_loss)
    loss_max = max(all_loss)
    plt.ylim(loss_min, loss_max * 1.05)
    loss_ticks = np.linspace(loss_min, loss_max * 1.05, 15)
    # 去掉与1.0非常接近的刻度，只保留真正的1.0
    loss_ticks = [tick for tick in loss_ticks if abs(tick - 1.01) > 1e-2]
    plt.yticks(sorted(set(np.append(loss_ticks, [1.0]))))  # 损失曲线也加1.0刻度损失曲线也加1.0刻度
    plt.xticks(epochs)
    plt.legend()

    # 准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_original['train_acc'], label='原始模型-训练准确率', linestyle='-', color='tab:blue')
    plt.plot(epochs, history_original['val_acc'], label='原始模型-验证准确率', linestyle='--', color='tab:blue')
    plt.plot(epochs, history_selected['train_acc'], label='特征模型-训练准确率', linestyle='-', color='tab:orange')
    plt.plot(epochs, history_selected['val_acc'], label='特征模型-验证准确率', linestyle='--', color='tab:orange')
    plt.title('准确率曲线对比')
    plt.xlabel('轮数')
    plt.ylabel('准确率')
    all_acc = history_original['train_acc'] + history_original['val_acc'] + \
              history_selected['train_acc'] + history_selected['val_acc']
    acc_min = min(all_acc)
    acc_max = max(all_acc)
    plt.ylim(acc_min, max(acc_max * 1.05, 1.0))
    acc_ticks = np.linspace(acc_min, max(acc_max * 1.05, 1.0), 15)
    acc_ticks = [tick for tick in acc_ticks if abs(tick - 1.01) > 1e-2]
    plt.yticks(sorted(set(np.append(acc_ticks, [1.0]))))  # 损失曲线也加1.0刻度
    plt.yticks(np.linspace(acc_min, acc_max * 1.05, 15))
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300)
    plt.show()


def plot_confusion_comparison(conf_original, conf_selected, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 原始模型混淆矩阵
    ax = axes[0]
    norm_conf = conf_original.astype('float') / conf_original.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(norm_conf, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('原始模型混淆矩阵')
    ax.set_xticks(np.arange(len(class_names)));
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45);
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{norm_conf[i, j]:.2%}', ha='center', va='center',
                    color='white' if norm_conf[i, j] > 0.5 else 'black')

    # 特征模型混淆矩阵
    ax = axes[1]
    norm_conf = conf_selected.astype('float') / conf_selected.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(norm_conf, interpolation='nearest', cmap=plt.cm.Oranges)
    ax.set_title('特征筛选模型混淆矩阵')
    ax.set_xticks(np.arange(len(class_names)));
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45);
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{norm_conf[i, j]:.2%}', ha='center', va='center',
                    color='white' if norm_conf[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig('confusion_comparison.png', dpi=300)
    plt.show()


def visualize_single_sample_shap(shap_values_dict, sample_data_dict, sample_metadata_dict, feature_names, class_names, background_data, model, device='cuda'):
    model.eval()
    for cls in range(4):
        if sample_data_dict[cls] is None or len(sample_data_dict[cls]) < 2:
            continue
        sample_data = sample_data_dict[cls][1]  # 取第二个样本
        metadata = sample_metadata_dict[cls][1]  # 取第二个样本的元数据
        deep_shap = DeepSHAP(model, background_data, device)
        shap_values_for_sample = deep_shap.explain(sample_data[np.newaxis, :], class_index=cls)
        current_shap = shap_values_for_sample[0]
        abs_shap = np.abs(current_shap)
        top_k = min(5, len(abs_shap))
        top_indices = np.argsort(-abs_shap)[:top_k]
        top_values = current_shap[top_indices]
        top_features = [feature_names[i] for i in top_indices]

        plt.figure(figsize=(8, 4))  # 新建一个独立的 figure
        colors = ['#2ECC71' if val > 0 else '#FF5733' for val in top_values]
        plt.barh(range(top_k), top_values[::-1], color=colors[::-1], height=0.6)
        plt.yticks(range(top_k), top_features[::-1], fontsize=10)
        plt.xlabel('SHAP值', fontsize=12)
        plt.title(f'{class_names[cls]} - 样本特征重要性TOP{top_k}\n样本元数据：{metadata[0]}（行号{metadata[1]}）', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'single_sample_shap_{class_names[cls].replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # 配置参数
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = r"D:\\虚拟c盘\\基于物理信息的特征神经网络模型的故障诊断\\数据预处理\\mfpt数据集处理后"  # 替换为实际路径
    batch_size = 32
    num_epochs = 30
    lr1 =0.0008
    lr2 = 0.00107   #resnet14 0.00107 ；cnn 0.00082
    lr3 = 0.0021
    skip_original_training = False  # 跳过原始模型训练
    skip_selected_training = False  # 跳过特征筛选模型训练
    class_names = ["正常状态", "外圈故障", "变载荷外圈故障", "变载荷内圈故障"]
    
    # 数据加载与划分（原始模型）
    print("[1/10] 加载原始数据集...")
    important_features = list(range(480))
    original_dataset = MFPTSegmentLoader(root_dir, important_features=important_features)
    dataset_size = len(original_dataset)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        original_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # 模型初始化（原始模型）
    print("[2/10] 初始化原始CNN模型...")
    sample_features, _, _ = original_dataset[0]  # 忽略元数据
    num_features = len(sample_features)
    #original_model = CNN1D(input_size=num_features, num_classes=4)
    original_model = Resnet10(input_size=num_features, num_classes=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(original_model.parameters(), lr=lr3)
    original_model_save_path = 'best_original_mfpt_model.pth'  # 原始模型保存路径

    # 模型训练（原始模型）
    original_history = {}
    if not skip_original_training:
        print("[3/10] 开始训练原始模型...")
        original_history = train_model(original_model, train_loader, val_loader, criterion, optimizer, num_epochs, device, original_model_save_path)
        # 训练曲线可视化（原始模型）
        print("[4/10] 可视化原始模型训练曲线...")
        plot_training_curves(original_history, num_epochs)
    else:
        print("[3/10] 跳过原始模型训练环节（skip_original_training=True）")
        original_model.load_state_dict(torch.load(original_model_save_path))

    # 模型评估（原始模型）
    print("[5/10] 评估原始模型测试集...")
    original_y_true, original_y_pred, original_conf_matrix = evaluate_model(original_model, test_loader, device)

    # 百分比混淆矩阵可视化（原始模型）
    print("[6/10] 可视化原始模型百分比混淆矩阵...")
    plot_confusion_matrix_percentage(original_conf_matrix, class_names)

    # SHAP分析（原始模型）
    print("[7/10] 准备原始模型背景数据...")
    background_sample_size = min(100, len(train_dataset))
    background_indices = np.random.choice(len(train_dataset), size=background_sample_size, replace=False)
    background_data = np.array([train_dataset[idx][0] for idx in background_indices])

    print("[7/10] 按类别提取原始模型测试样本...")
    class_test_data = {}
    class_test_metadata = {}  # 新增：存储元数据
    for cls in range(4):
        cls_indices = [idx for idx, (_, label, metadata) in enumerate(test_dataset) if label == cls]
        if cls_indices:
            class_test_data[cls] = np.array([test_dataset[idx][0] for idx in cls_indices])
            class_test_metadata[cls] = [test_dataset[idx][2] for idx in cls_indices]  # 存储元数据
        else:
            class_test_data[cls] = None
            class_test_metadata[cls] = None
        if class_test_data[cls] is not None:
            print(f"类别 {cls}({class_names[cls]}) 测试样本数: {len(class_test_data[cls])}")
        else:
            print(f"警告：类别 {cls} 无测试样本，跳过SHAP计算")

    print("[7/10] 按类别计算原始模型SHAP值...")
    deep_shap = DeepSHAP(original_model, background_data, device)
    class_shap_values = {}
    for cls in range(4):
        if class_test_data[cls] is None:
            class_shap_values[cls] = None
            continue
        try:
            class_shap_values[cls] = deep_shap.explain(class_test_data[cls], class_index=cls)
            print(f"类别 {cls} SHAP值形状: {class_shap_values[cls].shape}")
        except Exception as e:
            print(f"类别 {cls} 计算SHAP值失败: {e}")
            class_shap_values[cls] = None

    # 原始类别TOP特征可视化（原始模型）
    print("[7/10] 生成原始模型类别TOP特征图...")
    feature_names = [f"频率特征-{i + 1}" for i in range(num_features)]
    visualize_shap_values_by_class(
        shap_values=[class_shap_values[cls] for cls in range(4)],
        feature_names=feature_names,
        class_names=class_names
    )

    # 新增：单个样本SHAP可视化（取第二个样本并显示元数据）
    print("[7/10] 生成各类别代表样本SHAP图...")
    visualize_single_sample_shap(
        shap_values_dict=class_shap_values,
        sample_data_dict=class_test_data,
        sample_metadata_dict=class_test_metadata,
        feature_names=feature_names,
        class_names=class_names,
        background_data=background_data,
        model=original_model,
        device=device
    )

    # 故障vs正常差异可视化（原始模型）
    print("[7/10] 生成原始模型故障类别对比图...")
    visualize_fault_vs_normal(
        shap_values_dict=class_shap_values,
        feature_names=feature_names,
        class_names=class_names
    )

    # 提取关键特征
    print("[8/10] 提取每类TOP10特征...")
    selected_features = extract_key_features(class_shap_values, num_features_per_class=10)
    print(f"选中的关键特征索引（共{len(selected_features)}个）: {selected_features}")

    # 数据加载与划分（特征筛选模型）
    print("[9/10] 加载特征筛选数据集...")
    selected_dataset = MFPTSegmentLoader(root_dir, important_features=selected_features)
    dataset_size = len(selected_dataset)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset_selected, val_dataset_selected, test_dataset_selected = random_split(
        selected_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader_selected = DataLoader(train_dataset_selected, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader_selected = DataLoader(val_dataset_selected, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader_selected = DataLoader(test_dataset_selected, batch_size=batch_size, shuffle=False, num_workers=1)

    # 模型初始化（特征筛选模型）
    print("[9/10] 初始化特征筛选CNN模型...")
    sample_features_selected, _, _ = selected_dataset[0]  # 忽略元数据
    num_features_selected = len(sample_features_selected)
    #selected_model = CNN1D(input_size=num_features_selected, num_classes=4)
    selected_model = Resnet10(input_size=num_features_selected, num_classes=4)
    criterion_selected = nn.CrossEntropyLoss()
    optimizer_selected = optim.Adam(selected_model.parameters(), lr=lr3)
    selected_model_save_path = 'best_selected_mfpt_model.pth'  # 特征筛选模型保存路径

    # 模型训练（特征筛选模型）
    selected_history = {}
    if not skip_selected_training:
        print("[9/10] 开始训练特征筛选模型...")
        selected_history = train_model(selected_model, train_loader_selected, val_loader_selected, criterion_selected, optimizer_selected, num_epochs, device, selected_model_save_path)
        # 训练曲线可视化（特征筛选模型）
        print("[9/10] 可视化特征筛选模型训练曲线...")
        plot_training_curves(selected_history, num_epochs)
    else:
        print("[9/10] 跳过特征筛选模型训练环节（skip_selected_training=True）")
        selected_model.load_state_dict(torch.load(selected_model_save_path))

    # 模型评估（特征筛选模型）
    print("[9/10] 评估特征筛选模型测试集...")
    selected_y_true, selected_y_pred, selected_conf_matrix = evaluate_model(selected_model, test_loader_selected, device)

    # 百分比混淆矩阵可视化（特征筛选模型）
    print("[9/10] 可视化特征筛选模型百分比混淆矩阵...")
    plot_confusion_matrix_percentage(selected_conf_matrix, class_names)

    # 可视化对比
    print("[10/10] 绘制训练曲线对比...")
    plot_training_comparison(original_history, selected_history, num_epochs)

    print("[10/10] 绘制混淆矩阵对比...")
    plot_confusion_comparison(original_conf_matrix, selected_conf_matrix, class_names)

    print("\n全流程执行完毕！")