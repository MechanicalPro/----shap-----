import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
from torch.nn import functional as F
# 设置中文字体（解决中文显示问题）

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 数据加载类（支持特征筛选） ======================
def add_noise_snr(data, desired_snr_db):
    """
    给定数据和目标信噪比（dB），返回加噪后的数据
    data: numpy.ndarray, shape=(样本数, 特征数)
    desired_snr_db: float, 目标信噪比（分贝）
    """
    signal_power = np.mean(data ** 2, axis=1, keepdims=True)
    snr_linear = 10 ** (desired_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    noisy_data = data + noise
    return noisy_data.astype(np.float32)  # 强制转换为float32


class CWRUDataset(Dataset):
    def __init__(self, root_dir, feature_size=450, selected_features=None, snr_db=None):
        self.root_dir = root_dir
        self.feature_size = feature_size
        self.selected_features = selected_features  # 新增：选择的特征索引
        self.snr_db = snr_db  # 新增：信噪比参数
        self.label_map = {
            "Normal.csv": 0,  # 正常状态
            "IR007.csv": 1,  # 内圈故障（Inner Race）
            "OR007.csv": 2,  # 外圈故障（Outer Race）
            "B007.csv": 3  # 滚动体故障（Ball）
        }
        self.features = None
        self.labels = None
        self.sample_indices = []  # 新增：记录每个样本在 CSV 文件中的行号
        self._load_data()
        self._print_class_distribution()

    def _load_data(self):
        print(f"加载数据路径: {self.root_dir}")
        if not os.path.isdir(self.root_dir):
            print(f"错误：{self.root_dir} 不是有效的目录。")
            return

        all_data = []
        all_labels = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".csv") and filename in self.label_map:
                file_path = os.path.join(self.root_dir, filename)
                try:
                    label = self.label_map[filename]
                    data = pd.read_csv(file_path, header=None).values[:, :self.feature_size]
                    if self.selected_features is not None:  # 筛选特征
                        data = data[:, self.selected_features]
                    data = data.astype(np.float32)
                    # 新增：加噪声
                    if self.snr_db is not None:
                        data = add_noise_snr(data, self.snr_db)
                    all_data.append(data)
                    all_labels.append(np.full(len(data), label, dtype=np.int64))
                    # 记录每个样本在 CSV 文件中的行号
                    self.sample_indices.extend([(filename, i) for i in range(len(data))])
                    print(f"从 {filename} 加载了 {len(data)} 个样本")
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")

        if all_data:
            self.features = np.vstack(all_data)
            self.labels = np.hstack(all_labels)
        else:
            print("错误：未加载任何有效数据。")

    def _print_class_distribution(self):
        if self.labels is None or len(self.labels) == 0:
            print("警告：标签数据为空，无法统计类别分布")
            return
        class_counts = np.bincount(self.labels, minlength=4)
        class_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
        print("\n数据集类别分布：")
        for cls, count in enumerate(class_counts):
            print(f"  {class_names[cls]}: {count} 个样本")

    def __len__(self):
        return len(self.labels) if self.labels is not None else 0

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


# ====================== 2. 模型构建（动态输入尺寸） ======================
##第一种CNN模型
class CWRUCNN(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(CWRUCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 动态计算全连接层输入尺寸（池化后尺寸 = input_size // 2 // 2 // 2 = input_size // 8）
        fc_input_size = (input_size // 8) * 64
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
# 在您的代码中添加以下模型：

# 1. 简单CNN（最轻量）
class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        fc_input_size = (input_size // 4) * 16
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 2. 深层CNN（更重）
class DeepCNN(nn.Module):

    def __init__(self, input_size, num_classes=4):
        super(DeepCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        fc_input_size = (input_size // 16) * 256
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class LightSEModule(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class CWRUCNN_LightSE(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.se1 = LightSEModule(16, reduction=32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.se2 = LightSEModule(32, reduction=32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.se3 = LightSEModule(64, reduction=32)
        self.pool3 = nn.MaxPool1d(2)

        fc_input_size = (input_size // 8) * 64
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.se1(self.bn1(self.conv1(x))))
        x = self.pool2(self.se2(self.bn2(self.conv2(x))))
        x = self.pool3(self.se3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class MultiScaleCNN(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(MultiScaleCNN, self).__init__()
        # 第一层多尺度卷积并联
        self.conv3 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 后续卷积层
        self.conv2 = nn.Conv1d(48, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3_2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # 动态计算全连接层输入尺寸
        self.seq_len = input_size // 8  # 池化后长度
        fc_input_size = self.seq_len * 64
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)
        x = torch.cat([x1, x2, x3], dim=1)  # (batch_size, 48, input_size)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)  # (batch_size, 64, seq_len)

        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x  # 加上这一行

# ResNet18的基本残差块
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNet18_1D(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(ResNet18_1D, self).__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class CWRUCNN_LightSE_Optimized(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super().__init__()
        
        # 第一层 - 减少通道数
        self.conv1 = nn.Conv1d(1, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(12)
        self.se1 = OptimizedLightSEModule(12, reduction=8)  # 更小的reduction
        self.pool1 = nn.MaxPool1d(2)

        # 第二层
        self.conv2 = nn.Conv1d(12, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(24)
        self.se2 = OptimizedLightSEModule(24, reduction=8)
        self.pool2 = nn.MaxPool1d(2)

        # 第三层
        self.conv3 = nn.Conv1d(24, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(48)
        self.se3 = OptimizedLightSEModule(48, reduction=8)
        self.pool3 = nn.MaxPool1d(2)

        # 更小的全连接层
        fc_input_size = (input_size // 8) * 48
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 64, bias=False),  # 减少到64
            nn.BatchNorm1d(64),  # 用BN替代Dropout
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.se1(x)
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.se2(x)
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.se3(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
# 定义 OptimizedLightSEModule
class OptimizedLightSEModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
class CWRUCNN_LightSE_Ultra(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super().__init__()
        
        # 使用更小的通道数和优化的SE模块
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.se1 = UltraLightSEModule(8, reduction=4)
        
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(16)
        self.se2 = UltraLightSEModule(16, reduction=4)
        
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(32)
        self.se3 = UltraLightSEModule(32, reduction=4)
        
        # 计算feature map尺寸: input_size // 8 (三次stride=2)
        fc_input_size = (input_size // 8) * 32
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),  # 这里保持不变
            nn.Linear(32, num_classes)
        )
        
        # 添加SHAP模式标志
        self.shap_mode = False

    def set_shap_mode(self, mode=True):
        """设置SHAP分析模式，关闭inplace操作"""
        self.shap_mode = mode
        # 同时设置SE模块的SHAP模式
        self.se1.shap_mode = mode
        self.se2.shap_mode = mode
        self.se3.shap_mode = mode

    def forward(self, x):
        x = x.unsqueeze(1)
        
        # 根据模式选择是否使用inplace操作
        if self.shap_mode:
            # SHAP模式：不使用inplace
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.se1(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.se2(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.se3(x)
        else:
            # 训练模式：使用inplace提升性能
            x = F.relu(self.bn1(self.conv1(x)), inplace=True)
            x = self.se1(x)
            
            x = F.relu(self.bn2(self.conv2(x)), inplace=True)
            x = self.se2(x)
            
            x = F.relu(self.bn3(self.conv3(x)), inplace=True)
            x = self.se3(x)
        
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class UltraLightSEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 使用全局平均池化 + 单个线性层
        hidden_dim = max(2, channels // reduction)  # 最小为2
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        # 动态构建激活层
        self.linear1 = nn.Linear(channels, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # SHAP模式标志
        self.shap_mode = False

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        # 高效的squeeze操作
        y = self.squeeze(x).view(b, c)
        y = self.linear1(y)
        
        # 根据模式选择激活函数
        if self.shap_mode:
            y = F.relu(y)  # 不使用inplace
        else:
            y = F.relu(y, inplace=True)  # 使用inplace
            
        y = self.linear2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y


# ====================== 3. 训练函数（增加保存路径参数） ======================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda', model_save_path='best_cwru_model.pth'):
    torch.backends.cudnn.benchmark = True
    if device == 'cuda':
        torch.cuda.empty_cache()
    model.to(device)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=5, verbose=True, min_lr=1e-6
    )
    
    # 早停机制
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15  # 增加patience
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 温和的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        model.eval()
        running_val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        # 学习率调度
        scheduler.step(epoch_val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 2 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs} | '
                  f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | '
                  f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | '
                  f'LR: {current_lr:.6f}')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f'\n训练完成！最佳验证准确率: {best_val_acc:.4f}')
    return model_save_path, train_losses, train_accuracies, val_losses, val_accuracies

# ====================== 4. 评估函数（保持不变） ======================
def evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=1)

    print(f'\n测试集评估结果：')
    print(f'准确率: {accuracy:.4f}')
    print('混淆矩阵:\n', conf_matrix)
    print('分类报告:\n', report)

    return y_true, y_pred, conf_matrix


# ====================== 5. 新增：训练曲线可视化函数 ======================
def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, num_epochs=None, title_prefix=""):
    import numpy as np
    plt.figure(figsize=(12, 6))

    # 使用实际的训练轮数，而不是预设的num_epochs
    actual_epochs = len(train_losses)  # 使用实际的训练轮数
    epochs = list(range(1, actual_epochs + 1))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='训练损失')
    plt.plot(epochs, val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(f'{title_prefix}训练与验证损失曲线')
    plt.legend()
    
    # 设置x轴刻度
    if actual_epochs <= 10:
        plt.xticks(epochs)  # 如果轮数少，显示所有epoch
    else:
        plt.xticks(range(1, actual_epochs + 1, max(1, actual_epochs // 10)))  # 否则显示大约10个刻度
    
    # 设置更详细的纵坐标刻度
    loss_min = min(train_losses + val_losses)
    loss_max = max(train_losses + val_losses)
    plt.yticks(np.linspace(loss_min, loss_max, num=10))  # 减少到10个刻度
    plt.tick_params(axis='y', labelsize=10)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='训练准确率')
    plt.plot(epochs, val_accuracies, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title(f'{title_prefix}训练与验证准确率曲线')
    plt.legend()
    
    # 设置x轴刻度
    if actual_epochs <= 10:
        plt.xticks(epochs)  # 如果轮数少，显示所有epoch
    else:
        plt.xticks(range(1, actual_epochs + 1, max(1, actual_epochs // 10)))  # 否则显示大约10个刻度
    
    # 设置更详细的纵坐标刻度
    acc_min = min(train_accuracies + val_accuracies)
    acc_max = max(train_accuracies + val_accuracies)
    plt.yticks(np.linspace(acc_min, acc_max, num=10))  # 减少到10个刻度
    plt.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    plt.show()


# ====================== 6. 新增：百分比混淆矩阵可视化函数 ======================
def plot_confusion_matrix(conf_matrix, class_names, title="混淆矩阵（百分比形式）"):
    plt.figure(figsize=(8, 6))
    norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(norm_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = norm_conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f'{conf_matrix[i, j] / conf_matrix[i].sum() * 100:.2f}%',
                     horizontalalignment="center",
                     color="white" if norm_conf_matrix[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()


# ====================== 7. SHAP可解释性分析（含特征提取） ======================
class DeepSHAP:
    def __init__(self, model, background_data, device='cuda'):
        self.model = model.to(device)
        self.device = device
        background = torch.tensor(background_data, dtype=torch.float32).to(device)
        
        # 在创建explainer之前设置SHAP模式
        if hasattr(self.model, 'set_shap_mode'):
            self.model.set_shap_mode(True)
            
        self.explainer = shap.DeepExplainer(self.model, background)

    def explain(self, test_data, class_index=None):
        was_training = self.model.training
        
        # 确保SHAP模式开启
        if hasattr(self.model, 'set_shap_mode'):
            self.model.set_shap_mode(True)
            
        self.model.train()  # 切换到训练模式
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(self.device)
        shap_values = self.explainer.shap_values(test_tensor, check_additivity=False)
        
        # 恢复原始状态
        if hasattr(self.model, 'set_shap_mode'):
            self.model.set_shap_mode(False)
        if not was_training:
            self.model.eval()
            
        if isinstance(shap_values, list):
            return shap_values[class_index] if class_index is not None else shap_values
        else:
            return shap_values[:, :, class_index] if class_index is not None else shap_values



def visualize_shap_values_by_class(shap_values, feature_names, class_names):
    num_classes = 4
    top_features_all = []  # 收集所有类别的TOP10特征索引

    for cls_idx in range(num_classes):
        cls_shap = shap_values.get(cls_idx, None)
        if cls_shap is None or cls_shap.size == 0:
            print(f'{class_names[cls_idx]} 无有效特征，跳过可视化')
            continue

        mean_abs_shap = np.abs(cls_shap).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-10:][::-1]  # 提取TOP10特征索引
        top_features_all.extend(top_indices)  # 合并所有类别特征

        mean_abs_shap = np.abs(cls_shap).mean(axis=0)
        top_values = mean_abs_shap[top_indices]
        top_names = [feature_names[i] for i in top_indices]

        plt.figure(figsize=(8, 4))
        plt.barh(range(10), top_values, color=plt.cm.viridis(np.linspace(0, 1, 10)))
        plt.yticks(range(10), top_names, fontsize=8, rotation=0)
        plt.xlabel('平均|SHAP值|', fontsize=10)
        plt.title(f'{class_names[cls_idx]}的TOP10重要特征', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'shap_{class_names[cls_idx].replace(" ", "_")}_top10.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 去重并排序特征索引
    selected_features = sorted(list(set(top_features_all)))  # 去重，确保唯一特征
    print(f"\n提取的TOP特征索引（去重后）: {selected_features}")
    print(f"关键特征数量: {len(selected_features)}")

    return selected_features  # 返回选择的特征索引


def visualize_fault_vs_normal(shap_values_dict, feature_names, class_names):
    base_class = 0  # 正常状态
    base_shap = np.abs(shap_values_dict[base_class]).mean(axis=0)

    for fault_cls in [1, 2, 3]:
        fault_data = shap_values_dict[fault_cls]
        if fault_data is None or fault_data.size == 0:
            print(f"警告：类别 {class_names[fault_cls]} 无有效SHAP值，跳过对比")
            continue

        fault_shap = np.abs(fault_data).mean(axis=0)
        diff = fault_shap - base_shap
        top_indices = np.argsort(-np.abs(diff))[:10]  # 差异最大的10个特征
        top_features = [feature_names[i] for i in top_indices]
        top_diff = diff[top_indices]

        plt.figure(figsize=(12, 4))
        colors = ['#FF5733' if val < 0 else '#2ECC71' for val in top_diff]
        plt.barh(range(10), top_diff, color=colors, edgecolor='black')
        plt.yticks(range(10), top_features, fontsize=10)
        plt.xlabel('SHAP值差异（故障 - 正常）', fontsize=12)
        plt.title(f'{class_names[fault_cls]} vs 正常状态的TOP10差异特征', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'shap_diff_{class_names[fault_cls].replace(" ", "_")}_vs_normal.png', dpi=300)
        plt.show()


# 新增：单个样本 SHAP 可视化
def visualize_single_sample_shap(shap_values_dict, sample_data_dict, feature_names, class_names, background_data, model,
                                 device='cuda', sample_indices=None):
    model.eval()
    for cls in range(4):
        if sample_data_dict[cls] is None or len(sample_data_dict[cls]) == 0:
            print(f"类别 {class_names[cls]} 无样本，跳过SHAP计算")
            continue
        sample_data = sample_data_dict[cls][0]  # 取第一个样本
        sample_index = sample_indices[cls][0]  # 获取样本在 CSV 文件中的行号
        deep_shap = DeepSHAP(model, background_data, device)
        #deep_shap = KernelSHAP(model_original, background_data, device)
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
        plt.title(f'{class_names[cls]} - 样本特征重要性TOP{top_k} (文件: {sample_index[0]}, 行号: {sample_index[1]})', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'single_sample_shap_{class_names[cls].replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
   

# ====================== 8. 主函数（支持跳过训练） ======================
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True  # 启用 cudnn benchmark模式
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 加速
    # root_dir = r"D:\本科毕业设计\可解释性机器学习\频域特征提取\cwru1"
    root_dir = r"E:\Artifical intelligence 666\\某人的本科毕设\\本科毕业设计\\可解释性机器学习\\数据预处理\\cwru数据集处理后"
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.0005   #resnet的训练率0.0002；cnn的训练率0.00082
    feature_size = 448
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skip_original_training = False  # 跳过原始模型训练
    skip_top_training = False  # 跳过筛选后模型训练
    desired_snr_db = 10 # 设置信噪比（单位dB），例如20dB，可自行调整

    print(f"当前设备: {device}")
   

    # 定义类别名称（修复此处）
    class_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]

    # 1. 原始数据加载与划分（450维特征）
    print("[1/10] 加载原始数据集...")
    dataset_original = CWRUDataset(root_dir, feature_size=feature_size, snr_db=desired_snr_db)
    dataset_size = len(dataset_original)
    train_size, val_size, test_size = int(0.6 * dataset_size), int(0.2 * dataset_size), dataset_size - int(
        0.6 * dataset_size) - int(0.2 * dataset_size)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset_original, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. 原始模型处理
    print("[2/10] 处理原始模型...")
    model_original = CWRUCNN_LightSE_Ultra(input_size=feature_size, num_classes=4)
    
    #model_original = ResNet18_1D(input_size=feature_size, num_classes=4)
    #model_original = MultiScaleCNN(input_size=feature_size, num_classes=4)
    #model_original = DeepCNN(input_size=feature_size, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model_original.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5,  # 大幅减少权重衰减
        betas=(0.9, 0.999)
    )
    model_save_path_original = 'best_cwru_original_model.pth'

    if not skip_original_training:
        print("[2-1] 训练原始模型...")
        import time
        t1_start = time.time()
        model_save_path_original, train_losses_original, train_accuracies_original, val_losses_original, val_accuracies_original = train_model(
            model_original, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path_original
        )
        t1 = time.time() - t1_start
        print(f"原始模型训练时间: {t1:.2f} 秒")
    else:
        print("[2-2] 跳过原始模型训练，直接加载已有模型...")
        if not os.path.exists(model_save_path_original):
            raise FileNotFoundError(
                f"未找到原始模型文件 {model_save_path_original}，请先设置 skip_original_training=False 进行训练")
        model_original.load_state_dict(torch.load(model_save_path_original))

    # 在主函数的SHAP分析部分修改：
# 3. SHAP分析并提取TOP特征
    print("[5/10] 执行SHAP分析并提取关键特征...")
    background_indices = np.random.choice(len(train_dataset), size=min(100, len(train_dataset)), replace=False)
    background_data = np.array([train_dataset[idx][0].numpy() for idx in background_indices])

# 确保模型处于正确模式
    model_original.eval()  # 设置为评估模式
    if hasattr(model_original, 'set_shap_mode'):
            model_original.set_shap_mode(True)  # 开启SHAP模式

    deep_shap = DeepSHAP(model_original, background_data, device)

# ... SHAP分析代码保持不变 ...

# SHAP分析完成后恢复训练模式
    if hasattr(model_original, 'set_shap_mode'):
        model_original.set_shap_mode(False)  # 关闭SHAP模式

    class_test_data = {cls: [] for cls in range(4)}
    class_test_indices = {cls: [] for cls in range(4)}  # 新增：记录每个类别的样本索引
    for idx in range(len(test_dataset)):
        data, label = test_dataset[idx]
        cls = label.item()
        class_test_data[cls].append(data.numpy())
        class_test_indices[cls].append(dataset_original.sample_indices[test_dataset.indices[idx]])  # 记录样本索引
    for cls in range(4):
        class_test_data[cls] = np.array(class_test_data[cls]) if class_test_data[cls] else None

    class_shap_values = {}
    for cls in range(4):
        if class_test_data[cls] is not None:
            class_shap_values[cls] = deep_shap.explain(class_test_data[cls], class_index=cls)

    feature_names = [f"频率特征-{i + 1}" for i in range(feature_size)]
    selected_features = visualize_shap_values_by_class(class_shap_values, feature_names, class_names)

    # 新增：单个样本 SHAP 可视化
    print("[5/10] 生成各类别代表样本SHAP图...")
    visualize_single_sample_shap(
        shap_values_dict=class_shap_values,
        sample_data_dict=class_test_data,
        feature_names=feature_names,
        class_names=class_names,
        background_data=background_data,
        model=model_original,
        device=device,
        sample_indices=class_test_indices
    )

    # 故障vs正常差异可视化（原始模型）
    print("[5/10] 生成原始模型故障类别对比图...")
    visualize_fault_vs_normal(
        shap_values_dict=class_shap_values,
        feature_names=feature_names,
        class_names=class_names
    )

    # 4. 使用TOP特征构建新数据集
    print(f"[6/10] 使用{len(selected_features)}个关键特征重新加载数据...")
    dataset_top = CWRUDataset(
        root_dir=root_dir,
        feature_size=feature_size,
        selected_features=selected_features,
        snr_db=desired_snr_db  # 保持一致的噪声设置
    )
    dataset_top_size = len(dataset_top)
    train_size_top, val_size_top, test_size_top = int(0.6 * dataset_top_size), int(
        0.2 * dataset_top_size), dataset_top_size - int(0.6 * dataset_top_size) - int(0.2 * dataset_top_size)
    train_dataset_top, val_dataset_top, test_dataset_top = random_split(
        dataset_top, [train_size_top, val_size_top, test_size_top], generator=torch.Generator().manual_seed(42)
    )
    train_loader_top = DataLoader(train_dataset_top, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader_top = DataLoader(val_dataset_top, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader_top = DataLoader(test_dataset_top, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. 筛选后模型处理
    print(f"[7/10] 处理TOP特征模型（输入尺寸={len(selected_features)}）...")
    

    model_top = MultiScaleCNN(input_size=len(selected_features), num_classes=4)
    criterion_top = nn.CrossEntropyLoss()
    optimizer_top = optim.Adam(model_top.parameters(), lr=learning_rate)
    model_save_path_top = 'best_cwru_top_model.pth'

    if not skip_top_training:
        print("[7-1] 训练TOP特征模型...")
        t2_start = time.time()
        model_save_path_top, train_losses_top, train_accuracies_top, val_losses_top, val_accuracies_top = train_model(
            model_top, train_loader_top, val_loader_top, criterion_top, optimizer_top, num_epochs, device, model_save_path_top
        )
        t2 = time.time() - t2_start
        print(f"TOP特征模型训练时间: {t2:.2f} 秒")
    else:
        print("[7-2] 跳过TOP特征模型训练，直接加载已有模型...")
        if not os.path.exists(model_save_path_top):
            raise FileNotFoundError(
                f"未找到TOP特征模型文件 {model_save_path_top}，请先设置 skip_top_training=False 进行训练")
        model_top.load_state_dict(torch.load(model_save_path_top))

    # 6. 评估与可视化（仅在训练后执行，跳过训练时可注释以下部分）
    # 在主函数的可视化部分修改调用方式：
if not skip_original_training or not skip_top_training:
    # 原始模型可视化（需训练后执行）
    if not skip_original_training:
        print("[8/10] 可视化原始模型训练曲线...")
        # 移除num_epochs参数，让函数自动计算
        plot_training_curves(train_losses_original, train_accuracies_original, val_losses_original,
                             val_accuracies_original, title_prefix="原始模型 ")
        print("[8/20] 评估原始模型并可视化混淆矩阵...")
        y_true_original, y_pred_original, conf_matrix_original = evaluate_model(model_original, test_loader, device)
        plot_confusion_matrix(conf_matrix_original, class_names, "原始模型混淆矩阵")

    # 筛选后模型可视化（需训练后执行）
    if not skip_top_training:
        print("[9/10] 可视化TOP特征模型训练曲线...")
        # 移除num_epochs参数，让函数自动计算
        plot_training_curves(train_losses_top, train_accuracies_top, val_losses_top, val_accuracies_top,
                             title_prefix="TOP特征模型 ")
        print("[10/10] 评估TOP特征模型并可视化混淆矩阵...")
        y_true_top, y_pred_top, conf_matrix_top = evaluate_model(model_top, test_loader_top, device)
        plot_confusion_matrix(conf_matrix_top, class_names, "TOP特征模型混淆矩阵")

    print("\n全流程执行完毕！")