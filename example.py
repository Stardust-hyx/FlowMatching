import numpy as np
import torch
from tqdm import tqdm

# 设备自动选择（优先使用GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

# ---------------------- 生成带颜色的四象限2×2棋盘数据 ----------------------
grid_size = 2  # 2×2棋盘
x_range = (-2, 2)  # x轴范围：-2到2（中间0为分界）
y_range = (-2, 2)  # y轴范围：-2到2（中间0为分界）
n_per_grid = 250  # 每个格子的点数

# 生成每个格子的点（集中在格子内，增强棋盘感）
sampled_points = []
# 第一象限（x∈(0,2), y∈(0,2)）：红色
x1 = np.random.uniform(0, x_range[1], n_per_grid)  # x>0
y1 = np.random.uniform(0, y_range[1], n_per_grid)  # y>0
color1 = np.tile([1, 0, 0], (n_per_grid, 1))  # 红色
sampled_points.append(np.hstack([x1[:, None], y1[:, None], color1]))

# 第二象限（x∈(-2,0), y∈(0,2)）：黄色
x2 = np.random.uniform(x_range[0], 0, n_per_grid)  # x<0
y2 = np.random.uniform(0, y_range[1], n_per_grid)  # y>0
color2 = np.tile([1, 1, 0], (n_per_grid, 1))  # 黄色
sampled_points.append(np.hstack([x2[:, None], y2[:, None], color2]))

# 第三象限（x∈(-2,0), y∈(-2,0)）：绿色
x3 = np.random.uniform(x_range[0], 0, n_per_grid)  # x<0
y3 = np.random.uniform(y_range[0], 0, n_per_grid)  # y<0
color3 = np.tile([0, 1, 0], (n_per_grid, 1))  # 绿色
sampled_points.append(np.hstack([x3[:, None], y3[:, None], color3]))

# 第四象限（x∈(0,2), y∈(-2,0)）：蓝色
x4 = np.random.uniform(0, x_range[1], n_per_grid)  # x>0
y4 = np.random.uniform(y_range[0], 0, n_per_grid)  # y<0
color4 = np.tile([0, 0, 1], (n_per_grid, 1))  # 蓝色
sampled_points.append(np.hstack([x4[:, None], y4[:, None], color4]))

# 合并为5维数据（x,y,R,G,B）
sampled_points = np.vstack(sampled_points)
sampled_points = torch.tensor(sampled_points, dtype=torch.float32, device=device)

# ---------------------- 可视化 sampled_points 并保存图片 ----------------------
import matplotlib.pyplot as plt

with torch.no_grad():
    sp_np = sampled_points.detach().cpu().numpy()
    xy = sp_np[:, :2]
    rgb = np.clip(sp_np[:, 2:5], 0.0, 1.0)

plt.figure(figsize=(6, 6))
plt.scatter(xy[:, 0], xy[:, 1], c=rgb, s=8, alpha=0.8)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid(True, linestyle=':', linewidth=0.6)
plt.title("Sampled Points (Checkerboard)")
plt.tight_layout()
plt.savefig("sampled_points.png", dpi=200, bbox_inches='tight')
plt.close()

# ---------------------- 定义象限Condition生成函数 ----------------------
def get_cond(coord):
    """根据坐标(x,y)生成二维binary象限Condition"""
    x, y = coord[:, 0], coord[:, 1]
    cond = torch.zeros(coord.shape[0], 2, device=coord.device, dtype=coord.dtype)  # 初始化2维Condition
    # 第一象限：x>0且y>0 → [1,1]
    cond[(x > 0) & (y > 0)] = torch.tensor([1, 1], dtype=coord.dtype, device=coord.device)
    # 第二象限：x<0且y>0 → [0,1]
    cond[(x < 0) & (y > 0)] = torch.tensor([0, 1], dtype=coord.dtype, device=coord.device)
    # 第三象限：x<0且y<0 → [0,0]
    cond[(x < 0) & (y < 0)] = torch.tensor([0, 0], dtype=coord.dtype, device=coord.device)
    # 第四象限：x>0且y<0 → [1,0]
    cond[(x > 0) & (y < 0)] = torch.tensor([1, 0], dtype=coord.dtype, device=coord.device)
    return cond

# ---------------------- 定义神经网络结构 ----------------------
class MLP(torch.nn.Module):
    def __init__(self, layers=5, channels=512):
        super().__init__()
        # 输入维度：5（数据）+1（时间t）+2（Condition）=8
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(8, channels),
            torch.nn.ReLU(),
            *[torch.nn.Sequential(
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU()
            ) for _ in range(layers-2)],
            torch.nn.Linear(channels, 5)  # 输出5维（与x1-x0匹配）
        )
    
    def forward(self, xt, t, cond):
        t = t.unsqueeze(-1)  # 时间t扩展为(batch,1)
        x = torch.cat([xt, t, cond], dim=-1)  # 拼接输入
        return self.layers(x)

# ---------------------- 初始化模型与优化器 ----------------------
model = MLP(layers=5, channels=512).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------------- 训练循环 ----------------------
data = sampled_points
training_steps = 10000
batch_size = 96
pbar = tqdm(range(training_steps))

for i in pbar:
    # 1. 采样目标点x1（含坐标和颜色）
    idx = torch.randint(len(data), (batch_size,), device=device)
    x1 = data[idx]
    
    # 2. 生成初始噪声x0（5维标准正态分布）
    x0 = torch.randn_like(x1)
    
    # 3. 随机时间步t∈[0,1)
    t = torch.rand(batch_size, device=device)
    
    # 4. 插值点xt = (1-t)*x0 + t*x1
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    
    # 5. 生成当前批次的Condition（基于x1的坐标）
    cond = get_cond(x1[:, :2])  # x1前两维是坐标
    
    # 6. 模型预测与损失计算
    pred = model(xt, t, cond)
    target = x1 - x0  # Flow Matching目标：预测x1与x0的差值
    loss = ((pred - target) **2).mean()
    
    # 7. 反向传播
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    pbar.set_postfix(loss=loss.item())


""" 推理 """
import matplotlib.pyplot as plt

def generate_by_cond(cond_vec, num_points=512, num_gen_steps=10):
    """根据指定的Condition生成点"""
    x0 = torch.randn(num_points, 5, device=device)  # 初始噪声
    t_start, t_end = 0.99, 0.02  # 线性时间调度：由大到小
    cond = torch.tensor([cond_vec] * num_points, dtype=torch.float32, device=device)  # 重复Condition
    
    # 多步迭代生成
    xt = x0
    model.eval()
    for k in range(num_gen_steps):
        with torch.no_grad():
            tk = t_start + (t_end - t_start) * (k / max(1, (num_gen_steps - 1)))
            t = torch.full((num_points,), tk, device=device)
            pred = model(xt, t, cond)
        xt = xt + pred * (1/num_gen_steps)  # 逐步更新
    
    # 转换为numpy并截断颜色值到[0,1]
    generated = xt.detach().cpu().numpy()
    generated[:, 2:5] = np.clip(generated[:, 2:5], 0, 1)
    return generated

# 生成四个象限的点并可视化
plt.figure(figsize=(8,8))

# 第一象限（Condition [1,1]）
points1 = generate_by_cond([1,1], num_points=512)
plt.scatter(points1[:,0], points1[:,1], c=points1[:,2:5], alpha=0.7)

# 第二象限（Condition [0,1]）
points2 = generate_by_cond([0,1], num_points=512)
plt.scatter(points2[:,0], points2[:,1], c=points2[:,2:5], alpha=0.7)

# 第三象限（Condition [0,0]）
points3 = generate_by_cond([0,0], num_points=512)
plt.scatter(points3[:,0], points3[:,1], c=points3[:,2:5], alpha=0.7)

# 第四象限（Condition [1,0]）
points4 = generate_by_cond([1,0], num_points=512)
plt.scatter(points4[:,0], points4[:,1], c=points4[:,2:5], alpha=0.7)

# 绘制棋盘格边界（x=0和y=0）
plt.axvline(x=0, color='black', linestyle='--')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid(True)
plt.legend()
plt.title("2×2 Checkerboard Squares with Colors (Flow Matching)")
plt.show()
