# FlowMatching

本项目展示了如何用条件型 Flow Matching 神经网络在二维空间生成棋盘格分布的彩色点云，实现条件控制生成。核心功能为根据象限条件（四象限）生成指定位置和颜色的点样本。

## 主要功能
- 生成带有颜色的信息的四象限棋盘点云数据，并可视化保存。
- 基于条件（象限）训练Flow Matching神经网络实现点集“配对”生成。
- 支持通过不同条件生成特定象限的点，并可视化对比生成效果。

## 依赖环境
- Python 3.7+
- numpy
- torch
- matplotlib
- tqdm

建议使用如下命令安装依赖：
```bash
pip install numpy torch matplotlib tqdm
```

## 使用方法
直接运行 `example.py` 脚本：
```bash
python example.py
```

脚本主要流程为：
1. 生成带颜色的二维棋盘点云数据（每个点为五维向量x,y+三通道颜色）（自动保存为`sampled_points.png`）。
2. 训练条件型Flow Matching神经网络。
3. 训练后，分别以四种Condition在四个象限生成采样点并可视化（弹出结果窗口）。

## 输出说明
- `sampled_points.png`：原始采样点云的可视化。
- 运行结束后会弹出带有四个象限预测点的可视化窗口。

## 联系与参考
如有疑问欢迎提出 issue。本项目代码参考 [Flow Matching论文](https://arxiv.org/abs/2210.02747) 并配合条件控制生成思想实现。
