import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

read_noise = 1.265e-05
shot_noise = 0.00251

mean = 1 # out of 959
kSigmaCalibLevel = 959 # 1023 - black_level(64), copy from run_benchmark.py
k = shot_noise / kSigmaCalibLevel
sigma = read_noise / kSigmaCalibLevel / kSigmaCalibLevel 

input_bayer = torch.ones([3000,1]) * mean
shot_noise = torch.poisson(input_bayer / k) * k
read_noise = torch.randn(input_bayer.shape) * torch.sqrt(torch.tensor(sigma))

# 将张量转换为numpy数组以便绘图
shot_noise_np = shot_noise.numpy().flatten()

# 计算统计信息
mean_val = np.mean(shot_noise_np)
# 创建图形
plt.figure(figsize=(12, 8))

# 绘制直方图
plt.subplot(2, 2, 1)
counts, bins, patches = plt.hist(shot_noise_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Shot Noise Value')
plt.ylabel('Frequency')
plt.title('Histogram of Shot Noise')
plt.grid(True, alpha=0.3)

# # 添加均值和标准差的垂直线
# plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.6f}')
# plt.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1.5, label=f'±1 Std')
# plt.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1.5)
# plt.legend()

# 绘制累积分布函数
plt.subplot(2, 2, 2)
plt.hist(shot_noise_np, bins=50, cumulative=True, alpha=0.7, color='green', edgecolor='black', density=True)
plt.xlabel('Shot Noise Value')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Function')
plt.grid(True, alpha=0.3)

# 绘制箱线图
plt.subplot(2, 2, 3)
plt.boxplot(shot_noise_np, vert=True, patch_artist=True)
plt.ylabel('Shot Noise Value')
plt.title('Box Plot of Shot Noise')
plt.grid(True, alpha=0.3)

# # 绘制Q-Q图（与理论泊松分布比较）
# plt.subplot(2, 2, 4)
# # 计算理论分位数
# lambda_param = mean / k  # 泊松分布的lambda参数
# n = len(shot_noise_np)
# sorted_data = np.sort(shot_noise_np)

# # 生成理论分位数（基于泊松分布，但需要转换为乘以k后的尺度）
# # 这里我们使用泊松分布的百分位数
# percentiles = np.arange(1, n + 1) / (n + 1)
# # 注意：由于我们处理的是缩放后的泊松分布，理论分位数应该是泊松分位数乘以k
# try:
#     # 使用泊松分布的PPF（百分位函数）
#     from scipy import stats
#     theoretical_quantiles = stats.poisson.ppf(percentiles, lambda_param) * k
#     plt.scatter(theoretical_quantiles, sorted_data, alpha=0.6)
    
#     # 添加参考线
#     min_plot = min(np.min(theoretical_quantiles), np.min(sorted_data))
#     max_plot = max(np.max(theoretical_quantiles), np.max(sorted_data))
#     plt.plot([min_plot, max_plot], [min_plot, max_plot], 'r--', alpha=0.8)
    
#     plt.xlabel('Theoretical Quantiles (Poisson)')
#     plt.ylabel('Sample Quantiles')
#     plt.title('Q-Q Plot (vs Poisson Distribution)')
#     plt.grid(True, alpha=0.3)
# except ImportError:
#     plt.text(0.5, 0.5, 'scipy not available\nfor Q-Q plot', 
#              horizontalalignment='center', verticalalignment='center')
#     plt.title('Q-Q Plot (scipy required)')

plt.tight_layout()
plt.show()

# # 额外：打印直方图的详细信息
# print("\nHistogram bins information:")
# for i in range(min(5, len(bins)-1)):
#     print(f"  Bin {i+1}: [{bins[i]:.8f}, {bins[i+1]:.8f}] - {counts[i]} samples")
    
# if len(bins) > 6:
#     print("  ...")
#     for i in range(max(0, len(bins)-6), len(bins)-1):
#         print(f"  Bin {i+1}: [{bins[i]:.8f}, {bins[i+1]:.8f}] - {counts[i]} samples")

print("I am here")