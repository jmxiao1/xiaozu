import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数设置
alpha_values = np.linspace(0.01, 0.99, 100)  # 参数 alpha 的范围
s_values = np.linspace(0.01, 1.99, 200)  # 参数 s 的范围
alpha_values, s_values = np.meshgrid(alpha_values, s_values)

# 定义局部目标函数
def local_objective_function(alpha, s):
    q = alpha + (1 - alpha) * abs(s - 1)
    return (s**2 * q**2) / (alpha**2 * (1 - abs(s - 1))**2 * (1 - q**2))

# 计算目标函数值
phi_values = local_objective_function(alpha_values, s_values)
phi_values[phi_values >7] = 7  # 将负值设为0
# 绘制三维曲面图
# 创建一个大小为12x8的图形窗口
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
#surf = ax.plot_surface(alpha_values, s_values, phi_values, cmap='viridis', edgecolor='none')
surf = ax.plot_surface(s_values , alpha_values, phi_values, cmap='viridis', edgecolor='none')
ax.view_init(elev=15, azim=-100)  # 设置视角
ax.set_aspect('auto')  # 先将比例设置为自动
ax.set_box_aspect((2, 1, 1))
# 设置标签和标题
ax.set_xlabel('s')
ax.set_ylabel('Alpha')
ax.set_zlabel('Objective Function Value')
ax.set_title('Local Objective Function φ(α, s)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()
