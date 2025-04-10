import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

# 参数设置（基于[17]描述）
n = 50  # 智能体数量
runs = 25  # 每组实验重复次数
epsilon_range = np.logspace(-2, 2, 50)  # ε对数刻度范围[10^-2,10^2]
S = np.eye(n)  # 噪声增益矩阵[17]
q = np.zeros(n)  # 噪声衰减参数[17]
# 生成固定初始条件（基于[15]）
np.random.seed(42)
theta0 = np.random.normal(50, 10, n)  # N(50,100)分布
# 存储结果
errors = []
variance=[]
for eps in epsilon_range:
    epsilon = eps * np.ones(n)  # 统一隐私参数[17]
    ci = 1 / (epsilon * (1 - np.abs(1 - 1)))  # 根据[15]公式调整
    epsilon1=eps*np.eye(n)
    variance=[]
    run_results = []
    var1 = (2 / (n ** 2)) *np.sum(1/ (epsilon ** 2))
    variance.append(var1)
    for _ in range(runs):
        # 动态系统模拟（简化的共识算法[12][14]）
        theta = theta0.copy()
        for _ in range(100):  # 固定迭代次数
            noise = laplace.rvs(scale=ci, size=n)  # 拉普拉斯噪声[9]
            theta = (np.eye(n) - 0.1 * np.ones((n, n)) / n) @ theta + noise  # 简化共识动态
        run_results.append(np.mean(theta) - np.mean(theta0))
    errors.extend(run_results)

# 绘制图4(a)[17][19]
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(np.repeat(epsilon_range, runs), np.abs(errors), alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Privacy parameter ε (log scale)')
plt.ylabel('|θ∞ - Ave(θ₀)|')
plt.title('Empirical error per run [Fig4(a)]')
# 绘制图4(b)[17]
plt.subplot(122)
sample_var = [np.var(errors[i * runs:(i + 1) * runs]) for i in range(len(epsilon_range))]
theory_var =8/ (n* epsilon_range ** 2)  # 理论值简化公式[17]
#plt.loglog(epsilon_range, variance, 'bo', label='sample_var')
plt.loglog(epsilon_range, sample_var, 'bo-', label='sample_var')
plt.loglog(epsilon_range, theory_var, 'r--', label='theory_var')
plt.xlabel('Privacy parameter ε (log scale)')
plt.ylabel('Variance of θ∞')
plt.title('Accuracy-Privacy trade-off [Fig4(b)]')
plt.legend()
plt.tight_layout()
plt.show()