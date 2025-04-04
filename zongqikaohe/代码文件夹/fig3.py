import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from numpy.linalg import norm
from scipy.linalg import eigvals
# 设置随机种子以保证可重复性
np.random.seed(42)
# 1. 初始化参数（根据论文[15][16]设置）
n = 50  # 代理数量
p = 0.1  # Bernoulli分布参数
delta =1  # 隐私参数[15]
epsilon = 0.1 *np.ones(n) # 隐私预算[16]
num_runs = 10000  # 模拟次数(论文中使用10^4次)[16]
tolerance = 1e-2  # 收敛容差[16]
num_iterations = 500# 最大迭代次数
Pi_n = np.ones((n, n)) / n#投影矩阵



# 2. 创建随机图（根据论文[15]描述）
def create_random_graph(n, p):
    """创建随机图，边权是两个独立伯努利变量的和"""
    G = nx.erdos_renyi_graph(n, p)
    for u, v in G.edges():
        # 边权是两个i.i.d伯努利变量之和
        G[u][v]['weight'] = np.random.binomial(1, p) + np.random.binomial(1, p)
    return G


# 3. 构建拉普拉斯矩阵
def construct_laplacian(G):
    """构建图的拉普拉斯矩阵"""
    n = G.number_of_nodes()
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i, j] = sum(G[i][j]['weight'] for j in G.neighbors(i))
            elif G.has_edge(i, j):
                L[i, j] = -G[i][j]['weight']
    return L


# 4. 分布式共识算法实现（根据论文公式(12)-(14)[16]）
def dp_consensus(L, initial_states, s, alpha=1e-6, delta=1, epsilon=0.1, max_iter=num_iterations, tol=tolerance):
    """
    实现差分隐私平均共识算法
    参数:
        L: 拉普拉斯矩阵
        initial_states: 初始状态
        s: 噪声增益参数(论文中的s)[16]
        alpha: 噪声衰变参数(接近0)[16]
    """
    n = len(initial_states)
    theta = initial_states.copy()
    theta_history = [theta.copy()]
    theta0=theta.copy()
    # 计算q_i参数[16]
    q = alpha + (1 - alpha) * abs(s - 1)
    qmax = np.max(q)
    c = delta * q / (epsilon * (q - abs(s - 1)))  # [16]
    #var1=(2*(delta**2)/(n**2))*np.sum((s**2)*(q**2)/((epsilon**2)*((q-abs(s-1))**2)*(1-q**2)))
    #var1 = (2  / (n ** 2)) * np.sum(c**2)
    var1 = (2 *(delta**2) / (n ** 2)) * np.sum((s ** 2) * (q ** 2) / ((epsilon ** 2) * ((q - abs(s - 1)) ** 2) * (1 - q ** 2)))
    A = np.eye(n) - 0.5 * L #s * np.eye(n)
    B = s * np.eye(n) - 0.5 * L
    A_minus_Pi = A - Pi_n# 计算 A - Π_n
    # 计算 A - Π_n 的谱半径 λ
    eigenvalues = eigvals(A_minus_Pi)
    lambda_max = np.max(np.abs(eigenvalues))
    mu=0.84
    for k in range(max_iter):
        # 生成噪声（拉普拉斯分布）[16]
        noise = np.random.laplace(0, c, size=n)

        # 更新状态[16]
        #theta_new = theta - L @ theta + noise * (s ** k)  # 这里简化了公式
        theta_new = A @ theta + B @ noise # 这里简化了公式
        theta_history.append(theta_new.copy())
        theta = theta_new

        # 检查收敛
        if norm(theta - np.mean(theta)) < tol:
            break
    #kmax= np.log(np.linalg.norm(theta0-np.mean(theta))/ tolerance) / np.log(1/mu)
    kmax = np.log(var1 / tolerance) / np.log(1 / mu)
    return var1,theta, theta_history, kmax#k


# 5. 模拟实验（扫描s参数）
def run_sweep_experiment(s_values, num_runs=num_runs):
    """运行参数扫描实验"""
    # 固定图和初始条件
    G = create_random_graph(n, p)
    L = construct_laplacian(G)
    initial_states = np.random.normal(50, 10, size=n)  # N(50,100)[15]

    # 存储结果
    std_devs = []
    settling_times = []

    for s in tqdm(s_values):
        conv_points = []
        times = []

        for _ in range(num_runs):
            var1,final_theta, _, num_iter = dp_consensus(L, initial_states, s)
            conv_points.append(np.mean(final_theta))
            times.append(num_iter)
        # 计算样本标准差和平均收敛时间
        std_devs.append(var1)
        settling_times.append(np.mean(times))

    return std_devs, settling_times


# 6. 生成s值对数尺度范围[16]
s_values = np.logspace(np.log10(0.8), np.log10(1.2), num=50)  # [0.8,1.2]对数间隔[16]
# 7. 运行实验
std_devs, settling_times = run_sweep_experiment(s_values, num_runs=100)  # 这里仅用100次减少计算时间
# 8. 绘制结果图
plt.figure(figsize=(14, 6))
# 图3(a): 收敛点的样本方差[16][19]
plt.subplot(1, 2, 1)
plt.semilogy(s_values, np.array(std_devs) ** 2, 'o-')
plt.xlabel('s (noise-to-state gain)')
plt.ylabel('Empirical variance of θ∞')
plt.title('(a) Variance of convergence point vs s')
plt.grid(True)
# 图3(b): 收敛时间[16][19]
plt.subplot(1, 2, 2)
plt.plot(s_values, settling_times, 'o-')
plt.xlabel('s (noise-to-state gain)')
plt.ylabel('Settling time (iterations)')
plt.title('(b) Settling time vs s')
plt.grid(True)
plt.tight_layout()
plt.show()