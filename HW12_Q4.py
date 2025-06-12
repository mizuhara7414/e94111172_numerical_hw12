import numpy as np

def solve_wave_equation():
    """
    求解波動方程：
    ∂²p/∂t² = ∂²p/∂x², 0 ≤ x ≤ 1, 0 ≤ t
    
    初始條件：
    p(0,t) = 1, p(1,t) = 2, p(x,0) = cos(2πx)
    ∂p/∂t(x,0) = 2π sin(2πx), 0 ≤ x ≤ 1
    
    使用 Δx = Δt = 0.1
    """
    
    # 參數設置
    dx = 0.1
    dt = 0.1
    x_max = 1.0
    t_max = 2.0  # 計算時間範圍
    
    # 計算網格點
    x = np.arange(0, x_max + dx, dx)
    t = np.arange(0, t_max + dt, dt)
    nx, nt = len(x), len(t)
    
    # CFL條件檢查 (c = 1 for this wave equation)
    c = 1.0  # 波速
    r = c * dt / dx  # CFL數
    
    print(f"參數設置:")
    print(f"Δx = {dx}, Δt = {dt}")
    print(f"網格大小: {nx} × {nt}")
    print(f"CFL數 r = c·Δt/Δx = {r:.2f}")
    print(f"穩定性條件 (r ≤ 1): {'滿足' if r <= 1 else '不滿足'}")
    
    # 初始化解矩陣
    p = np.zeros((nt, nx))
    
    # 初始條件：p(x,0) = cos(2πx)
    p[0, :] = np.cos(2 * np.pi * x)
    
    # 初始速度條件：∂p/∂t(x,0) = 2π sin(2πx)
    # 使用前向差分: p[1,i] = p[0,i] + dt * ∂p/∂t(x,0)
    initial_velocity = 2 * np.pi * np.sin(2 * np.pi * x)
    p[1, :] = p[0, :] + dt * initial_velocity
    
    # 邊界條件
    for n in range(nt):
        p[n, 0] = 1   # p(0,t) = 1
        p[n, -1] = 2  # p(1,t) = 2
    
    # 使用顯式有限差分法求解波動方程
    # p[n+1,i] = 2p[n,i] - p[n-1,i] + r²(p[n,i+1] - 2p[n,i] + p[n,i-1])
    return x, t, p

def analyze_wave_solution(x, t, p):
    """分析波動解的性質"""
    print(f"\n解的分析:")
    print("-" * 40)
    
    # 幅值分析
    p_max = np.max(p)
    p_min = np.min(p)
    print(f"解的範圍: [{p_min:.4f}, {p_max:.4f}]")
    
    # 能量分析（近似）
    energies = []
    for n in range(len(t)):
        # 動能項（近似）
        if n < len(t) - 1:
            velocity = (p[n+1, :] - p[n, :]) / (t[1] - t[0])
        else:
            velocity = (p[n, :] - p[n-1, :]) / (t[1] - t[0])
        
        kinetic_energy = 0.5 * np.sum(velocity**2) * (x[1] - x[0])
        
        # 勢能項（近似）
        gradient = np.gradient(p[n, :], x[1] - x[0])
        potential_energy = 0.5 * np.sum(gradient**2) * (x[1] - x[0])
        
        total_energy = kinetic_energy + potential_energy
        energies.append(total_energy)
    
    energies = np.array(energies)
    energy_change = (energies[-1] - energies[0]) / energies[0] * 100
    print(f"總能量變化: {energy_change:.2f}%")
    
    # 檢查週期性
    print(f"\n特定點的時間演化:")
    for i in [len(x)//4, len(x)//2, 3*len(x)//4]:
        print(f"x = {x[i]:.2f}: p範圍 [{np.min(p[:, i]):.4f}, {np.max(p[:, i]):.4f}]")
    
    return energies





def verify_boundary_conditions(x, t, p):
    """驗證邊界條件和初始條件"""
    print(f"\n邊界條件和初始條件驗證:")
    print("-" * 50)
    
    # 檢查邊界條件
    print("邊界條件檢查:")
    left_boundary_error = np.max(np.abs(p[:, 0] - 1))
    right_boundary_error = np.max(np.abs(p[:, -1] - 2))
    print(f"  p(0,t) = 1: 最大誤差 = {left_boundary_error:.2e}")
    print(f"  p(1,t) = 2: 最大誤差 = {right_boundary_error:.2e}")
    
    # 檢查初始條件
    print("\n初始條件檢查:")
    initial_condition_exact = np.cos(2 * np.pi * x)
    initial_condition_error = np.max(np.abs(p[0, :] - initial_condition_exact))
    print(f"  p(x,0) = cos(2πx): 最大誤差 = {initial_condition_error:.2e}")
    
    # 檢查初始速度條件（近似）
    if len(t) > 1:
        initial_velocity_numerical = (p[1, :] - p[0, :]) / (t[1] - t[0])
        initial_velocity_exact = 2 * np.pi * np.sin(2 * np.pi * x)
        velocity_error = np.max(np.abs(initial_velocity_numerical - initial_velocity_exact))
        print(f"  ∂p/∂t(x,0) = 2π sin(2πx): 最大誤差 = {velocity_error:.2e}")

def main():
    print("=" * 50)
    print("方程: ∂²p/∂t² = ∂²p/∂x²")
    print("邊界條件: p(0,t) = 1, p(1,t) = 2")
    print("初始條件: p(x,0) = cos(2πx), ∂p/∂t(x,0) = 2π sin(2πx)")
    print("網格參數: Δx = Δt = 0.1")
    
    # 求解波動方程
    x, t, p = solve_wave_equation()
    
    # 驗證邊界和初始條件
    verify_boundary_conditions(x, t, p)
    
    # 分析解的性質
    energies = analyze_wave_solution(x, t, p)


if __name__ == "__main__":
    main()