import numpy as np
import matplotlib.pyplot as plt

def forward_difference_method(r, T0, boundary_condition, t_max, dt, dx):
    """前向差分法"""
    x = np.arange(0, 1 + dx, dx)
    t = np.arange(0, t_max + dt, dt)
    nx, nt = len(x), len(t)
    
    # 初始化溫度矩陣
    T = np.zeros((nt, nx))
    
    # 初始條件
    T[0, :] = T0(x)
    
    # 邊界條件 ∂T/∂r + 3T = 0 at r = 1/2
    # 這裡假設 r = 1/2 對應 x = 1
    
    # 時間步進
    for n in range(nt-1):
        for i in range(1, nx-1):
            T[n+1, i] = T[n, i] + r * (T[n, i+1] - 2*T[n, i] + T[n, i-1])
        
        # 邊界條件處理
        T[n+1, 0] = T[n+1, 1]  # 左邊界（對稱條件或其他）
        
        # 右邊界：∂T/∂r + 3T = 0 at r = 1/2
        # 使用後向差分：(T[nx-1] - T[nx-2])/dx + 3*T[nx-1] = 0
        T[n+1, -1] = T[n+1, -2] / (1 + 3*dx)
    
    return x, t, T

def backward_difference_method(r, T0, boundary_condition, t_max, dt, dx):
    """後向差分法"""
    x = np.arange(0, 1 + dx, dx)
    t = np.arange(0, t_max + dt, dt)
    nx, nt = len(x), len(t)
    
    # 初始化溫度矩陣
    T = np.zeros((nt, nx))
    T[0, :] = T0(x)
    
    # 構建係數矩陣 A
    A = np.zeros((nx, nx))
    
    # 內部點
    for i in range(1, nx-1):
        A[i, i-1] = -r
        A[i, i] = 1 + 2*r
        A[i, i+1] = -r
    
    # 邊界條件
    A[0, 0] = 1
    A[0, 1] = -1  # 左邊界
    
    # 右邊界：∂T/∂r + 3T = 0
    A[-1, -2] = -1/dx
    A[-1, -1] = 1/dx + 3
    
    # 時間步進
    for n in range(nt-1):
        b = T[n, :].copy()
        b[0] = 0  # 左邊界
        b[-1] = 0  # 右邊界
        
        T[n+1, :] = np.linalg.solve(A, b)
    
    return x, t, T

def crank_nicolson_method(r, T0, boundary_condition, t_max, dt, dx):
    """Crank-Nicolson方法"""
    x = np.arange(0, 1 + dx, dx)
    t = np.arange(0, t_max + dt, dt)
    nx, nt = len(x), len(t)
    
    # 初始化溫度矩陣
    T = np.zeros((nt, nx))
    T[0, :] = T0(x)
    
    # 構建係數矩陣
    alpha = r / 2
    
    # 左側矩陣 A (隱式項)
    A = np.zeros((nx, nx))
    for i in range(1, nx-1):
        A[i, i-1] = -alpha
        A[i, i] = 1 + 2*alpha
        A[i, i+1] = -alpha
    
    # 邊界條件
    A[0, 0] = 1
    A[0, 1] = -1
    A[-1, -2] = -1/dx
    A[-1, -1] = 1/dx + 3
    
    # 右側矩陣 B (顯式項)
    B = np.zeros((nx, nx))
    for i in range(1, nx-1):
        B[i, i-1] = alpha
        B[i, i] = 1 - 2*alpha
        B[i, i+1] = alpha
    
    B[0, 0] = 1
    B[0, 1] = -1
    B[-1, -2] = 1/dx
    B[-1, -1] = -1/dx - 3
    
    # 時間步進
    for n in range(nt-1):
        b = np.dot(B, T[n, :])
        b[0] = 0  # 左邊界
        b[-1] = 0  # 右邊界
        
        T[n+1, :] = np.linalg.solve(A, b)
    
    return x, t, T

def solve_heat_equation_1d():
    """
    求解一維熱傳導方程：
    ∂²T/∂r² + (1/r)∂T/∂r = (1/4K)∂T/∂t, 1/2 ≤ r ≤ 1, 0 ≤ t
    T(r,0) = 100 + 40r, 0 ≤ r ≤ 10; ∂T/∂r + 3T = 0 at r = 1/2
    T(r,0) = 200(r - 0.5), 0.5 ≤ r ≤ 1
    使用 Δt = 0.5, Δr = 0.1, K = 0.1
    """
    
    # 參數設置
    dt = 0.5
    dr = 0.1
    K = 0.1
    t_max = 5.0  # 計算到 t = 5
    
    # 計算 r 值（對於數值計算，我們將 r 範圍映射到 [0,1]）
    r_coeff = 1 / (4 * K)  # = 2.5
    r = r_coeff * dt / (dr**2)
    
    print(f"參數設置:")
    print(f"Δt = {dt}, Δr = {dr}, K = {K}")
    print(f"r = {r:.4f}")
    print(f"穩定性條件 (r ≤ 0.5): {'滿足' if r <= 0.5 else '不滿足'}")
    
    # 初始條件函數
    def T0(x):
        # 將 x ∈ [0,1] 映射到實際的 r 範圍
        r_actual = 0.5 + 0.5 * x  # r ∈ [0.5, 1]
        T = np.zeros_like(x)
        
        for i, r_val in enumerate(r_actual):
            if r_val <= 1.0:  # 在題目給定範圍內
                if r_val <= 0.6:  # 對應第一個初始條件區域
                    T[i] = 100 + 40 * r_val
                else:  # 對應第二個初始條件區域
                    T[i] = 200 * (r_val - 0.5)
        return T
    
    # 邊界條件（這裡簡化處理）
    boundary_condition = lambda t: 0
    
    print("\n求解方法:")
    print("1. 前向差分法")
    x1, t1, T1 = forward_difference_method(r, T0, boundary_condition, t_max, dt, dr)
    
    print("2. 後向差分法")
    x2, t2, T2 = backward_difference_method(r, T0, boundary_condition, t_max, dt, dr)
    
    print("3. Crank-Nicolson方法")
    x3, t3, T3 = crank_nicolson_method(r, T0, boundary_condition, t_max, dt, dr)
    
    return (x1, t1, T1), (x2, t2, T2), (x3, t3, T3)


def main():
    print("問題2：求解一維熱傳導方程（三種方法比較）")
    print("=" * 60)
    
    # 求解方程
    results = solve_heat_equation_1d()
    
    # 輸出數值結果
    (x1, t1, T1), (x2, t2, T2), (x3, t3, T3) = results
    
    print(f"\n計算完成！")
    print(f"網格點數: {len(x1)} (空間) × {len(t1)} (時間)")
    print(f"時間範圍: [0, {t1[-1]}]")
    
    print(f"\n最終時刻 t = {t1[-1]} 的溫度分布:")
    print("位置\t前向差分\t後向差分\tCrank-Nicolson")
    for i in range(0, len(x1), max(1, len(x1)//5)):
        print(f"{x1[i]:.2f}\t{T1[-1, i]:.4f}\t\t{T2[-1, i]:.4f}\t\t{T3[-1, i]:.4f}")
    

if __name__ == "__main__":
    main()