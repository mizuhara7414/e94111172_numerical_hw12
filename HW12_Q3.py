import numpy as np


def solve_cylindrical_heat_equation():
    """
    求解圓柱坐標系熱傳導方程：
    ∂²T/∂r² + (1/r)∂T/∂r + (1/r²)∂²T/∂θ² = 0, 1/2 ≤ r ≤ 1, 0 ≤ t ≤ π/3
    
    邊界條件：
    T(r,0) = 0, T(r,π/3) = 0
    T(1/2,θ) = 50, T(1,θ) = 100
    """
    
    # 參數設置
    r_min, r_max = 0.5, 1.0
    theta_min, theta_max = 0, np.pi/3
    
    # 網格設置
    nr, ntheta = 21, 16  # 網格點數
    dr = (r_max - r_min) / (nr - 1)
    dtheta = (theta_max - theta_min) / (ntheta - 1)
    
    print(f"網格設置:")
    print(f"r 範圍: [{r_min}, {r_max}], 網格點數: {nr}, Δr = {dr:.4f}")
    print(f"θ 範圍: [0, π/3], 網格點數: {ntheta}, Δθ = {dtheta:.4f}")
    
    # 創建網格
    r = np.linspace(r_min, r_max, nr)
    theta = np.linspace(theta_min, theta_max, ntheta)
    
    # 初始化溫度矩陣
    T = np.zeros((nr, ntheta))
    
    # 設置邊界條件
    # T(r,0) = 0
    T[:, 0] = 0
    # T(r,π/3) = 0
    T[:, -1] = 0
    # T(1/2,θ) = 50
    T[0, :] = 50
    # T(1,θ) = 100
    T[-1, :] = 100
    
    # 保持邊界條件
    T[0, 0] = T[0, -1] = 50  # 角點處理
    T[-1, 0] = T[-1, -1] = 100
    
    # 使用有限差分法求解拉普拉斯方程
    # ∂²T/∂r² + (1/r)∂T/∂r + (1/r²)∂²T/∂θ² = 0
    
    max_iterations = 5000
    tolerance = 1e-6
    omega = 1.5  # SOR方法的鬆弛因子
    
    print(f"\n開始迭代求解...")
    
    for iteration in range(max_iterations):
        T_old = T.copy()
        max_change = 0
        
        # 更新內部點（使用SOR方法）
        for i in range(1, nr-1):
            for j in range(1, ntheta-1):
                # 有限差分近似
                # ∂²T/∂r² ≈ (T[i+1,j] - 2T[i,j] + T[i-1,j]) / dr²
                # ∂T/∂r ≈ (T[i+1,j] - T[i-1,j]) / (2*dr)
                # ∂²T/∂θ² ≈ (T[i,j+1] - 2T[i,j] + T[i,j-1]) / dtheta²
                
                r_i = r[i]
                
                # 係數計算
                coeff_r2 = 1.0 / (dr**2)
                coeff_r1 = 1.0 / (r_i * 2 * dr)
                coeff_theta2 = 1.0 / (r_i**2 * dtheta**2)
                
                # 新值計算
                T_new = ((coeff_r2 + coeff_r1) * T[i+1, j] + 
                        (coeff_r2 - coeff_r1) * T[i-1, j] + 
                        coeff_theta2 * (T[i, j+1] + T[i, j-1])) / \
                       (2 * coeff_r2 + 2 * coeff_theta2)
                
                # SOR更新
                change = omega * (T_new - T[i, j])
                T[i, j] += change
                max_change = max(max_change, abs(change))
        
        # 檢查收斂性
        if max_change < tolerance:
            print(f"在第 {iteration+1} 次迭代後收斂，最大變化: {max_change:.2e}")
            break
        
        if (iteration + 1) % 500 == 0:
            print(f"迭代 {iteration+1}, 最大變化: {max_change:.2e}")
    
    return r, theta, T



def verify_boundary_conditions(r, theta, T):
    """驗證邊界條件"""
    print("\n邊界條件驗證:")
    print("-" * 40)
    
    # T(r,0) = 0
    print(f"T(r,0) 邊界 (應為0):")
    for i in range(0, len(r), max(1, len(r)//5)):
        print(f"  T({r[i]:.2f}, 0) = {T[i, 0]:.6f}")
    
    # T(r,π/3) = 0
    print(f"\nT(r,π/3) 邊界 (應為0):")
    for i in range(0, len(r), max(1, len(r)//5)):
        print(f"  T({r[i]:.2f}, π/3) = {T[i, -1]:.6f}")
    
    # T(1/2,θ) = 50
    print(f"\nT(0.5,θ) 邊界 (應為50):")
    for j in range(0, len(theta), max(1, len(theta)//5)):
        angle_deg = theta[j] * 180 / np.pi
        print(f"  T(0.5, {angle_deg:.1f}°) = {T[0, j]:.6f}")
    
    # T(1,θ) = 100
    print(f"\nT(1.0,θ) 邊界 (應為100):")
    for j in range(0, len(theta), max(1, len(theta)//5)):
        angle_deg = theta[j] * 180 / np.pi
        print(f"  T(1.0, {angle_deg:.1f}°) = {T[-1, j]:.6f}")

def analyze_solution(r, theta, T):
    """分析解的性質"""
    print("\n解的分析:")
    print("-" * 40)
    
    # 找到最高和最低溫度
    T_max = np.max(T)
    T_min = np.min(T)
    max_idx = np.unravel_index(np.argmax(T), T.shape)
    min_idx = np.unravel_index(np.argmin(T), T.shape)
    
    print(f"最高溫度: {T_max:.4f} at (r={r[max_idx[0]]:.3f}, θ={theta[max_idx[1]]*180/np.pi:.1f}°)")
    print(f"最低溫度: {T_min:.4f} at (r={r[min_idx[0]]:.3f}, θ={theta[min_idx[1]]*180/np.pi:.1f}°)")
    
    # 中心點溫度
    center_r = len(r) // 2
    center_theta = len(theta) // 2
    print(f"中心點溫度: T({r[center_r]:.3f}, {theta[center_theta]*180/np.pi:.1f}°) = {T[center_r, center_theta]:.4f}")
    
    # 沿徑向的平均溫度
    print(f"\n沿徑向的平均溫度:")
    for i in range(0, len(r), max(1, len(r)//5)):
        avg_temp = np.mean(T[i, :])
        print(f"  r = {r[i]:.3f}: 平均溫度 = {avg_temp:.4f}")
    
    # 沿角向的平均溫度
    print(f"\n沿角向的平均溫度:")
    for j in range(0, len(theta), max(1, len(theta)//5)):
        avg_temp = np.mean(T[:, j])
        angle_deg = theta[j] * 180 / np.pi
        print(f"  θ = {angle_deg:.1f}°: 平均溫度 = {avg_temp:.4f}")

def calculate_heat_flux(r, theta, T):
    """計算熱流密度"""
    print("\n熱流密度分析:")
    print("-" * 40)
    
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    
    # 計算溫度梯度
    dT_dr = np.zeros_like(T)
    dT_dtheta = np.zeros_like(T)
    
    # 徑向梯度 (中心差分)
    dT_dr[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2 * dr)
    dT_dr[0, :] = (T[1, :] - T[0, :]) / dr  # 前向差分
    dT_dr[-1, :] = (T[-1, :] - T[-2, :]) / dr  # 後向差分
    
    # 角向梯度 (中心差分)
    dT_dtheta[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dtheta)
    dT_dtheta[:, 0] = (T[:, 1] - T[:, 0]) / dtheta  # 前向差分
    dT_dtheta[:, -1] = (T[:, -1] - T[:, -2]) / dtheta  # 後向差分
    
    # 轉換為每個半徑處的角向梯度
    for i in range(len(r)):
        dT_dtheta[i, :] /= r[i]
    
    # 計算熱流密度大小
    heat_flux = np.sqrt(dT_dr**2 + dT_dtheta**2)
    
    # 輸出邊界處的熱流密度
    print("內邊界 (r = 0.5) 熱流密度:")
    for j in range(0, len(theta), max(1, len(theta)//3)):
        angle_deg = theta[j] * 180 / np.pi
        flux = heat_flux[0, j]
        print(f"  θ = {angle_deg:.1f}°: |∇T| = {flux:.4f}")
    
    print("\n外邊界 (r = 1.0) 熱流密度:")
    for j in range(0, len(theta), max(1, len(theta)//3)):
        angle_deg = theta[j] * 180 / np.pi
        flux = heat_flux[-1, j]
        print(f"  θ = {angle_deg:.1f}°: |∇T| = {flux:.4f}")
    
    return heat_flux

def main():
    print("=" * 50)
    print("方程: ∂²T/∂r² + (1/r)∂T/∂r + (1/r²)∂²T/∂θ² = 0")
    print("邊界條件:")
    print("  T(r,0) = 0, T(r,π/3) = 0")
    print("  T(1/2,θ) = 50, T(1,θ) = 100")
    
    # 求解方程
    r, theta, T = solve_cylindrical_heat_equation()
  
    # 驗證邊界條件
   ## verify_boundary_conditions(r, theta, T)

    # 分析解的性質
    analyze_solution(r, theta, T)

"""    
    # 計算熱流密度
    heat_flux = calculate_heat_flux(r, theta, T)
    

    
    # 額外的數值驗證
    print(f"\n數值解驗證:")
    print(f"溫度範圍: [{np.min(T):.4f}, {np.max(T):.4f}]")
    print(f"符合物理期望 (0 ≤ T ≤ 100): {'是' if 0 <= np.min(T) and np.max(T) <= 100 else '否'}")
   
    # 檢查拉普拉斯方程的滿足程度
    print(f"\n拉普拉斯方程殘差檢查:")
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    
    max_residual = 0
    for i in range(1, len(r)-1):
        for j in range(1, len(theta)-1):
            # 計算拉普拉斯算子
            laplacian = ((T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dr**2 + 
                        (T[i+1, j] - T[i-1, j]) / (r[i] * 2 * dr) + 
                        (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / (r[i]**2 * dtheta**2))
            max_residual = max(max_residual, abs(laplacian))
    
    print(f"最大拉普拉斯殘差: {max_residual:.2e}")
    print(f"數值解質量: {'良好' if max_residual < 1e-2 else '需要改進'}")
"""
if __name__ == "__main__":
    main()