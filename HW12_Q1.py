import numpy as np

def solve_2d_heat_equation():
    """
    求解二維熱傳導方程：
    ∂²u/∂x² + ∂²u/∂y² = xy, 0 < x < π, 0 < y < π/2
    u(0,y) = cos y, u(π,y) = -cos y, 0 ≤ y ≤ π/2
    u(x,0) = cos x, u(x,π/2) = 0, 1 ≤ y ≤ 2
    使用 h = k = 0.1π
    """
    
    # 參數設置
    h = k = 0.1 * np.pi  # 網格間距
    x_max, y_max = np.pi, np.pi/2
    
    # 創建網格
    x = np.arange(0, x_max + h, h)
    y = np.arange(0, y_max + k, k)
    nx, ny = len(x), len(y)
    
    print(f"網格大小: {nx} x {ny}")
    print(f"x範圍: [0, {x_max:.4f}], 步長: {h:.4f}")
    print(f"y範圍: [0, {y_max:.4f}], 步長: {k:.4f}")
    
    u = np.zeros((nx, ny))
    
    # 設置邊界條件
    # u(0,y) = cos y
    u[0, :] = np.cos(y)
    # u(π,y) = -cos y
    u[-1, :] = -np.cos(y)
    # u(x,0) = cos x
    u[:, 0] = np.cos(x)
    # u(x,π/2) = 0
    u[:, -1] = 0
    
    # 使用有限差分法求解
    # ∂²u/∂x² + ∂²u/∂y² = xy
    # (u[i+1,j] - 2u[i,j] + u[i-1,j])/h² + (u[i,j+1] - 2u[i,j] + u[i,j-1])/k² = x[i]*y[j]
    
    max_iterations = 10000
    tolerance = 1e-6
    
    for iteration in range(max_iterations):
        u_old = u.copy()
        
        # 更新內部點
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # 源項
                source = x[i] * y[j]
                
                # 五點差分格式
                u[i, j] = (h**2 * k**2 * source + 
                          k**2 * (u[i+1, j] + u[i-1, j]) + 
                          h**2 * (u[i, j+1] + u[i, j-1])) / (2 * (h**2 + k**2))
        
        # 檢查收斂性
        if np.max(np.abs(u - u_old)) < tolerance:
            print(f"在第 {iteration+1} 次迭代後收斂")
            break
    
    return x, y, u



def main():
    print("=" * 50)
    
    # 求解方程
    x, y, u = solve_2d_heat_equation()
    
    # 輸出一些關鍵點的值
    print("\n關鍵點的解值:")
    print(f"u(π/2, π/4) = {u[len(x)//2, len(y)//2]:.6f}")
    print(f"u(π/4, π/8) = {u[len(x)//4, len(y)//4]:.6f}")
    
    # 驗證邊界條件
    print("\n邊界條件驗證:")
    print(f"u(0, π/4) = {u[0, len(y)//2]:.6f}, 理論值 cos(π/4) = {np.cos(np.pi/4):.6f}")
    print(f"u(π, π/4) = {u[-1, len(y)//2]:.6f}, 理論值 -cos(π/4) = {-np.cos(np.pi/4):.6f}")
    print(f"u(π/2, 0) = {u[len(x)//2, 0]:.6f}, 理論值 cos(π/2) = {np.cos(np.pi/2):.6f}")
    print(f"u(π/2, π/2) = {u[len(x)//2, -1]:.6f}, 理論值 0 = 0")

if __name__ == "__main__":
    main()