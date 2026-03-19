import numpy as np

def phi_function(x, l, delta, k, i):
    """
    Compute phi(x) for a specific segment P_i.

    Parameters:
        x (float or array-like): Input value(s).
        l (float): Minimum value of the real interval.
        delta (float): Quantization interval length.
        k (float): Scaling factor.
        i (int): Segment index.

    Returns:
        float or np.ndarray: The value of phi(x).
    """
    # Calculate the midpoint m_i of segment P_i
    m_i = l + (i + 0.5) * delta

    # Compute the scaling factor s
    # s = 1 / np.tanh(0.5 * k * delta)
    s = delta/2

    # Calculate phi(x)
    phi = s * np.tanh(k * (x - m_i))+m_i

    return phi 

def phi_function_n(x, l, delta, k, i):
    """
    Compute phi(x) for a specific segment P_i.

    Parameters:
        x (float or array-like): Input value(s).
        l (float): Minimum value of the real interval.
        delta (float): Quantization interval length.
        k (float): Scaling factor.
        i (int): Segment index.

    Returns:
        float or np.ndarray: The value of phi(x).
    """
    # Calculate the midpoint m_i of segment P_i
    m_i = l + (i + 0.5) * delta

    # Compute the scaling factor s
    # s = 1 / np.tanh(0.5 * k * delta)
    s = delta/2

    # Calculate phi(x)
    phi = s * np.tanh(k * (x + m_i))-m_i

    return phi 
   
def phi_function_p(x, l, delta, k, i):

    m_i = l + (i + 0.5) * delta
    s = delta/2

    # m_i = torch.tensor(m_i)
    # s = torch.tensor(s)
    # k = torch.tensor(k)

    phi = s * np.tanh(k * (x - m_i)) + m_i

    return phi  

def round_ltq(self, x):

    x_forward = x
    x_backward = x
    step_right = self.start + 0.0
    step_left = self.start + 0.0

    a_pos = torch.where(self.a > self.eps, self.a, self.eps)
    b_pos = torch.where(self.b > self.eps, self.b, self.eps)
    p_1_p = self.start + 0.0
    p_2_p = a_pos[0]
    p_1_n = self.start + 0.0
    p_2_n = b_pos[0]

    for i in range(int((self.n_val-1)/2)):
        # import pdb; pdb.set_trace()  
        step_right += a_pos[i]
        step_left += b_pos[i]
        k_p = sum(a_pos) - a_pos[i]
        k_n = sum(b_pos) - b_pos[i]

        if i == 0:
            thre_forward_p = self.start + a_pos[0]
            thre_forward_n = self.start - b_pos[0]
            x_forward = torch.where(x > thre_forward_p, step_right, x)
            x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
            x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
            l_p = p_1_p            
            delta_p = p_2_p - p_1_p 
            l_n = p_1_n            
            delta_n = p_2_n - p_1_n 
            y_step_p = self.phi_function_p(x, l_p, delta_p, k_p if k_p>1 else 1, 0)
            y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
            x_backward = torch.where(x > self.start, 0.4*x+0.6*y_step_p, x)
            x_backward = torch.where(x < self.start, 0.4*x+0.6*y_step_n, x_backward)
        else:
            thre_forward_p += a_pos[i]
            thre_forward_n -= b_pos[i]
            p_1_p += a_pos[i-1]
            p_2_p = p_1_p + a_pos[i]
            p_1_n += b_pos[i-1]
            p_2_n = p_1_n + b_pos[i]
            l_p = p_1_p            
            delta_p = p_2_p - p_1_p 
            l_n = p_1_n            
            delta_n = p_2_n - p_1_n 
            y_step_p = self.phi_function_p(x, l_p, delta_p, k_p if k_p>1 else 1, 0)
            y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
            x_forward = torch.where(x > thre_forward_p, step_right, x_forward)
            x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
            x_backward = torch.where(x > p_1_p, 0.4*x+0.6*y_step_p, x_backward)
            x_backward = torch.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

    p_1_p += a_pos[i]
    p_1_n += b_pos[i]
    x_backward = torch.where(x > p_1_p, x.max(), x_backward)
    x_backward = torch.where(x < -p_1_n, x.min(), x_backward)
    out = x_forward.detach() + x_backward - x_backward.detach()
    # plot_tensor_histogram(x_backward, name="ltq_out_back_round")
    # plot_tensor_histogram(x_forward, name="ltq_out_for_round")
    # plot_tensor_histogram(x, name="ltq_x_for_round")
    # import pdb; pdb.set_trace()

    return out

def taylor_expansion_2n_minus_1( n):
        """
        计算 2^n - 1 的泰勒展开前若干项。
        
        参数:
        - n: 指数 n 的值。
        - terms: 泰勒展开的项数（默认为 8 项）。
        
        返回:
        - 每一项的值组成的列表。
        """

        terms = 2**(n-1)
        import math
        ln2 = math.log(2)  # 计算 ln(2)
        expansion_terms = []  # 用于存储展开的每一项
        
        for k in range(1,1+terms):  # 从第 1 项开始展开
            term = (ln2 ** k) * (n ** k) / math.factorial(k)
            expansion_terms.append(term/2)
        negative_list = [-x for x in reversed(expansion_terms[0:])]
        expansion_terms = expansion_terms + negative_list
        return expansion_terms
# Example usage
if __name__ == "__main__":
    # # Parameters
    # l = -2            # Minimum value of the interval
    # delta = 4        # Interval length
    # k = 10           # Scaling factor
    # i = 0            # Segment index (for P_0)

    # # Input values
    # x = np.linspace(-2, 2, 500)

    # # Compute phi(x) for segment P_0
    # y = phi_function(x, l, delta, k, i)

    # # Plot the result
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, y, label=f"Segment P_{i}")
    # plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    # plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    # plt.title(f"Phi Function for Segment P_{i}")
    # plt.xlabel("x")
    # plt.ylabel("phi(x)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.savefig("/cluster/home/yujichen/QuantIR/basicsr/archs/noneed")

    tyler = list(reversed(taylor_expansion_2n_minus_1(n=4)))
    # print(tyler)
    # tyler = [0,1,2,4,8,10,12,14]
    # tyler = [0.043304, 4.24950, 0.31546, 0.68267, 1.23112, 1.776131, 1.921812, 1.38629]
    # p_1 = 0
    # p_2 = tyler[0]
    # k = sum(tyler)*2
    # Input values
    x = np.linspace(-1, 1, 5000)
    eps = [1e-4]
    zero = [0]
    # a = [-2.9813e-04, -1.2250e-03, -4.3027e-04,  4.1343e-01,  1.1925e+00,
    #      2.1055e+00,  2.4842e+00]
    # b = [0.2408, 0.3064, 0.4254, 0.8210, 1.4467, 2.0243, 2.1988]  
    # abs:
    # a = [-5.5210e-41, -5.5259e-41, -2.2631e-41,  4.2022e-01,  1.1880e+00,
    #      2.0289e+00,  2.4334e+00] 
    # b = [0.2456, 0.3198, 0.4503, 0.8406, 1.4613, 2.0433, 2.2222] 
    # a = np.abs(a)
    # start = 0
    n_val = 2 ** 4 - 1
    # for j in range(len(tyler)-1):

    #     k=sum(tyler) - tyler[j]
    #     i = 0           
        
    #     if j == 0:
    #         l = p_1            
    #         delta = p_2 - p_1 
    #         y_step = phi_function(x, l, delta, k if k>1 else 1, i)
    #         y_step_n = phi_function_n(x, l, delta, k if k>1 else 1, i)
    #         y = np.where(x>0 , y_step, x)
    #         y = np.where(x<0 , y_step_n, y)
    #     else:
    #         # y = np.where(x>(p_1-tyler[j]) , 0.1*x+0.9*y_step, y)
    #         # y = np.where(x<-(p_1-tyler[j])  , 0.1*x+0.9*y_step_n, y)
    #         p_1 += tyler[j-1]
    #         p_2 = p_1+tyler[j]
    #         l = p_1          
    #         delta = p_2 - p_1 
    #         y_step = phi_function(x, l, delta, k if k>1 else 1, i)
    #         y_step_n = phi_function_n(x, l, delta, k if k>1 else 1, i)
    #         y = np.where(x>p_1, y_step, y)
    #         y = np.where(x<-p_1, y_step_n, y)
    # x_forward = x
    # x_backward = x
    # step_right = start + 0.0
    # step_left = start + 0.0
    # sorted_list = sorted(tyler)
    # a_pos = [-8,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,8]
    additive_pot = [0.0000, 0.3333, 0.6667, 0.0833, 0.0208, 1.0000, 0.7500, 0.6875, 0.1667,
        0.5000, 0.2500, 0.1875, 0.0417, 0.3750, 0.1250, 0.0625]
    negative_list = [-additive_pot for additive_pot in reversed(additive_pot[1:])]
    additive_pot = additive_pot + negative_list
    additive_pot = sorted(additive_pot)
    # a_pos = [-7.5590, -4.7060, -2.8803, -1.7626, -1.0350, -0.5172, -0.2079,  0.0992,
    #      0.4543,  0.9385,  1.7517,  3.0348,  4.8975,  7.6349]
    # 2. 分离负值和正值
    # b = list(reversed([-x for x in sorted_list if x < 0]))  # 负值
    # a = [x for x in sorted_list if x >= 0]  # 正值（包含0） 
    # print(b)   
    print(additive_pot) 
    # a_pos = np.where(a > eps, a, eps)
    # b_pos = np.where(b > eps, b, eps)
    # p_1_p = start + 0.0
    # p_2_p = a_pos[0]
    # p_1_n = start + 0.0
    # p_2_n = b_pos[0]

    # for i in range(int((n_val-1)/2)):
    #     # import pdb; pdb.set_trace()  
    #     step_right += a_pos[i]
    #     step_left += b_pos[i]
    #     k_p = sum(a_pos) - a_pos[i]
    #     k_n = sum(b_pos) - b_pos[i]

    #     if i == 0:
    #         thre_forward_p = start + a_pos[0]
    #         thre_forward_n = start - b_pos[0]
    #         x_forward = np.where(x > thre_forward_p, step_right, x)
    #         x_forward = np.where(x < thre_forward_p, zero, x_forward)
    #         x_forward = np.where(x < thre_forward_n, -step_left, x_forward)
    #         l_p = p_1_p            
    #         delta_p = p_2_p - p_1_p 
    #         l_n = p_1_n            
    #         delta_n = p_2_n - p_1_n 
    #         y_step_p = phi_function_p(x, l_p, delta_p, k_p if k_p>1 else 1, 0)
    #         y_step_n = phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
    #         x_backward = np.where(x > start, 0.4*x+0.6*y_step_p, x)
    #         x_backward = np.where(x < start, 0.4*x+0.6*y_step_n, x_backward)
    #     else:
    #         thre_forward_p += a_pos[i]
    #         thre_forward_n -= b_pos[i]
    #         p_1_p += a_pos[i-1]
    #         p_2_p = p_1_p + a_pos[i]
    #         p_1_n += b_pos[i-1]
    #         p_2_n = p_1_n + b_pos[i]
    #         l_p = p_1_p            
    #         delta_p = p_2_p - p_1_p 
    #         l_n = p_1_n            
    #         delta_n = p_2_n - p_1_n 
    #         y_step_p = phi_function_p(x, l_p, delta_p, k_p if k_p>1 else 1, 0)
    #         y_step_n = phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
    #         x_forward = np.where(x > thre_forward_p, step_right, x_forward)
    #         x_forward = np.where(x < thre_forward_n, -step_left, x_forward)
    #         x_backward = np.where(x > p_1_p, 0.4*x+0.6*y_step_p, x_backward)
    #         x_backward = np.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

    # p_1_p += a_pos[i]
    # p_1_n += b_pos[i]
    # x_backward = np.where(x > p_1_p, x, x_backward)
    # x_backward = np.where(x < -p_1_n, x, x_backward)
    # out = x_forward.detach() + x_backward - x_backward.detach()    
    # [-1.0, -0.75, -0.6875, -0.6667, -0.5, -0.375, -0.3333, -0.25, -0.1875, -0.1667, 
    # -0.125, -0.0833, -0.0625, -0.0417, -0.0208, 0.0, 0.0208, 0.0417, 0.0625, 0.0833, 
    # 0.125, 0.1667, 0.1875, 0.25, 0.3333, 0.375, 0.5, 0.6667, 0.6875, 0.75, 1.0]
    a_pos = [-1.0, -0.75, -0.6875, -0.6667, -0.5, -0.375, -0.3333, -0.25, -0.1875, 
    -0.1667, -0.125, -0.0833, -0.0625, -0.0417, -0.0208, 0.0, 0.0208, 0.0417, 
    0.0625, 0.0833, 0.125, 0.1667, 0.1875, 0.25, 0.3333, 0.375, 0.4, 0.6667, 0.6875, 
    0.75, 1.0]
    a_pos = a_pos*4
    additive_pot = a_pos
    start = additive_pot[0]
    x_forward = x
    x_backward = x
    # step_right = a_pos[0]  
    # step_left = self.start + 0.0
    # self.a.data = self.a.abs()
    # self.b.data = self.b.abs()
    # a_pos = torch.where(self.a > self.eps, self.a, self.eps)
    p_1_p = additive_pot[0]
    p_2_p = additive_pot[1] 
    for i in range(len(a_pos)):
        # import pdb; pdb.set_trace() 
        
        # step_right += a_pos[i]
        # step_left += b_pos[i]
        k_p = 100 - (additive_pot[i]-additive_pot[i-1])
        # k_n = sum(b_pos) - b_pos[i]

        if i == 0:
            thre_forward_p = start
            # thre_forward_n = self.start - b_pos[0]
            x_forward = np.where(x > thre_forward_p,additive_pot[i], additive_pot[i])
            # x_forward = torch.where(x < thre_forward_p, self.zero, x_forward)
            # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
            l_p = p_1_p            
            delta_p = p_2_p - p_1_p 
            # l_n = p_1_n            
            # delta_n = p_2_n - p_1_n 
            y_step_p = phi_function_p(x, a_pos[0], delta_p, k_p if k_p>1 else 1, 0)
            # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
            x_backward = np.where(x > start, 0.4*x+0.6*y_step_p, x.min())
            # x_backward = torch.where(x < self.start, 0.4*x+0.6*y_step_n, x_backward)
        else:
            thre_forward_p = (a_pos[i-1] + a_pos[i])/2
            thre_backward = a_pos[i-1]
            # thre_forward_n -= b_pos[i]
            p_1_p = a_pos[i-1]
            p_2_p = a_pos[i]
            # p_1_n += b_pos[i-1]
            # p_2_n = p_1_n + b_pos[i]
            l_p = p_1_p            
            delta_p = additive_pot[i] - additive_pot[i-1] 
            # l_n = p_1_n            
            # delta_n = p_2_n - p_1_n
            y_step_p = phi_function_p(x, a_pos[i-1], delta_p, k_p if k_p>1 else 1, 0)
            # y_step_n = self.phi_function_n(x, l_n, delta_n, k_n if k_n>1 else 1, 0)
            x_forward = np.where(x > thre_forward_p, additive_pot[i], x_forward)
            # x_forward = torch.where(x < thre_forward_n, -step_left, x_forward)
            x_backward = np.where(x > thre_backward, 0.4*x+0.6*y_step_p, x_backward)
            # x_backward = torch.where(x < -p_1_n, 0.4*x+0.6*y_step_n, x_backward)

        p_1_p = additive_pot[i]
        # p_1_n += b_pos[i]
        # x_backward = torch.where(x > p_1_p, x.max(), x_backward)
        # x_backward = torch.where(x < -p_1_n, x.min(), x_backward)
        x_backward = np.where(x > p_1_p, x.max(), x_backward)
        # x_backward = torch.where(x < -p_1_n, x, x_backward)
        # out = x_forward.detach() + x_backward - x_backward.detach()

    # Plot the result
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(x, x_forward, label=f"Segment P_{i}")
    plt.plot(x, x_backward, label=f"Segment P_{i}")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(f"Phi Function for Segment P_{i}")
    plt.xlabel("x")
    plt.ylabel("phi(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("/cluster/home/yujichen/QuantIR/basicsr/archs/noneed/ltq_yj_total_out")