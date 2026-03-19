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


if __name__ == "__main__":
    # Parameters
    l = -2            # Minimum value of the interval
    delta = 3        # Interval length
    k = 10           # Scaling factor
    i = 0            # Segment index (for P_0)

    # Input values
    x = np.linspace(-1.2, 1.2, 500)

    # Compute phi(x) for segment P_0
    y = phi_function(x, l, delta, k, i)
    
    # additive_pot = [-4,-3,-1,0.0, 3.0, 3.5, 4.0]
    # a_pos = [-4,-3,-1,0.0, 1, 3.0, 4.0]
    a_pos = [-1.0, -0.75, -0.6875, -0.6666666865348816, -0.5, -0.375,
     -0.3333333432674408, -0.25, -0.1875, -0.1666666716337204, -0.125,
      -0.0833333358168602, -0.0625, -0.0416666679084301, -0.02083333395421505, 
      0.0, 0.02083333395421505, 0.0416666679084301, 0.0625, 0.0833333358168602, 
      0.125, 0.1666666716337204, 0.1875, 0.25, 0.3333333432674408, 0.375, 0.5,
       0.6666666865348816, 0.6875, 0.75, 1.0]
    additive_pot = [-1.0, -0.75, -0.6875, -0.6666666865348816, -0.5, -0.375,
     -0.3333333432674408, -0.25, -0.1875, -0.1666666716337204, -0.125,
      -0.0833333358168602, -0.0625, -0.0416666679084301, -0.02083333395421505, 
      0.0, 0.02083333395421505, 0.0416666679084301, 0.0625, 0.0833333358168602, 
      0.125, 0.1666666716337204, 0.1875, 0.25, 0.3333333432674408, 0.375, 0.5,
       0.6666666865348816, 0.6875, 0.75, 1.0]
    print(a_pos)
    print(additive_pot)
    start = additive_pot[0]
    x_forward = x
    x_backward = x
    # thre_forward = []
    # # thre_backward = []
    # delta = []
    p_1_p = additive_pot[0]
    p_2_p = additive_pot[1] 
    for i in range(len(a_pos)):
        # import pdb; pdb.set_trace() 
        
        # step_right += a_pos[i]
        # step_left += b_pos[i]
        k_p = 100 - (a_pos[i]-a_pos[i-1])
        # k_p = 20 - (additive_pot[i]-additive_pot[i-1])
        # k_n = sum(b_pos) - b_pos[i]

        if i == 0:
            thre_forward_p = start
            # thre_forward.append(thre_forward_p)
            thre_backward_p = start
            # thre_backward.append(thre_backward_p)
            x_forward = np.where(x > thre_forward_p,additive_pot[i], additive_pot[i])
            l_p = p_1_p            
            delta_p = additive_pot[1]-additive_pot[0] #(p_2_p - p_1_p)*2
            # delta.append(delta_p)
            y_step_0 = phi_function_p(x+ delta_p/2-(a_pos[0] + a_pos[1])/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i] #+ (delta_p/2-thre_forward_p*2)
            x_backward = np.where(x > thre_backward_p, y_step_0, 0.1*x+additive_pot[i]-0.1*a_pos[i])
            # x_backward = torch.where(x < self.start, 0.4*x+0.6*y_step_n, x_backward)
        else:
            thre_forward_p = (a_pos[i-1] + a_pos[i])/2
            thre_backward_p = a_pos[i]
            # thre_forward.append(thre_forward_p)
            # thre_backward.append(thre_backward_p)
            p_1_p = a_pos[i-1]
            p_2_p = a_pos[i]
            l_p = p_1_p     
            if i!=len(a_pos)-1:
                delta_p = (additive_pot[i+1] - additive_pot[i])
                y_step_p = phi_function_p(x-(a_pos[i] + a_pos[i+1])/2+delta_p/2, 0, delta_p, k_p if k_p>1 else 1, 0)+additive_pot[i]
            else:
                delta_p = 0
                y_step_p = phi_function_p(x, a_pos[i-1], delta_p, k_p if k_p>1 else 1, 0)
            # delta.append(delta_p)
            x_forward = np.where(x > thre_forward_p, additive_pot[i], x_forward)
            x_backward = np.where(x > thre_backward_p, y_step_p, x_backward)
    p_1_p = additive_pot[i]
    x_backward = np.where(x > p_1_p, 0.1*x+additive_pot[i]-0.1*a_pos[i], x_backward)
    # Plot the result
    # print(f"thre_forward = {thre_forward}")
    # print(f"thre_backward = {thre_backward}")
    # print(f"delta = {delta}")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(x, x_forward, label=f"Segment P_{i}")
    plt.plot(x, x_backward, label=f"Segment P_{i}")
    # plt.plot(x, y_step_0, label=f"Segment P_{i}")
    # plt.plot(x, y_step_1, label=f"Segment P_{i}")
    
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(f"Phi Function for Segment P_{i}")
    plt.xlabel("x")
    plt.ylabel("phi(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("/cluster/home/yujichen/QuantIR/basicsr/archs/phi")

    