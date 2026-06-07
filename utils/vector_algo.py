import numpy as np
import math

def calculate_vector_analysis(u_g_rms_line, i_rms, angle_deg, u_dc, L_val, f_grid=50.0):
    # ... (复制 calculate_vector_analysis 的代码) ...
    # 记得把原来 math_func.py 最后面那个大的函数粘过来
    omega = 2 * np.pi * f_grid
    U_g_peak = u_g_rms_line * math.sqrt(2) / math.sqrt(3)
    vec_Ug = U_g_peak + 0j
    I_peak = i_rms * math.sqrt(2)
    phi = math.radians(angle_deg)
    vec_I = I_peak * (math.cos(phi) + 1j * math.sin(phi))
    vec_VL = 1j * omega * L_val * vec_I
    vec_Uconv = vec_Ug + vec_VL
    limit_radius = u_dc / math.sqrt(3)
    margin = limit_radius - abs(vec_Uconv)
    is_over_modulation = margin < 0
    
    return {
        "vec_Ug": vec_Ug,
        "vec_VL": vec_VL,
        "vec_Uconv": vec_Uconv,
        "limit_radius": limit_radius,
        "margin": margin,
        "is_over": is_over_modulation
    }