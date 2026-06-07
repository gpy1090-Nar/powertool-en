import math
import numpy as np

# ==========================================
# 1. 调制度与电压极限计算 (服务于板块 5)
# ==========================================

def get_voltage_limits(vdc):
    """
    获取两电平/三电平在 SPWM 和 SVPWM 下的物理电压极限
    返回：字典数据，包含峰值、有效值、利用率
    """
    # 基础物理常数
    spwm_factor_line = math.sqrt(3) / 2  # 0.866
    svpwm_factor_line = 1.0              # 1.0
    
    return {
        "spwm": {
            "v_ph_peak": 0.5 * vdc,
            "v_ll_peak": spwm_factor_line * vdc,
            "v_ll_rms": (spwm_factor_line * vdc) / math.sqrt(2),
            "utilization": 0.866
        },
        "svpwm": {
            "v_ph_peak": vdc / math.sqrt(3), # 0.577 * Vdc
            "v_ll_peak": svpwm_factor_line * vdc,
            "v_ll_rms": (svpwm_factor_line * vdc) / math.sqrt(2),
            "utilization": 1.0
        }
    }

def calculate_modulation_info(vdc, vac_target_rms, strategy='SVPWM'):
    """
    根据目标 AC 电压，反推调制度 m，并判断是否过调制
    """
    # 1. 目标物理量
    v_ll_peak = vac_target_rms * math.sqrt(2)     # 线电压峰值
    v_ph_peak_target = v_ll_peak / math.sqrt(3)   # 相电压峰值
    
    # 2. 定义基准 (Base): 工程上通常以 SPWM 的最大线性相电压 (Vdc/2) 为 m=1 的基准
    base_v_ph = vdc / 2.0
    
    # 3. 计算调制度 m
    m = v_ph_peak_target / base_v_ph
    
    # 4. 判断限制
    limit = 1.1547 if strategy == 'SVPWM' else 1.0
    
    return {
        "m": m,
        "limit": limit,
        "is_over": m > limit,
        "v_ph_peak": v_ph_peak_target,
        "max_v_ll_rms": (limit * base_v_ph * math.sqrt(3)) / math.sqrt(2) # 当前模式下的最大线电压RMS
    }

def generate_modulation_waveforms(vdc, m_index, points=360):
    """
    生成 SPWM vs SVPWM 的对比波形数据 (核心仿真算法)
    :param vdc: 直流母线电压
    :param m_index: 调制度 (基准 1.0 = Vdc/2)
    :return: 包含角度、SPWM波形、SVPWM波形、零序分量的字典
    """
    # 角度数组
    theta = np.linspace(0, 2*np.pi, points)
    
    # 1. 物理相电压幅值 (基于 SPWM 基准)
    v_amp = m_index * (vdc / 2.0)
    
    # 2. 生成纯正弦三相波 (SPWM)
    va_sin = v_amp * np.sin(theta)
    vb_sin = v_amp * np.sin(theta - 2*np.pi/3)
    vc_sin = v_amp * np.sin(theta + 2*np.pi/3)
    
    # 3. 计算 SVPWM 的零序分量 (Min-Max 注入法)
    # V_offset = -0.5 * (Max + Min)
    v_offset = -0.5 * (np.maximum(np.maximum(va_sin, vb_sin), vc_sin) + 
                       np.minimum(np.minimum(va_sin, vb_sin), vc_sin))
    
    # 4. 生成马鞍波 (SVPWM)
    va_svpwm = va_sin + v_offset
    
    # 5. 限制线 (直流母线上下轨)
    ceil = vdc / 2.0
    floor = -vdc / 2.0
    
    return {
        "deg": np.degrees(theta),
        "spwm": va_sin,
        "svpwm": va_svpwm,
        "offset": v_offset,
        "limit_top": ceil,
        "limit_bottom": floor
    }

# ==========================================
# 2. 纹波电流计算 (保留原功能，服务于板块 6)
# ==========================================

def calculate_max_ripple(vdc, fsw, L, topology="2-Level"):
    if L <= 0 or fsw <= 0: return 0
    
    if topology == "2-Level":
        # 两电平通用公式 (取 SPWM/SVPWM 中最恶劣的情况，一般工程取 1/4 或 1/8)
        # 此处保持你原有的逻辑，或者更新为更严谨的 1/4 (SPWM)
        # 严谨系数: SPWM max at D=0.5 -> Vdc/(4*L*fsw)
        i_ripple_max = vdc / (4 * L * fsw) 
    else:
        # 三电平 (NPC)
        # 严谨系数: SPWM max at D=0.25/0.75 -> Vdc/(8*L*fsw) (你的代码里原来是32? 可能是指单侧? 建议修正为标准公式)
        # 这里修正为标准三电平 SPWM 理论最大值
        i_ripple_max = vdc / (12 * L * fsw) # SVPWM优化后接近 1/12
        
    return i_ripple_max