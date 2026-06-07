import numpy as np
import math

def calculate_lcl_parameters(ug, fg, l1, l2, c):
    if l1 <= 0 or l2 <= 0 or c <= 0:
        return {"error": "Inductance and capacitance values must be greater than zero."}
    w_res = math.sqrt((l1 + l2) / (l1 * l2 * c))
    f_res = w_res / (2 * math.pi)
    w_grid = 2 * math.pi * fg
    z_c = 1 / (w_grid * c)
    return {"f_res": f_res, "z_c": z_c, "status": "ok"}

def get_simulation_waveform(ug, fg, fsw, cycles=3):
    t = np.linspace(0, cycles/fg, 1000)
    v_grid = ug * math.sqrt(2) * np.sin(2 * np.pi * fg * t)
    i_fundamental = 10 * np.sin(2 * np.pi * fg * t)
    i_ripple = 2 * np.sin(2 * np.pi * fsw * t)
    i_total = i_fundamental + i_ripple
    return t, v_grid, i_total
