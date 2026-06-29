import numpy as np
import matplotlib.pyplot as plt

def get_luminance_based_strength(luminance, strength_params=None):
    """
    根据亮度图确定每个像素的回加强度
    strength与luminance的关系为两侧低、中间高的曲线
    Args:
        luminance: (H, W) 亮度图，范围[0,1]
        strength_params: 回加强度参数字典，包含：
            - 'peak_position': 峰值位置,默认0.5(中间亮度)
            - 'peak_strength': 峰值回加强度,默认0.9
            - 'min_strength': 最小回加强度(最暗和最亮处),默认0.3
            - 'curve_type': 曲线类型，可选 'quadratic'(二次函数), 'gaussian'(高斯), 'cubic'(三次函数)，默认'quadratic'
            - 'width': 曲线宽度参数(仅对gaussian有效),默认0.3
    Returns:
        strength_map: (H, W) 回加强度图
    """
    if strength_params is None:
        strength_params = {
            'peak_position': 0.6,      # 峰值位置（亮度0.5处回加强度最高）
            'peak_strength': 0.9,      # 峰值回加强度
            'min_strength': 0.8,       # 最小回加强度（最暗和最亮处）
            'curve_type': 'quadratic',  # 曲线类型：'gaussian' 或 'quadratic'
            'width': 0.3               # 宽度参数（仅高斯曲线使用）
        }

    peak_pos = strength_params['peak_position']
    peak_str = strength_params['peak_strength']
    min_str = strength_params['min_strength']
    curve_type = strength_params.get('curve_type', 'quadratic')
    #curve_type = 'linear'
    width = strength_params.get('width', 0.3)

    if curve_type == 'quadratic':
        # 二次函数曲线：在peak_pos处达到峰值，两端为最小值
        # 使用分段二次函数实现两侧低中间高
        # 左半段：从0到peak_pos，强度从min_str增加到peak_str
        # 右半段：从peak_pos到1，强度从peak_str减少到min_str
        strength_map = np.zeros_like(luminance)
        
        left_mask = luminance <= peak_pos
        if peak_pos > 0:
            t = luminance[left_mask] / peak_pos  # 归一化到[0,1]
            # 二次函数：f(t) = min_str + (peak_str - min_str) * (1 - (1-t)^2)
            # 或者使用简单的二次函数：f(t) = min_str + (peak_str - min_str) * (2*t - t^2)
            strength_map[left_mask] = min_str + (peak_str - min_str) * (2*t - t**2)
        
        right_mask = luminance > peak_pos
        if peak_pos < 1:
            t = (luminance[right_mask] - peak_pos) / (1 - peak_pos)  # 归一化到[0,1]
            # 二次函数递减：f(t) = peak_str - (peak_str - min_str) * t^2
            strength_map[right_mask] = peak_str - (peak_str - min_str) * (t**2)
    elif curve_type == 'gaussian':
        strength_map = min_str + (peak_str - min_str) * np.exp(-((luminance - peak_pos)**2) / (2 * width**2))

    else:
        strength_map = peak_str

    return strength_map

if __name__ == "__main__":
    # ---------- add back map ---------- #
    lstLum = np.linspace(0.0, 1.0, 100)
    lstMapF32 = get_luminance_based_strength(lstLum)
    lstMapU8 = np.round(lstMapF32 * 255).astype(np.uint8)

    plt.subplot(1,2,1)
    plt.plot(lstLum, lstMapF32)
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(lstMapU8)
    plt.show()

    cpp_str = np.array2string(
        lstMapU8,
        separator=', ',
        prefix='uint8_t lstMap[] = {',
        suffix='};',
        threshold=np.inf,
        edgeitems=0,
        # formatter={'int': lambda x: f'0x{x:02X}'}
    )
    print(cpp_str)