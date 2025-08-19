# 导入必要的库
import spectral  # 用于处理光谱数据
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import glob  # 用于文件路径模式匹配
import cv2  # 用于图像处理（如果需要）
import pandas as pd  # 用于数据处理
import os  # 用于文件和目录操作

# 定义白板校正类
class WhiteBoardCorrection:
    """
    白板校正类，用于对线扫相机获取的光谱数据进行反射率校正。
    
    参数:
    white_board_path (str): 白板图像的文件路径。
    """
    
    def __init__(self, white_board_path: str) -> None:
        """
        初始化白板校正类。
        
        参数:
        white_board_path (str): 白板图像的文件路径。
        """
        # 加载白板图像并计算其平均反射率
        self.baiban_hengzhou = np.mean(spectral.open_image(white_board_path).load(), axis=0)
    
    def get_reflect_line_sweep(self, msi, reflectivity=0.5) -> np.ndarray:
        """
        使用白板图像对线扫相机获取的光谱数据进行反射率校正。
        
        参数:
        msi (np.ndarray): 线扫相机获取的光谱数据。
        reflectivity (float): 白板的反射率，默认为0.5。
        
        返回:
        np.ndarray: 校正后的反射率数据。
        """
        # 初始化反射率数组
        reflect = np.zeros(msi.shape)
        
        # 对每个波段进行反射率校正
        for i in range(msi.shape[2]):
            # 对每个像素点进行反射率计算
            for j in range(msi.shape[0]):
                reflect[j, :, i] = (msi[j, :, i].reshape(-1) / self.baiban_hengzhou[:, i]) * reflectivity
        
        return reflect
if __name__ == '__main__':
    # 加载样品光谱数据
    msi = spectral.open_image(r'G:\咖啡豆20250731\20250804_100607_895\vSpex.hdr').load()

    # 创建白板校正对象
    correction = WhiteBoardCorrection(rf'G:\烟丝加料3D高光谱数据\250729\LOW\newdata20250729_095037.hdr')

    # 使用白板校正对象对样品光谱数据进行反射率校正
    reflect = correction.get_reflect_line_sweep(msi)

    # 绘制校正后的第8个波段的反射率图像
    plt.imshow(reflect[:, :, 8], cmap='gray')
    plt.colorbar()  # 显示颜色条
    plt.title('Corrected Reflectance Line Sweep')  # 图像标题
    plt.xlabel('Pixel')  # x轴标签
    plt.ylabel('Wavelength')  # y轴标签
    plt.show()  # 显示图像