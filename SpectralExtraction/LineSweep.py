import spectral
import glob
import spectral
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import pandas 
import os
import pandas as pd
import glob

# # 读取高光谱文件
def read_hsi(file_path):
    msi = spectral.open_image(file_path).load()
    return msi



def read_roi(file_path):
    with open(file_path, 'r') as f: # 这行代码的意思是打开文件file_path，模式为只读
        lines = f.readlines() # 读取所有行  
        rois = []
        for line in lines:
            line = line.strip()
            if line:
                rois.append(line)
    roi_list = []
    for roi in rois :
        roi = roi.split(' ')
        x = float(roi[1])
        y = float(roi[2])
        w = float(roi[3])
        h = float(roi[4])

        x = float(x-w/2)
        y = float(y-h/2)
        roi_list.append([roi[0] , x, y, w, h ])

    return roi_list 

def get_spectral_split( rois , msi ,show_roi=False , rows = 5, columns = 2  ) : 
    #输入ROI文件和一张原始光谱数据，和想要的roilabels，能够得出平均切分roi内部的光谱
    roi_list = []
    labels = []
    for roi in rois :
            x = int(np.array(msi.shape[1]*float(roi[1])))
            y = int(np.array(msi.shape[0]*float(roi[2])))

            w = int(np.array(msi.shape[1]*float(roi[3])))
            h = int(np.array(msi.shape[0]*float(roi[4])))

            roi_list.append([x,y,w,h])
            labels.append(np.array([roi[0]]*rows*columns ).reshape(-1))




    if show_roi :
        img = msi[:,:, int(msi.shape[2]/2)]
        min_val = np.min(img)
        max_val = np.max(img)
        # 将图像线性映射到0-255范围内
        converted_image = (img - min_val) * (255 / (max_val - min_val))
        # 将图像的值限制在0-255范围内
        converted_image = np.clip(converted_image, 0, 255)
        # 将图像的数据类型转换为无符号8位整数
        converted_image = converted_image.astype(np.uint8)
        image_colored = cv2.cvtColor(converted_image, cv2.COLOR_GRAY2BGR)


    spectrum = []
    for roi in roi_list:
        x1 = roi[0]
        y1 = roi[1]
        x2 = roi[0] + roi[2]
        y2 = roi[1] + roi[3]
           

        sub_boxes = split_roi(x1, y1, x2, y2 , rows , columns ) # 切分ROI,row是切分行数，columns是切分列
        Spectral = []
        for j in range(len(sub_boxes)):
            Spe =np.mean( msi[ sub_boxes[j][1] : sub_boxes[j][3] , sub_boxes[j][0] : sub_boxes[j][2] ], axis=(0, 1))
            Spectral.append(Spe)

        if show_roi :# 显示划分出来的ROI是否正确
          for j in range(len(sub_boxes)):
            image_colored = cv2.rectangle(image_colored, (sub_boxes[j][0], sub_boxes[j][1]), (sub_boxes[j][2], sub_boxes[j][3]), (0, 255, 255), 2)  # 在图像上绘制红色矩形框
        
        spectrum.append(np.array(Spectral))


    if show_roi :# 显示划分出来的ROI是否正确
      plt.imshow(image_colored)


    return spectrum , labels

def split_roi(x1, y1, x2, y2 , rows = 5, columns = 2 ) :  # 计算子矩形框的坐标
  sub_boxes = []
  for i in range(rows):
    for j in range(columns):
      sub_x1 = int(x1 + j * (x2 - x1) / columns)
      sub_y1 = int(y1 + i * (y2 - y1) / rows)
      sub_x2 = int(x1 + (j + 1) * (x2 - x1) / columns)
      sub_y2 = int(y1 + (i + 1) * (y2 - y1) / rows)
      sub_boxes.append((sub_x1, sub_y1, sub_x2, sub_y2))
  return sub_boxes

def get_spectral_Threshold( rois , msi  , show_roi = False) : 

    spectrum = []
    labels = []

    for roi in rois :
        label = roi[0] 
        x = int(np.array(msi.shape[1]*float(roi[1])))
        y = int(np.array(msi.shape[0]*float(roi[2])))
        w = int(np.array(msi.shape[1]*float(roi[3])))
        h = int(np.array(msi.shape[0]*float(roi[4])))


        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h


        img = msi[y1:y2, x1:x2, 300]
        min_val = np.min(img)
        max_val = np.max(img)
        # 将图像线性映射到0-255范围内
        converted_image = (img - min_val) * (255 / (max_val - min_val))
        # 将图像的值限制在0-255范围内
        converted_image = np.clip(converted_image, 0, 255)
        # 将图像的数据类型转换为无符号8位整数
        converted_image = converted_image.astype(np.uint8)

        _, mask = cv2.threshold(converted_image, 60, 255 , cv2.THRESH_BINARY)# 把亮的部分分出来，白扁豆
        # _, mask = cv2.threshold(msi[y1:y2,x1:x2,300], 155, 255 , cv2.THRESH_BINARY_INV)#把暗的部分分出来，川楝子
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.erode(mask, kernel)
        mask_bool = mask.astype(bool)
        
        if show_roi : 
            plt.figure()
            plt.imshow(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
        average_spectral_values = []
        for contour in contours:
            mask2 = np.zeros_like(mask)
            cv2.drawContours(mask2, [contour], 0, 255, -1)
            values = []
            mask2 = mask2.astype(bool)
            shape = msi[y1:y2, x1:x2, :].shape
            for i in range(msi.shape[2]):
                flattened_msi = msi[y1:y2, x1:x2, i].reshape(shape[0] , shape[1])
                values.append(np.mean(flattened_msi[mask2]))
            average_spectral_values.append(values)


        average_spectral_values = np.array(average_spectral_values)
        
        labels.append([label] * average_spectral_values.shape[0])
        spectrum.append(average_spectral_values)
        

    spectrum = np.concatenate(spectrum)
    labels = np.concatenate(labels)
  
    return spectrum, labels

def get_reflect_line_sweep(msi, baiban_path  , show = False) : # 线扫相机通过大白板进行反射率校正 , 仅仅输入白板横轴
    reflect = np.zeros(msi.shape)
    baiban_msi = spectral.open_image(baiban_path).load()
    baiban_hengzhou = np.mean(baiban_msi, axis=0)
    for i in range(msi.shape[2]):
        for j in range(msi.shape[0]) :
            reflect[j,:, i] =  msi[j,:, i].reshape(-1) / baiban_hengzhou[:,i]
    
    if show : 
        plt.imshow(msi[:,:,int(msi.shape[2]/2)])
        # 设置名称
        plt.title('original')
        plt.show()
        plt.imshow(baiban_msi[:,:,int(msi.shape[2]/2)])
        # 设置名称
        plt.title('white board')
        plt.show()
        plt.imshow(reflect[:,:,int(msi.shape[2]/2)])
        # 设置名称
        plt.title('reflect')
        plt.show()
    return reflect