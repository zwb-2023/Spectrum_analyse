import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from collections import Counter

#——————————————————————绘图可视化————————————————————————————

def split_data(labels, datas):
    """
    根据标签将数据分割为不同类别。

    参数：
    labels (list): 数据的类别标签。
    datas (list): 待分割的数据。

    返回：
    category (list): 数据类别的标签列表。
    split_datas (list): 按照标签分割后的数据列表，每个元素是一个类别对应的数据。
    """
    # 检查输入的标签和数据是否匹配
    if len(labels) != len(datas):
        raise ValueError("标签和数据的长度不匹配")

    # 统计标签的数量
    label_counts = Counter(labels)

    split_datas = []
    start_idx = 0
    for label, count in label_counts.items():
        # 根据标签的计数切分数据
        split_datas.append(datas[start_idx:start_idx + count])
        start_idx += count

    # 获取类别标签的顺序
    categories = list(label_counts.keys())

    return categories, split_datas



def plot_duplicate_data(category, split_datas, x, type='all'):
    import matplotlib.font_manager as fm
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择合适的中文字体，这里使用了黑体作为示例
    plt.rcParams['axes.unicode_minus'] = False  # 使负号能够正常显示
    # 动态生成颜色
    num_colors = len(category)
    cmap = plt.cm.get_cmap('tab20', num_colors)  # 使用 'tab20' 调色板，并指定颜色数量

    fig, ax = plt.subplots()
    
    # 为每个类别分配一种颜色
    for i in range(len(category)):
        y = split_datas[i]
        color = cmap(i)  # 从调色板中取颜色
        if type == 'mean':
            ax.plot(x, np.mean(y.transpose(), axis=1), color=color, label=category[i])
        else:
            ax.plot(x, y.transpose(), color=color, label=category[i])

    ax.set(xlabel='Wavelength(nm)', ylabel='reflectivity')

    handles, labels = plt.gca().get_legend_handles_labels()
    
    # 创建新的图例，合并相同名称和颜色的项
    unique_labels = set(labels)
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    
    # 按标签顺序排序
    sorted_labels_handles = sorted(set(zip(unique_labels, unique_handles)), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_labels_handles)
    
    # 创建新的图例，按排序后的顺序
    plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()



#——————————————————————标准化处理————————————————————————————


def snv(ori_specs):
    """标准正态化
    对多个光谱数据进行标准正态化校正，理想光谱采用的是平均光谱
    @param ori_specs: Numpy数组,原始光谱,形状为[n, spec]
    @return: 标准正态变化后的光谱
    """
    # Define v0.1.1-a new array and populate it with the corrected data
    snv_specs = np.zeros_like(ori_specs)
    for i in range(ori_specs.shape[0]):
        # Apply correction
        snv_specs[i, :] = (ori_specs[i, :] - np.mean(ori_specs[i, :])) / np.std(ori_specs[i, :])
    return snv_specs


#————————————————————————计算差异性——————————————————————————————
# 计算光谱角（Spectral Angle）
def spectral_angle(spectrum_array):
    """计算所有样本光谱之间的光谱角，并返回其平均值"""
    num_samples = spectrum_array.shape[0]
    spectral_angles = np.zeros((num_samples, num_samples))  # 存储每对光谱的角度

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dot_product = np.dot(spectrum_array[i], spectrum_array[j])
            norm_spectrum1 = np.linalg.norm(spectrum_array[i])
            norm_spectrum2 = np.linalg.norm(spectrum_array[j])
            angle = np.arccos(dot_product / (norm_spectrum1 * norm_spectrum2))
            spectral_angles[i, j] = angle
            spectral_angles[j, i] = angle
    
    return spectral_angles  # 返回光谱角的平均值

# 计算均方差（Mean Squared Error, MSE）
def mean_squared_error(spectrum_array):
    """计算所有样本光谱之间的均方差，并返回其平均值"""
    num_samples = spectrum_array.shape[0]
    mse_values = np.zeros((num_samples, num_samples))  # 存储每对光谱的均方差

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            mse = np.mean((spectrum_array[i] - spectrum_array[j]) ** 2)
            mse_values[i, j] = mse
            mse_values[j, i] = mse
    
    return mse_values  # 返回均方差的平均值

# 计算变异系数（Coefficient of Variation, CV）
def coefficient_of_variation(spectrum_array):
    """计算所有样本光谱的变异系数，并返回其平均值"""
    cv_values = np.array([np.std(spectrum) / np.mean(spectrum) if np.mean(spectrum) != 0 else 0 for spectrum in spectrum_array])
    return cv_values # 返回变异系数的平均值



#————————————————————相关性分析————————————————

def pearson_correlation(data_frame, labels):
    """
    计算每个光谱与标签之间的皮尔逊相关系数。

    参数：
    data_frame (numpy.ndarray or pd.DataFrame): 光谱数据，形状为 (n_samples, n_features)。
    labels (list or numpy.ndarray): 标签列表，长度与样本数一致。

    返回：
    list: 每个波长和标签的皮尔逊相关系数。
    """
    correlation_pearson = []
    
    for i in range(data_frame.shape[0]):
        xx = np.array(data_frame[i, :])  # 提取当前样本的数据
        result = np.corrcoef(xx, labels)  # 计算皮尔逊相关系数矩阵
        correlation_pearson.append(result[0][1])  # 提取相关系数

    return correlation_pearson


def spearman_correlation(data_frame, labels):
    """
    计算每个光谱与标签之间的斯皮尔曼相关系数。

    参数：
    data_frame (numpy.ndarray or pd.DataFrame): 光谱数据，形状为 (n_samples, n_features)。
    labels (list or numpy.ndarray): 标签列表，长度与样本数一致。

    返回：
    list: 每个波长和标签的斯皮尔曼相关系数。
    """
    correlation_spearman = []
    
    for i in range(data_frame.shape[0]):
        xx = np.array(data_frame[i, :])  # 提取当前样本的数据
        df = pd.DataFrame({'spectrum': xx, 'label': labels})
        result = df.corr(method='spearman')  # 计算斯皮尔曼相关系数
        correlation_spearman.append(result.iloc[0, 1])  # 提取相关系数

    return correlation_spearman





#——————————————————————波长筛选————————————————
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def SPA_select(spectrum, label, k ):
    # 初始化空的特征集和得分
    selected_features = []
    best_scores = []

    # 迭代选择k个最佳特征
    for _ in range(k):
        best_score = float('-inf')
        best_feature_idx = None

        # 对于每个未选择的特征
        for i in range(spectrum.shape[1]):
            if i not in selected_features:
                # 添加当前特征到已选择特征集
                selected_features.append(i)
                X_selected = spectrum[:, selected_features]

                # 划分训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(X_selected, label, test_size=0.2, random_state=42)

                # 训练模型并评估性能
                model = LinearRegression()
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                # 如果当前得分更好，则更新最佳特征索引和最佳得分
                if score > best_score:
                    best_score = score
                    best_feature_idx = i

                # 从已选择特征集中移除当前特征
                selected_features.remove(i)

        # 将最佳特征索引添加到已选择特征集中
        selected_features.append(best_feature_idx)
        best_scores.append(best_score)


    return selected_features, best_scores


def plot_selected_features(wavelength, spectrum , selected_features, best_scores, k):
    """
    绘制选定的特征波长与分数

    参数：
    wavelength (list or numpy.ndarray): 波长列表。
    selected_features (list): 选定的特征索引。
    best_scores (list): 选定特征的得分。
    spectrum (numpy.ndarray): 光谱数据，形状为 (n_samples, n_features)。
    k (int): 要显示的选定特征的个数。
    
    返回：
    None
    """
    # 转置光谱数据
    X_plant = spectrum.transpose()

    # 创建图形
    fig, bx = plt.subplots()

    # 绘制光谱数据
    bx.plot(wavelength, np.array(X_plant))
    
    # 设置图例和标签
    bx.set(xlabel='Wavelength(nm)',
           title='SPA selected features',)

    # 设置字体
    font3 = {'family' : 'Arial',
             'weight' : 'normal',
             'size'   : 8}

    
    # 标出选定的特征
    for i in range(k):
        feature_idx = selected_features[i]
        score = best_scores[i]
        bx.axvline(x=wavelength[feature_idx], color='r')  # 标记选定的波长
        print(f'筛选出来的波长为: {wavelength[feature_idx]} nm, 分数为: {score:.4f}')
    
    # 显示网格
    bx.grid()
    
    # 显示图像
    plt.show()


from sklearn.model_selection import KFold

def k_fold_cross_validation(X, y, model, n_splits=5):
    """
    使用五折交叉验证训练模型，并绘制混淆矩阵。
    
    参数:
    X : numpy array
        特征矩阵 (n_samples, n_features)
    y : numpy array or list
        标签对应的特征矩阵
    model : sklearn model
        要训练的机器学习模型
    n_splits : int
        交叉验证的折数，默认为5
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    Y_pred = []
    Y_test = []

    # 进行五折交叉验证
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 汇总结果
        Y_pred.extend(y_pred)
        Y_test.extend(y_test)

    # 计算混淆矩阵
    cm = confusion_matrix(Y_test, Y_pred)

    # 获取唯一的标签
    unique_labels = np.unique(y)

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # 计算并打印准确率
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"五折交叉验证的平均准确率: {accuracy:.2f}")





from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def leave_one_out_cross_validation(X, y, model):
    """
    Perform Leave-One-Out Cross Validation (LOO CV) on the given dataset.
    
    Parameters:
    X : numpy array
        Feature matrix (n_samples, n_features)
    y : numpy array or list
        Labels corresponding to the feature matrix
    model : sklearn model
        A machine learning model (e.g., SVM) to train and predict
    
    Returns:
    float
        The mean accuracy of the model across all leave-one-out iterations
    """
    accuracies = []
    Y_pred = []
    Y_test = []
    
    # Perform Leave-One-Out Cross Validation
    for i in range(len(y)):
        # Split the data into train and test sets
        X_train, X_test = np.delete(X, i, axis=0), X[i, :]
        y_train, y_test = np.delete(y, i), y[i]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test.reshape(1, -1))
        
        # Store the predictions and true labels
        Y_pred.append(y_pred)
        Y_test.append(y_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracies.append(accuracy)
    
    # Mean accuracy
    mean_accuracy = np.mean(accuracies)
    print(f"留一交叉验证的平均准确率: {mean_accuracy:.2f}")
    
    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    
    # Get unique labels for plotting
    unique_labels = np.unique(y)
    
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    return mean_accuracy
