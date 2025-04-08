import os
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def add_gaussian_noise(data, noise_level_percentage):
    """
    添加高斯噪声到归一化数据
    Args:
        data (np.ndarray): 输入特征，值范围应在 [0, 1]。
        noise_level_percentage (float): 噪声强度，百分比形式。
    Returns:
        np.ndarray: 加噪后的数据，仍在 [0, 1] 范围内。
    """
    noise_stddev = noise_level_percentage / 100
    noise = np.random.normal(0, noise_stddev, data.shape)
    noisy_data = np.clip(data + noise, 0, 1)
    return noisy_data


def process_mat_files(input_folder, output_folder, noise_levels):
    """
    遍历输入文件夹中的所有 .mat 文件，为每个文件添加不同程度的高斯噪声，并保存为新的 .mat 文件。
    Args:
        input_folder (str): 原始 .mat 文件所在路径。
        output_folder (str): 添加噪声后的 .mat 文件保存路径。
        noise_levels (List[int]): 要添加的噪声强度百分比列表。
    """
    os.makedirs(output_folder, exist_ok=True)
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

    for file_name in file_list:
        file_path = os.path.join(input_folder, file_name)
        mat = scipy.io.loadmat(file_path)
        X_raw = mat['X']
        y_raw = np.array(mat['Y']).ravel()
        X = MinMaxScaler().fit_transform(X_raw)
        Y = LabelEncoder().fit_transform(y_raw)
        for level in noise_levels:
            noisy_X = add_gaussian_noise(X, level)
            save_path = os.path.join(output_folder, f'{file_name[:-4]}_noisy_{level}.mat')
            scipy.io.savemat(save_path, {'X': noisy_X, 'Y': Y})
            print(f'[INFO] Saved: {save_path}')


if __name__ == "__main__":
    input_dir = './data/noise/raw'
    output_dir = './data/noise'
    noise_levels = [5, 10, 15, 20]

    process_mat_files(input_dir, output_dir, noise_levels)
