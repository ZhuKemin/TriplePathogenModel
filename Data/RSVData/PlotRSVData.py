import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置文件夹路径，此处需要您自己指定包含CSV文件的文件夹路径
folder_path = r'./'

# 获取文件夹内所有CSV文件的路径
csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 创建颜色迭代器，从tab色系列中选择颜色
tab_colors = plt.cm.tab10(np.linspace(0, 1, 10))
color_iterator = iter(tab_colors)

# 读取每个CSV文件，并绘制曲线
for file in csv_files:
    df = pd.read_csv(file, parse_dates=[0])  # 假设第一列是日期，并将其解析为日期格式
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=os.path.basename(file), color=next(color_iterator))

# 设置图例
plt.legend()

# 显示图表
plt.show()
