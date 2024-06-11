import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",}
plt.rcParams.update(config)



for fn in ['Cao-2024', 'Jia-2022', 'Li-2024', 'Yu-2019', 'Zou-2016']:
    # fn = 'Cao-2024'
    csv_file_path = './%s.csv'%(fn)

    # 读取CSV文件
    df = pd.read_csv(csv_file_path, parse_dates=['t'])

    # 提取年份和月份
    df['Year'] = df['t'].dt.year
    df['Month'] = df['t'].dt.month

    # 分组计算每月的平均值
    monthly_avg = df.groupby('Month')['data'].mean()

    fig,ax = plt.subplots(figsize=(12, 3))

    unique_months = sorted(df['Month'].unique())

    # 绘制每年的浅色线
    for year in df['Year'].unique():
        yearly_data = df[df['Year'] == year]
        ax.plot(yearly_data['Month'], yearly_data['data'], color='royalblue', marker='.', alpha=0.1)

    # 绘制多年的平均深色线
    ax.plot(monthly_avg.index, monthly_avg.values, color='royalblue', marker='o', linewidth=2)

    ax.set_xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlim(1, 12)
    # ax.set_ylim(0, 40)

    # 设置图表的标题和轴标签
    ax.set_title(fn)
    # ax.set_xlabel('Month')
    ax.set_ylabel('Positive Rate(%)', labelpad=12)
    ax.grid(which='both', linestyle='-.', linewidth=0.5, color='lightgrey')

    if fn=='Yu-2019':
        ax.set_ylabel('Number of Cases', labelpad=12)

    plt.tight_layout()
    plt.savefig('./%s.png'%(fn))
