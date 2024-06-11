import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import boxcox1p, inv_boxcox1p

class PandemicImpactAdjuster:
    def __init__(self, csv_path, date_column, data_column, pandemic_start_date, output_path, lambda_param=0.0):
        self.csv_path = csv_path
        self.date_column = date_column
        self.data_column = data_column
        self.pandemic_start_date = pd.to_datetime(pandemic_start_date)
        self.output_path = output_path
        self.lambda_param = lambda_param
        self.data = None
        self.results = None

    def load_data(self):
        self.data = pd.read_csv(self.csv_path)
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column], format='%Y/%m/%d')
        self.data['Pandemic'] = np.where(self.data[self.date_column] >= self.pandemic_start_date, 1, 0)
        self.data[self.data_column] = boxcox1p(self.data[self.data_column], self.lambda_param)

    def fit_model(self):
        formula = f'{self.data_column} ~ Pandemic'
        model = smf.glm(formula=formula, data=self.data, family=sm.families.Gamma(link=sm.families.links.log()))
        self.results = model.fit()
        self.data['predicted_pandemic_impact'] = self.results.predict(self.data['Pandemic'])
        self.data['adjusted_data'] = np.where(self.data['Pandemic'] == 1, 
                                              self.data[self.data_column] - self.data['predicted_pandemic_impact'], 
                                              self.data[self.data_column])
        self.data['adjusted_data'] = inv_boxcox1p(self.data['adjusted_data'], self.lambda_param)
        self.data[self.data_column] = inv_boxcox1p(self.data[self.data_column], self.lambda_param)
        self.data['adjusted_data'] = self.data['adjusted_data'].clip(lower=0)

    def save_adjusted_data(self):
        self.data[[self.date_column, 'adjusted_data']].to_csv(self.output_path, index=False)

    def plot_data(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.data[self.date_column], self.data[self.data_column], label='Original Data', color='blue', marker='o')
        plt.plot(self.data[self.date_column], self.data['adjusted_data'], label='Adjusted Data', color='red', marker='o')
        plt.title('Remove the impact of the pandemic using GLM with Gamma distribution and Box-Cox transform')
        plt.xlabel('Date')
        plt.ylabel('Data')
        plt.legend()
        plt.grid(ls="-.", lw=0.4, color="lightgray")
        plt.savefig("./adjusted_data.png")
        # plt.show()

    def run(self):
        self.load_data()
        self.fit_model()
        self.save_adjusted_data()
        self.plot_data()

# 使用示例
config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
}
plt.rcParams.update(config)

adjuster = PandemicImpactAdjuster(
    csv_path="./Jia-2022.csv",
    date_column='t',
    data_column='data',
    pandemic_start_date="2019-12-01",
    output_path="adjusted_data.csv",
    lambda_param=0.0  # Box-Cox变换的lambda参数，可以根据需要调整
)

adjuster.run()



"""
在处理流感阳性率数据时，我们采用了一种结合广义线性模型（GLM）和Box-Cox变换的方法，以确保数据的合理性，特别是避免负值估计。具体的算法思路如下：

Box-Cox变换：

我们对数据列应用Box-Cox变换（带有指定的λ参数），将数据转换为更适合建模的形式。Box-Cox变换能够将非正态分布的数据转换为近似正态分布，并处理数据中的零和负值问题，使其适合后续的建模过程。
广义线性模型（GLM）：

使用假设为伽马分布的广义线性模型，并结合对数链接函数，对数据进行建模。伽马分布适用于正值数据，而对数链接函数则确保了模型的预测值始终为正。这对于阳性率数据尤为重要，因为它避免了负值估计。
模型拟合与调整：

通过大流行标记列（Pandemic）作为自变量，拟合广义线性模型。模型拟合后，计算出预测的大流行影响值。
对于大流行期间的数据，从实际数据中减去预测的大流行影响值，得到调整后的数据。为了确保调整后的数据不出现负值，对调整后的数据应用clip操作，使其下限为0。
反Box-Cox变换：

将调整后的数据和原始数据应用反Box-Cox变换，恢复到原始尺度。这一步确保调整后的数据在合理范围内，并保留了原始数据的特性。
通过这种方法，我们成功地避免了负值估计，保证了流感阳性率数据的合理性和准确性。Box-Cox变换处理了数据的非正态分布问题，而广义线性模型结合对数链接函数则确保了预测值的非负性。
"""
