from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os




def mae_with_regularization_scorer(estimator, X, y):
    df = pd.DataFrame(X, columns=['Remainder'])
    df['Remainder'] = y
    return estimator.calculate_mae_with_regularization(df)

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

class OptimizationPlotterWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dir, output_file, metrics, lambda_reg=0.1):
        self.input_dir = input_dir
        self.output_file = output_file
        self.metrics = metrics
        self.lambda_reg = lambda_reg
        self.plotter = OptimizationPlotter(input_dir, output_file, metrics, lambda_reg)

    def fit(self, X, y=None):
        # 不需要实际拟合，只是为了GridSearchCV接口的兼容
        return self

    def calculate_mae_with_regularization(self, df):
        return self.plotter.calculate_mae_with_regularization(df)

    def get_params(self, deep=True):
        return {
            'input_dir': self.input_dir,
            'output_file': self.output_file,
            'metrics': self.metrics,
            'lambda_reg': self.lambda_reg,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.plotter = OptimizationPlotter(self.input_dir, self.output_file, self.metrics, self.lambda_reg)
        return self


class OptimizationPlotter:
    def __init__(self, input_dir, output_file, metrics, lambda_reg=0.1):
        self.input_dir = input_dir
        self.output_file = output_file
        self.metrics = metrics
        self.data = []
        self.metric_functions = {
            'nll': self.calculate_nll,
            'mae': self.calculate_mae,
            'multiplicative': self.calculate_multiplicative,
            'rank': self.calculate_rank,
            'autocorr': self.calculate_autocorr_significance,
            'mae_reg': self.calculate_mae_with_regularization,
            'nll_': self.calculate_nll_,
            
        }
        self.results = {}

        self.lambda_reg = lambda_reg

    def load_data(self):
        for file in os.listdir(self.input_dir):
            if file.endswith(".csv"):
                cycles = file.split('_')[1:3]
                second_cycle = int(cycles[0])
                third_cycle = int(cycles[1].split('.')[0])
                df = pd.read_csv(os.path.join(self.input_dir, file))
                self.data.append((second_cycle, third_cycle, df))

    def calculate_mae_with_regularization(self, df):
        remainder = df['Remainder'].values
        mae = np.mean(np.abs(remainder))
        seasonal_columns = [col for col in df.columns if 'Seasonal' in col]
        cycle_lengths = [int(col.replace('Seasonal', '')) for col in seasonal_columns]
        regularization_term = self.lambda_reg * sum(length ** 2 for length in cycle_lengths)
        mae_with_regularization = mae + regularization_term
        return mae_with_regularization

    def calculate_nll(self, df):
        remainder = df['Remainder'].values
        sigma = np.std(remainder)
        
        if sigma < 1e-8:
            sigma = 1e-8
        nll = -(np.log(2 * np.pi * sigma ** 2) + (remainder ** 2) / (2 * sigma ** 2)).mean()
        return nll

    def calculate_mae(self, df):
        remainder = df['Remainder'].values
        mae = np.mean(np.abs(remainder))
        return mae

    def calculate_autocorr_significance(self, df, lags=10):
        remainder = df['Remainder'].values
        lag_acf = acf(remainder, nlags=lags)
        ljung_box_result = acorr_ljungbox(remainder, lags=[lags], return_df=True)
        p_value = ljung_box_result['lb_pvalue'].iloc[0]
        return -p_value

    def calculate_mae_with_regularization(self, df):
        remainder = df['Remainder'].values
        mae = np.mean(np.abs(remainder))
        seasonal_columns = [col for col in df.columns if 'Seasonal' in col]
        cycle_lengths = [int(float(col.replace('Seasonal', ''))) for col in seasonal_columns]
        regularization_term = self.lambda_reg * sum(length ** 2 for length in cycle_lengths)
        mae_with_regularization = mae + regularization_term
        return mae_with_regularization

    def find_best_lambda(self, X, y, param_grid):
        from sklearn.model_selection import GridSearchCV

        def mae_with_regularization_scorer(estimator, X, y):
            df = pd.DataFrame(X, columns=['Remainder'])
            df['Remainder'] = y
            return estimator.calculate_mae_with_regularization(df)

        grid_search = GridSearchCV(self, param_grid, scoring=mae_with_regularization_scorer, cv=5)
        grid_search.fit(X, y)

        self.lambda_reg = grid_search.best_params_['lambda_reg']
        print(f"Best lambda_reg: {self.lambda_reg}")

    def calculate_mae_with_regularization_(self, df, lambda_reg=0.01):
        """
        计算包含正则化惩罚项的MAE。
        """
        remainder = df['Remainder'].values
        mae = np.mean(np.abs(remainder))
        
        seasonal_columns = [col for col in df.columns if 'Seasonal' in col]
        cycle_lengths = [int(float(col.replace('Seasonal', ''))) for col in seasonal_columns]
        regularization_term = lambda_reg * sum(length ** 2 for length in cycle_lengths)
        mae_with_regularization = mae + regularization_term
        return mae_with_regularization

    def calculate_metrics(self):
        # 先计算基础指标
        basic_metrics = ['nll', 'mae']
        for metric in basic_metrics:
            if metric in self.metrics:
                self.results[metric] = []
                for second_cycle, third_cycle, df in self.data:
                    value = self.metric_functions[metric](df)
                    self.results[metric].append((second_cycle, third_cycle, value))
        
        # 再计算复合指标
        composite_metrics = [m for m in self.metrics if m not in basic_metrics]
        for metric in composite_metrics:
            if metric in self.metrics:
                self.results[metric] = []
                for second_cycle, third_cycle, df in self.data:
                    value = self.metric_functions[metric](df)
                    self.results[metric].append((second_cycle, third_cycle, value))
        
        return self.results

    def ensure_metric_calculated(self, metric):
        if metric not in self.results:
            self.results[metric] = []
            for second_cycle, third_cycle, df in self.data:
                value = self.metric_functions[metric](df)
                self.results[metric].append((second_cycle, third_cycle, value))

    def calculate_multiplicative(self, df=None):
        self.ensure_metric_calculated('mae')
        self.ensure_metric_calculated('nll')
        
        mae_values = pd.DataFrame(self.results['mae'], columns=["Second Cycle", "Third Cycle", "MAE"])
        nll_values = pd.DataFrame(self.results['nll'], columns=["Second Cycle", "Third Cycle", "NLL"])

        mae_values["MAE_norm"] = (mae_values["MAE"] - mae_values["MAE"].min()) / (mae_values["MAE"].max() - mae_values["MAE"].min())
        nll_values["NLL_norm"] = (nll_values["NLL"] - nll_values["NLL"].min()) / (nll_values["NLL"].max() - nll_values["NLL"].min())

        multiplicative = mae_values["MAE_norm"] * nll_values["NLL_norm"]
        multiplicative_result = list(zip(mae_values["Second Cycle"], mae_values["Third Cycle"], multiplicative))
        self.results['multiplicative'] = multiplicative_result
        
        return multiplicative.mean()

    def calculate_rank(self, df=None):
        self.ensure_metric_calculated('mae')
        self.ensure_metric_calculated('nll')
        
        mae_values = pd.DataFrame(self.results['mae'], columns=["Second Cycle", "Third Cycle", "MAE"])
        nll_values = pd.DataFrame(self.results['nll'], columns=["Second Cycle", "Third Cycle", "NLL"])

        mae_values["MAE_rank"] = mae_values["MAE"].rank()
        nll_values["NLL_rank"] = nll_values["NLL"].rank()

        rank = (mae_values["MAE_rank"] + nll_values["NLL_rank"]) / (2*mae_values.size)
        rank_result = list(zip(mae_values["Second Cycle"], mae_values["Third Cycle"], rank))
        self.results['rank'] = rank_result

        return rank.mean()

    def plot_heatmap(self, metric, data):
        df = pd.DataFrame(data, columns=["Second Cycle", "Third Cycle", metric])
        df.loc[df['Second Cycle'] == df['Third Cycle'], metric] = np.nan
        
        pivot_table = df.pivot_table(index="Third Cycle", columns="Second Cycle", values=metric)
        pivot_table = pivot_table.iloc[::-1]
        
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        cmap.set_bad(color='dimgray')
        
        row_to_highlight = pivot_table.index.get_loc(12)
        best_in_row_cycle = pivot_table.loc[12].idxmin()
        best_in_row_index = pivot_table.columns.get_loc(best_in_row_cycle)
        
        min_value = pivot_table.min().min()
        min_positions = np.where(pivot_table == min_value)
        
        plt.figure(figsize=(12, 12))
        ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'shrink': .8}, annot_kws={"size": 6},
                         mask=pivot_table.isnull(), cbar=True, linewidths=.5, linecolor='lightgrey', square=True)

        for i in range(pivot_table.shape[1]):
            ax.add_patch(plt.Rectangle((i, row_to_highlight), 1, 1, fill=False, edgecolor='dimgray', lw=0.5))

        ax.add_patch(plt.Rectangle((best_in_row_index, row_to_highlight), 1, 1, fill=False, edgecolor='black', lw=2))

        for y, x in zip(min_positions[0], min_positions[1]):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='white', lw=2))

        plt.title(f'Heatmap for {metric}', fontsize=20, fontname='Times New Roman')
        plt.xlabel("Second Cycle", fontsize=14, fontname='Times New Roman')
        plt.ylabel("Third Cycle", fontsize=14, fontname='Times New Roman')
        plt.xticks(fontsize=10, fontname='Times New Roman')
        plt.yticks(fontsize=10, fontname='Times New Roman')
        plt.tight_layout()
        plt.savefig(os.path.join(self.input_dir, f"optimization_heatmap_{metric}.png"), dpi=600)


    def plot_mesh(self, metric, data):
        df = pd.DataFrame(data, columns=["Second Cycle", "Third Cycle", metric])
        pivot_table = df.pivot_table(index="Third Cycle", columns="Second Cycle", values=metric)

        X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
        Z = pivot_table.values

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='coolwarm')
        
        ax.set_xlabel("Second Cycle", fontsize=14, fontname='Times New Roman')
        ax.set_ylabel("Third Cycle", fontsize=14, fontname='Times New Roman')
        ax.set_zlabel(metric, fontsize=14, fontname='Times New Roman')
        ax.set_title(f'Mesh Plot for {metric}', fontsize=20, fontname='Times New Roman')
        
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
            tick.label1.set_fontname('Times New Roman')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
            tick.label1.set_fontname('Times New Roman')
        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(12)
            tick.label1.set_fontname('Times New Roman')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.input_dir, f"optimization_mesh_{metric}.png"), dpi=600)


if __name__ == "__main__":
    input_dir = "../../Output/Seasonality/RSV-Yu-2019/AnnualFixed-DualCycle/"
    output_file = "optimization_mesh"
    metrics = ['nll', 'mae', 'mae_reg', 'multiplicative', 'rank', 'autocorr']

    plotter = OptimizationPlotter(input_dir, output_file, metrics)

    plotter.load_data()

    # 准备数据
    all_data = [df for _, _, df in plotter.data]
    all_remainders = np.concatenate([df['Remainder'].values for df in all_data])

    # 定义正则化参数范围
    param_grid = {'lambda_reg': np.logspace(-4, 2, 50)}

    # 创建包装类实例
    wrapped_plotter = OptimizationPlotterWrapper(input_dir, output_file, metrics)

    # 找到最优的 lambda_reg
    grid_search = GridSearchCV(wrapped_plotter, param_grid, scoring=mae_with_regularization_scorer, cv=5)
    grid_search.fit(all_remainders.reshape(-1, 1), all_remainders)

    best_lambda_reg = grid_search.best_params_['lambda_reg']
    print(f"Best lambda_reg: {best_lambda_reg}")

    # 使用最优的正则化参数
    plotter.lambda_reg = best_lambda_reg
    results = plotter.calculate_metrics()

    for metric, data in results.items():
        plotter.plot_mesh(metric, data)
        plotter.plot_heatmap(metric, data)
