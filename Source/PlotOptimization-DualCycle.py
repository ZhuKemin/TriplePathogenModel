from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


class OptimizationPlotter:
    def __init__(self, input_dir, output_file, metrics):
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

    def load_data(self):
        for file in os.listdir(self.input_dir):
            if file.endswith(".csv"):
                cycles = file.split('_')[1:3]
                second_cycle = int(cycles[0])
                third_cycle = int(cycles[1].split('.')[0])
                df = pd.read_csv(os.path.join(self.input_dir, file))
                self.data.append((second_cycle, third_cycle, df))

    def calculate_mae_with_regularization(self, df, lambda_reg=0.001):
        remainder = df['Remainder'].values
        mae = np.mean(np.abs(remainder))
        
        seasonal_columns = [col for col in df.columns if 'Seasonal' in col]
        cycle_lengths = [int(float(col.replace('Seasonal', ''))) for col in seasonal_columns]
        
        regularization_term = lambda_reg * sum(length ** 2 for length in cycle_lengths)
        mae_with_regularization = mae + regularization_term
        return mae_with_regularization


    def calculate_nll_(self, df):
        trend = df['Trend'].values
        seasonals = [col for col in df.columns if 'Seasonal' in col]
        cycles = df[seasonals].sum(axis=1).values
        
        predicted = trend + cycles
        observed = df['Data'].values
        sigma = np.std(observed)

        if sigma < 1e-8:
            sigma = 1e-8
        nll = -(np.log(2 * np.pi * sigma ** 2) + ((observed - predicted) ** 2) / (2 * sigma ** 2)).mean()
        return nll

    def calculate_nll(self, df):
        trend = df['Trend'].values
        seasonals = [col for col in df.columns if 'Seasonal' in col]
        cycles = df[seasonals].sum(axis=1).values
        
        predicted = trend + cycles
        observed = df['Data'].values
        dates = pd.to_datetime(df['Date'])

        # 计算每个数据点所属月份的标准差
        df['Residuals'] = observed - predicted
        df['Month'] = dates.dt.month
        monthly_sigma = df.groupby('Month')['Residuals'].transform(np.std)
        monthly_sigma = monthly_sigma.apply(lambda x: max(x, 1e-8))

        residuals = df['Residuals']
        log_likelihoods = -0.5 * np.log(2 * np.pi * monthly_sigma ** 2) - (residuals ** 2) / (2 * monthly_sigma ** 2)
        nll = -np.mean(log_likelihoods)
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
        
        mae_values = pd.DataFrame(self.results['mae'], columns=["First Cycle", "Second Cycle", "MAE"])
        nll_values = pd.DataFrame(self.results['nll'], columns=["First Cycle", "Second Cycle", "NLL"])

        mae_values["MAE_norm"] = (mae_values["MAE"] - mae_values["MAE"].min()) / (mae_values["MAE"].max() - mae_values["MAE"].min())
        nll_values["NLL_norm"] = (nll_values["NLL"] - nll_values["NLL"].min()) / (nll_values["NLL"].max() - nll_values["NLL"].min())

        multiplicative = mae_values["MAE_norm"] * nll_values["NLL_norm"]
        multiplicative_result = list(zip(mae_values["First Cycle"], mae_values["Second Cycle"], multiplicative))
        self.results['multiplicative'] = multiplicative_result
        
        return multiplicative.mean()

    def calculate_rank(self, df=None):
        self.ensure_metric_calculated('mae')
        self.ensure_metric_calculated('nll')
        
        mae_values = pd.DataFrame(self.results['mae'], columns=["First Cycle", "Second Cycle", "MAE"])
        nll_values = pd.DataFrame(self.results['nll'], columns=["First Cycle", "Second Cycle", "NLL"])

        mae_values["MAE_rank"] = mae_values["MAE"].rank()
        nll_values["NLL_rank"] = nll_values["NLL"].rank()

        rank = (mae_values["MAE_rank"] + nll_values["NLL_rank"]) / (2*mae_values.size)
        rank_result = list(zip(mae_values["First Cycle"], mae_values["Second Cycle"], rank))
        self.results['rank'] = rank_result

        return rank.mean()

    def plot_heatmap(self, metric, data):
        df = pd.DataFrame(data, columns=["First Cycle", "Second Cycle", metric])
        df.loc[df['First Cycle'] == df['Second Cycle'], metric] = np.nan
        
        pivot_table = df.pivot_table(index="Second Cycle", columns="First Cycle", values=metric)
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
        plt.xlabel("First Cycle", fontsize=14, fontname='Times New Roman')
        plt.ylabel("Second Cycle", fontsize=14, fontname='Times New Roman')
        plt.xticks(fontsize=10, fontname='Times New Roman')
        plt.yticks(fontsize=10, fontname='Times New Roman')
        plt.tight_layout()
        plt.savefig(os.path.join(self.input_dir, f"optimization_heatmap_{metric}.png"), dpi=600)


    def plot_mesh(self, metric, data):
        df = pd.DataFrame(data, columns=["First Cycle", "Second Cycle", metric])
        pivot_table = df.pivot_table(index="Second Cycle", columns="First Cycle", values=metric)

        X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
        Z = pivot_table.values

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='coolwarm')
        
        ax.set_xlabel("First Cycle", fontsize=14, fontname='Times New Roman')
        ax.set_ylabel("Second Cycle", fontsize=14, fontname='Times New Roman')
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
    input_dir = "../Output/RSV-Yu-2019/DualCycle/"
    output_file = "optimization_mesh"
    # metrics = ['nll', 'mae', 'multiplicative', 'rank', 'autocorr', 'mae_reg', 'nll_']
    metrics = ['mae_reg']
    metrics = ['nll_poisson']

    plotter = OptimizationPlotter(input_dir, output_file, metrics)
    plotter.load_data()

    results = plotter.calculate_metrics()
    for metric, data in results.items():
        # plotter.plot_mesh(metric, data)
        plotter.plot_heatmap(metric, data)
