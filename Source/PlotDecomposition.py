import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from statsmodels.tsa.stattools import ccf
from mpl_toolkits.axes_grid1 import make_axes_locatable



class DecompositionPlotter:
    def __init__(self, input_dir, pathogen):
        self.input_dir = input_dir
        self.pathogen = pathogen
        self.ext = 52 if pathogen == 'Flu' else 12

        self.output_dir = os.path.join(input_dir, "Plots")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        plt.rcParams["font.family"] = "Times New Roman"

        self.calculate_global_ylim()  # 计算全局最小值和最大值

    def calculate_global_ylim(self):
        self.global_min_max = {
            'Observed': [np.inf, -np.inf],
            'Trend': [np.inf, -np.inf],
            'Seasonal': [np.inf, -np.inf],
            'Residual': [np.inf, -np.inf],
            'Correlation': [-0.5, 1.1]  # Correlation ylim 是固定的
        }

        for file in os.listdir(self.input_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(self.input_dir, file))

                # 更新 Observed 的全局最小值和最大值
                observed = df['Data'].values
                self.global_min_max['Observed'][0] = min(self.global_min_max['Observed'][0], observed.min())
                self.global_min_max['Observed'][1] = max(self.global_min_max['Observed'][1], observed.max())

                # 更新 Trend 的全局最小值和最大值
                trend = df['Trend'].values
                self.global_min_max['Trend'][0] = min(self.global_min_max['Trend'][0], trend.min())
                self.global_min_max['Trend'][1] = max(self.global_min_max['Trend'][1], trend.max())

                # 更新 Seasonal 的全局最小值和最大值
                seasonal_columns = [col for col in df.columns if col.startswith('Seasonal')]
                for col in seasonal_columns:
                    seasonal = df[col].values
                    self.global_min_max['Seasonal'][0] = min(self.global_min_max['Seasonal'][0], seasonal.min())
                    self.global_min_max['Seasonal'][1] = max(self.global_min_max['Seasonal'][1], seasonal.max())

                # 更新 Residual 的全局最小值和最大值
                resid = df['Remainder'].values
                self.global_min_max['Residual'][0] = min(self.global_min_max['Residual'][0], resid.min())
                self.global_min_max['Residual'][1] = max(self.global_min_max['Residual'][1], resid.max())

        # 为每个类型的 ylim 增加 10% 的范围
        for key in ['Observed', 'Trend', 'Seasonal', 'Residual']:
            range_min, range_max = self.global_min_max[key]
            range_span = range_max - range_min
            self.global_min_max[key][0] = range_min - 0.1 * range_span
            self.global_min_max[key][1] = range_max + 0.1 * range_span

    def load_data(self, filename):
        filepath = os.path.join(self.input_dir, filename)
        df = pd.read_csv(filepath)
        return df

    def plot_decomposition(self, cycle1, cycle2):
        # if not ((cycle1==25)*(cycle2==19)):
        #     return

        filename = f"cycle_{cycle1}_{cycle2}.csv"
        df = self.load_data(filename)

        dates = pd.to_datetime(df['Date'].values)
        observed = df['Data'].values
        trend = df['Trend'].values
        resid = df['Remainder'].values
        predict = np.maximum(observed - resid, 0)

        # 动态识别季节性成分列
        seasonal_columns = [col for col in df.columns if col.startswith('Seasonal')]
        num_seasonal = len(seasonal_columns)

        num_axes = 4 + num_seasonal
        fig, axs = plt.subplots(num_axes, 1, figsize=(15, 2 * num_axes), sharex=False)
        ax1 = axs[0]
        ax_trend = axs[1]
        ax_resid = axs[2 + num_seasonal]
        ax_autocorr = axs[3 + num_seasonal]

        # Plot observation vs. prediction 
        ax1.plot(dates, observed, lw=2, color='tab:blue', label='Observed')
        ax1.plot(dates, predict, lw=1, ls='--', color='tab:red', label='Predicted')
        ax1.set_ylabel('Observed', fontsize=18)
        ax1.legend(fontsize=12)
        ax1.set_ylim(self.global_min_max['Observed'])  # 设置统一的 ylim

        # Plot trend
        ax_trend.plot(dates, trend, lw=2, color='tab:green')
        ax_trend.set_ylabel('Trend', fontsize=18)
        ax_trend.set_ylim(self.global_min_max['Trend'])  # 设置统一的 ylim

        # Plot multiple seasonality
        for i, col in enumerate(seasonal_columns):
            ax_seasonal = axs[2 + i]
            col_int = int(float(col.replace('Seasonal', '')))
            ax_seasonal.plot(dates, df[col].values, lw=2, color='tab:orange')
            ax_seasonal.set_ylabel(f'Seasonal{col_int}', fontsize=18)
            ax_seasonal.set_ylim(self.global_min_max['Seasonal'])  # 设置统一的 ylim

            # Draw auxiliary lines for the seasonal cycle
            if 'Flu' in self.pathogen:
                date_offset = pd.DateOffset(weeks=col_int)
            else:
                date_offset = pd.DateOffset(months=col_int)

            current_date = dates[0]
            while current_date <= dates[-1]:
                ax_seasonal.axvline(x=current_date, color='black', linestyle='--')
                current_date += date_offset

        # Plot resid
        ax_resid.scatter(dates, resid, color='tab:red')
        ax_resid.set_ylabel('Residual', fontsize=18)
        ax_resid.set_ylim(self.global_min_max['Residual'])  # 设置统一的 ylim

        # Calculate and plot auto-correlation with lags
        correlations = ccf(resid, resid, adjusted=True)
        correlations_combined = np.concatenate((correlations[1:self.ext + 1][::-1], correlations[:self.ext + 1]))
        lags = np.arange(-self.ext, self.ext + 1)

        ax_autocorr.stem(lags, correlations_combined, linefmt='-', markerfmt='o', basefmt='r-')
        ax_autocorr.grid(which='both', linestyle='-.', linewidth=0.5, color='lightgrey')
        ax_autocorr.set_ylabel('Correlation', fontsize=18)
        ax_autocorr.set_ylim(self.global_min_max['Correlation'])  # 设置统一的 ylim
        ax_autocorr.set_xlim(-self.ext - 0.2, self.ext + 0.2)

        # Plot kernel density of resid
        divider = make_axes_locatable(ax_resid)
        ax_histy = divider.append_axes("right", 0.8, pad=0.1, sharey=ax_resid)
        ax_histy.yaxis.set_tick_params(labelleft=False)
        sns.kdeplot(y=resid, ax=ax_histy, color='tab:gray', lw=2, alpha=0.6)
        ax_histy.xaxis.set_visible(False)
        ax_histy.set_xlim(0, 0.1)  # Adjust as needed

        # Formatting plots
        for ax in axs[:-1]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=14)
            ax.grid(which='both', linestyle='-.', linewidth=0.5, color='lightgrey')
            ax.set_axisbelow(True)
            ax.set_xlim(dates[0], dates[-1])

        # Format ax_autocorr separately
        ax_autocorr.xaxis.set_major_formatter(ticker.FuncFormatter(self.custom_format))
        ax_autocorr.yaxis.set_major_formatter(ticker.FuncFormatter(self.custom_format))
        ax_autocorr.xaxis.set_tick_params(labelsize=12)
        ax_autocorr.set_xlim(lags[0], lags[-1])
        ax_autocorr.yaxis.set_tick_params(labelsize=14)

        fig.align_labels()
        plt.suptitle(f"Cycle={cycle1}-{cycle2}", fontsize=24)
        plt.subplots_adjust(top=0.95)
        output_filepath = os.path.join(self.output_dir, f"cycle_{cycle1}_{cycle2}.png")
        plt.tight_layout()
        # plt.savefig(output_filepath, dpi=600)
        plt.savefig(output_filepath)
        plt.close()

    def batch_plot(self):
        for file in os.listdir(self.input_dir):
            if file.endswith(".csv"):
                cycle_info = file.split('_')[1:3]
                cycle1 = int(cycle_info[0])
                cycle2 = int(cycle_info[1].split('.')[0])
                self.plot_decomposition(cycle1, cycle2)

    def custom_format(self, x, pos):
        if x % 1 == 0:
            return f'{int(x)}'
        else:
            return f'{x:.1f}'

if __name__ == "__main__":
    input_dir = "../../Output/Seasonality/RSV-Yu-2019/AnnualFixed-DualCycle/"
    input_dir = "../../Output/Seasonality/RSV-Yu-2019/DualCycle/"
    pathogen = "RSV"

    plotter = DecompositionPlotter(input_dir, pathogen)
    plotter.batch_plot()
