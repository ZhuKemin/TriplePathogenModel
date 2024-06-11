from matplotlib import pyplot as plt

import pandas as pd
import os


config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",}
plt.rcParams.update(config)


class PlotFluDataOverview:
    def __init__(self):
        self.fn_lookup = r"../Data/FluData/China_influenza_seasonality-main/DES/Data_look_up.xlsx"
        self.path_data = r"../Data/FluData/China_influenza_seasonality-main/cleandata"
        self.path_output = r"../Output/Figure/FluDataOverview"

    def filter_data(self):
        df_lookup = pd.read_excel(self.fn_lookup, usecols=['id_study','csv','year', 'measure',
            'strain','time_unit','pop_denom','start','end','cty','时长'])
        filter_1 = df_lookup['时长']>=6
        filter_2 = df_lookup['time_unit']=='week'
        df_lookup_filtered = df_lookup.loc[filter_1&filter_2]
        return df_lookup_filtered

    def plot_data(self, df_lookup_filtered):
        for index,row in df_lookup_filtered.iterrows():
            filename = row['csv']
            fn_data = os.path.join(self.path_data, filename)+'.csv'
            df_data = pd.read_csv(fn_data, names=['year-week', 'date', 'value'], header=0)

            str_duration = "%i %ss"%(df_data.shape[0]-1, row['time_unit'])
            str_measure = row['measure']
            str_strain = row['strain']

            dates = pd.to_datetime(df_data['date'], format='%Y/%m/%d')
            values = df_data['value']

            fig,ax = plt.subplots(figsize=(12,4))
            ax.plot(dates, values, lw=2, marker='.', color='cornflowerblue')

            plt.grid(ls="-.", lw=0.4, color="lightgray")
            plt.title("%s.csv    %s    %s    %s"%(filename, str_duration, str_measure, str_strain))

            plt.savefig(os.path.join(self.path_output, '%s.png'%(filename)))


    def run(self):
        df_lookup_filtered = self.filter_data()
        self.plot_data(df_lookup_filtered)

if __name__ == '__main__':
    PFDO = PlotFluDataOverview()
    PFDO.run()