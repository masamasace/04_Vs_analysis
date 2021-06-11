from datetime import time
import os
from pathlib import Path
import pandas as pd
import traceback
import sys
import math
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import gc

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams['font.size'] = 14

class DataDir:

    def __init__(self, path, intial_accelerometer_distance=100.00, stage="con", time_dif=60, ratio_thres=0.05):
        
        print("---start of analysis---")


        # set constants
        self.time_dif = time_dif
        self.intial_accelerometer_distance = intial_accelerometer_distance
        self.dir_path = Path(path)
        self.ratio_thres = ratio_thres


        # check the stage
        if stage == "con":
            self.stage_num = 3
        elif stage == "aft_con":
            self.stage_num = 4
        else:
            self.stage_num = 2


        # initilize 
        self._import_data()
        self._analysis_vs()
    

    def update_time_dif(self, time_dif):
        self.time_dif = time_dif

    
    def _import_data(self):
        

        # find .spe, .out and .dat data
        spe_file_path = list(self.dir_path.glob("*.spe"))
        if len(spe_file_path) != 1:
            _raise_error(type="not_one_file_found", ext="spe")  

        out_file_path = list(self.dir_path.glob("*.out"))
        if len(out_file_path) != 1:
            _raise_error(type="not_one_file_found", ext="out")

        dat_file_path = list(self.dir_path.glob("*.dat"))
        if len(dat_file_path) != 1:
            _raise_error(type="not_one_file_found", ext="dat")
        
        
        # load data
        self.spe_data = pd.read_csv(spe_file_path[0], sep="\t", header=None)
        self.out_data = pd.read_csv(out_file_path[0], sep="\t")
        self.dat_data = pd.read_csv(dat_file_path[0], sep="\t")
        self.out_file_path_stem = out_file_path[0].stem


        # delete unnamed column
        self.out_data = self.out_data.loc[:, ~self.out_data.columns.str.contains('^Unnamed')]
        self.dat_data = self.dat_data.loc[:, ~self.dat_data.columns.str.contains('^Unnamed')]


        # create new column
        self.out_data["Time(s)_total"] = os.path.getmtime(out_file_path[0]) + self.out_data["Time(s)"]
        self.out_data["Vs_rise(m/s)"] = 0.0


        # find Vs data
        self.vs_dir_path = self.dir_path / "vs"
        self.vs_data = []
        for vs_data_path in self.vs_dir_path.glob("*.TXT"):
            self.vs_data.append([vs_data_path, vs_data_path.stem, os.path.getmtime(vs_data_path) + self.time_dif, 0., 0.])
        
        
        # make result directory
        self.result_dir_path = self.dir_path / "res"
        self.result_vs_dir_path = self.dir_path / "res" / "vs"

        if self.result_dir_path.exists():
            for p in self.result_dir_path.iterdir():
                if p.is_file():
                    p.unlink()
        else:
            self.result_dir_path.mkdir()
        
        if self.result_vs_dir_path.exists():
            for p in self.result_vs_dir_path.iterdir():
                if p.is_file():
                    p.unlink()
        else:
            self.result_vs_dir_path.mkdir()
 
    

    def _analysis_vs(self):


        # calculate current accelerometer distance
        accelerometer_distance_at_the_beginning_of_stage = self.intial_accelerometer_distance * self.spe_data.iloc[3, self.stage_num] / self.spe_data.iloc[3, 2]

        # self.vs_data = self.vs_data[:1]

        # multi process analysis of shear wave velocity
        with concurrent.futures.ProcessPoolExecutor() as executor:
            params = map(lambda vs_data_each: (accelerometer_distance_at_the_beginning_of_stage, self.out_data, vs_data_each, self.ratio_thres), self.vs_data)
            results = executor.map(self._analysis_vs_each, params)
        
        for result in results:
            self.out_data["Vs_rise(m/s)"].iloc[result[0]] = result[1]
        
        non_zero_vs_index = np.where(self.out_data["Vs_rise(m/s)"] != 0)[0]
        self.out_data_summary = self.out_data.iloc[non_zero_vs_index]
        
        fig, axes = setup_figure()
        axes.plot(self.out_data_summary["p'___(kPa)"], self.out_data_summary["Vs_rise(m/s)"])
        plt.show()

        answer_validation = input("put 'ok' or digit number to change the time difference >>")

        if answer_validation == "ok":
            out_data_summary_path = self.result_dir_path / (self.out_file_path_stem + ".csv")
            self.out_data_summary.to_csv(out_data_summary_path, index=False)
        else:
            try:
                temp = float(answer_validation)
            except ValueError:
                pass
            else:
                self.time_dif += temp
                print("current time_dif: " + str(self.time_dif))
                self._import_data()
                self._analysis_vs()


    def _analysis_vs_each(self, params):


        # compile params
        accelerometer_distance_at_the_beginning_of_stage, out_data, vs_data_each, ratio_thres = params


        # compute current accelerometer distance 
        time_dif_abs = abs(out_data["Time(s)_total"] - vs_data_each[2])
        measured_time_index = time_dif_abs.idxmin()
        current_accelerometer_distance = math.exp(-out_data["e(a)_(%)_"].iloc[measured_time_index] / 100) * accelerometer_distance_at_the_beginning_of_stage


        # load Vs file
        vs_data = pd.read_csv(vs_data_each[0],
                            delimiter=",", 
                            header=None, 
                            skiprows=9,
                            names=["timecol", "inputwavecol", "outputwavecol1", "outputwavecol2"]).astype('float64')


        # ignore offset
        initial_stabilized_index = int(np.abs(vs_data.iloc[:, 0]).idxmin() / 2)
        initial_offset = vs_data.iloc[:initial_stabilized_index, 1:4].mean(axis=0).values
        vs_data.iloc[:, 1:4] = vs_data.iloc[:, 1:4] - initial_offset


        # compute threshold value from maximum value
        max_value = np.abs(vs_data.iloc[:, 1:4]).max()
        threshold_value = max_value * ratio_thres


        # detect rise index
        value_over_thres = np.where(np.abs(vs_data.iloc[:, 1:4]) > threshold_value)

        rise_index = np.array([0, 0, 0])
        for i in range(3):
            first_index_over_thres = np.where(value_over_thres[1] == i)[0][0]
            rise_index[i] = value_over_thres[0][first_index_over_thres]


        # compute duration
        duration_time = abs(vs_data.iloc[rise_index[1], 0] - vs_data.iloc[rise_index[2], 0])


        # compute shear wave velocity
        vs_rise = current_accelerometer_distance / duration_time / 1000


        # draw result figure
        fig, axes = setup_figure()
        
        axes.plot(vs_data.iloc[:, 0], vs_data.iloc[:, 2], "b")
        axes.plot(vs_data.iloc[:, 0], vs_data.iloc[:, 3], "r")
        axes.plot(vs_data.iloc[rise_index[1], 0], vs_data.iloc[rise_index[1], 2], marker="*", markersize=10, color="w", markerfacecolor="b")
        axes.plot(vs_data.iloc[rise_index[2], 0], vs_data.iloc[rise_index[2], 3], marker="*", markersize=10, color="w", markerfacecolor="r")

        axes.hlines(0, xmin=vs_data.iloc[0, 0], xmax=vs_data.iloc[-1, 0])

        axes.set_xlabel("Time (sec)")
        axes.set_ylabel("Volatage (V)")
        axes.grid(linestyle=":", linewidth=0.5, color="k")

        label = vs_data_each[1] + "\n" + "Vs: " + '{:.2f}'.format(vs_rise) + "m/s"
        axes.text(0.05, 0.05, label, fontsize=14, transform=axes.transAxes, bbox=dict(facecolor='w', edgecolor='k', pad=5.0, linewidth=0.75))

        figure_path = self.result_vs_dir_path / (vs_data_each[1] + ".svg")
        fig.savefig(figure_path, format="svg", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        gc.collect

        # return value
        return(measured_time_index, vs_rise)
    
    

def _raise_error(type, ext):
    if type == "not_one_file_found":
        try:
            raise ValueError(ext + " file's number is not one. Please confirm the number of file with same extension code")
        except ValueError as e:
            traceback.print_exc()
            sys.exit(1)


def setup_figure(num_row=1, num_col=1, width=6, height=6, left=0.125, right=0.9, hspace=0.2, wspace=0.2):

    fig, axes = plt.subplots(num_row, num_col, figsize=(width, height))
    fig.subplots_adjust(left=left, right=right, hspace=hspace, wspace=wspace)
    return (fig, axes)


def main():
    dir_path = r"/Users/ms/Shiga Dropbox/01_work/04_2021-_assistant professor/01_research/03_codes/04_Vs_analysis/consolidation"
    wave_dir = DataDir(path=dir_path, intial_accelerometer_distance=100.0, stage="con")


if __name__ == "__main__":
    main()