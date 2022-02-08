import datetime
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
from scipy import signal
import time
from scipy.fft import fft, ifft
from numpy.fft.helper import fftfreq
from scipy import integrate

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams['font.size'] = 14
pd.options.mode.chained_assignment = None

class DataDir:

    def __init__(self, path, 
                 intial_accelerometer_distance=100.00, 
                 stage="con", 
                 time_dif=60, 
                 ratio_thres=0.1,
                 spe_skiprows=8,
                 method_computing_tau="direct",
                 label_tau="tau(zq)(kPa)",
                 method_computing_gamma="direct",
                 label_epsilon_r="e_(r)",
                 label_mean_effective_stress="p_(kPa)",
                 label_axial_strain="e_(z)",
                 label_effective_lateral_stress="sigma(r)(kPa)",
                 label_effective_axial_stress="sigma(z)(kPa)"):
        
        print("---start of analysis---")

        # set constants
        self.time_dif = time_dif
        self.intial_accelerometer_distance = intial_accelerometer_distance
        self.dir_path = Path(path)
        self.ratio_thres = ratio_thres
        self.low_pass_params = {"fp": 5000,
                                "fs":  7000,
                                "gpass": 3,
                                "gstop": 10}
        self.spe_skiprows = spe_skiprows
        self.method_computing_tau = method_computing_tau
        self.label_tau = label_tau
        self.method_computing_gamma = method_computing_gamma
        self.label_epsilon_r = label_epsilon_r
        self.label_mean_effective_stress = label_mean_effective_stress
        self.label_axial_strain = label_axial_strain
        self.label_effective_lateral_stress = label_effective_lateral_stress
        self.label_effective_axial_stress = label_effective_axial_stress

        # check the stage
        if stage == "con":
            self.stage_num = 3
        elif stage == "aft_con":
            self.stage_num = 4
        else:
            self.stage_num = 2


        # initilize 
        self._import_data()
    

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
        self.spe_data = pd.read_csv(spe_file_path[0], sep="\t", header=None, skiprows=self.spe_skiprows)
        self.out_data = pd.read_csv(out_file_path[0], sep="\t")
        self.dat_data = pd.read_csv(dat_file_path[0], sep="\t")
        self.out_file_path_stem = out_file_path[0].stem


        # delete unnamed and NaNcolumn
        self.spe_data = self.spe_data.dropna(axis=1, how='all')
        print(self.spe_data)
        self.out_data = self.out_data.loc[:, ~self.out_data.columns.str.contains('^Unnamed')]
        self.dat_data = self.dat_data.loc[:, ~self.dat_data.columns.str.contains('^Unnamed')]


        # create new column
        self.out_data["Time(s)_total"] = os.path.getmtime(out_file_path[0]) + self.out_data["Time(s)"]
        self.out_data["Vs_rise(m/s)"] = 0.0

        # 損失エネルギーの計算
        if self.method_computing_tau == "direct":
            self.out_data["tau__(kPa)"] = self.out_data[self.label_tau]
        else:
            self.out_data["tau__(kPa)"] = self.out_data["q____(kPa)"] / 2
        if self.method_computing_gamma == "direct":
            self.out_data["gamma_(%)_"] = self.out_data[self.label_epsilon_r]
        else:
            self.out_data["gamma_(%)_"] = self.out_data[self.label_axial_strain] * 3 / 2
        self.out_data["CDE__(kPa)"] = integrate.cumtrapz(self.out_data["tau__(kPa)"], self.out_data["gamma_(%)_"], initial=0)
        
        # 國生先生流の正規化累積損失エネルギー(Normalized Cumulative Dissipated Energy, NCDE)
        self.out_data["NCDE_(___)"] = self.out_data["CDE__(kPa)"] / self.out_data[self.label_mean_effective_stress].iloc[0]

        # find Vs data
        self.vs_dir_path = self.dir_path / "vs"
        self.vs_data = []
        for vs_data_path in self.vs_dir_path.glob("*.TXT"):
            self.vs_data.append([vs_data_path, vs_data_path.stem, self.time_dif, 0., 0.])
        
        
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

        print("---data imported---")
      
    def analysis_vs_with_offset(self):

        print("---analysis starts (offset: %d)) ---" % self.time_dif)
        # calculate current accelerometer distance
        accelerometer_distance_at_the_beginning_of_stage = self.intial_accelerometer_distance * self.spe_data.iloc[3, self.stage_num] / self.spe_data.iloc[3, 2]


        # multi process analysis of shear wave velocity
        with concurrent.futures.ProcessPoolExecutor() as executor:
            params = map(lambda vs_data_each: (accelerometer_distance_at_the_beginning_of_stage, 
                                               self.out_data, 
                                               vs_data_each, 
                                               self.ratio_thres,
                                               self.low_pass_params), self.vs_data)
            results = executor.map(self._analysis_vs_each_with_offset, params)
        
        for result in results:
            self.out_data["Vs_rise(m/s)"].iloc[result[0]] = result[1]
        
        non_zero_vs_index = np.where(self.out_data["Vs_rise(m/s)"] != 0)[0]
        self.out_data_summary = self.out_data.iloc[non_zero_vs_index, :]

        # compute geometric mean effective stress
        self.out_data_summary.loc[:, "GMES__(kPa)"] = (self.out_data_summary.loc[:, self.label_effective_axial_stress] * self.out_data_summary.loc[:, self.label_effective_lateral_stress]) ** (1/2)
        
        fig, axes = setup_figure()
        axes.plot(self.out_data_summary[self.label_mean_effective_stress], self.out_data_summary["Vs_rise(m/s)"], marker=".", linewidth=0, markersize=10, markeredgecolor="w",markerfacecolor="k", markeredgewidth=0.25)
        axes.set_xlabel("Mean Effective Stress, $\sqrt{\sigma'_1 \sigma'_3}$ (kPa)")
        axes.set_ylabel("Vs_rise, $V_s$ (m/s)")
        axes.set_ylim([0, 250])
        axes.grid(linestyle=":", linewidth=0.5, color="k")
        plt.show()

        answer_validation = input("put 'ok' or digit number to change the time difference >>")

        if answer_validation == "ok":
            print("---analysis ended (offset: %d)) ---" % self.time_dif)
            out_data_summary_path = self.result_dir_path / (self.out_file_path_stem + "_extracted.csv")
            summary_figure_path = self.result_dir_path / (self.out_file_path_stem + "_extracted.svg")
            self.out_data_summary.to_csv(out_data_summary_path, index=False)
            fig.savefig(summary_figure_path, format="svg", dpi=300, bbox_inches='tight')
        else:
            try:
                temp = float(answer_validation)
            except ValueError:
                pass
            else:
                self.time_dif += temp
                self._import_data()
                self.analysis_vs_with_offset()



    def _analysis_vs_each_with_offset(self, params):

        # compile params
        accelerometer_distance_at_the_beginning_of_stage, out_data, vs_data_each, ratio_thres, low_pass_params = params

        # load Vs file
        vs_data = pd.read_csv(vs_data_each[0],
                            delimiter=",", 
                            header=None, 
                            skiprows=0,
                            names=["timecol", "inputwavecol", "outputwavecol1", "outputwavecol2", "none"])

        vs_data = vs_data.iloc[:, :-1]
        
        # compute current accelerometer distance
        date = vs_data.iloc[1, 1]
        cur_time = vs_data.iloc[2, 1]
        vs_data_timestamp = datetime.datetime(int(date[6:10]), int(date[0:2]), int(date[3:5]), int(cur_time[0:2]), int(cur_time[3:5]), int(cur_time[6:8]))

        time_dif_abs = abs(out_data["Time(s)_total"] - int(time.mktime(vs_data_timestamp.timetuple())) - vs_data_each[2])
        measured_time_index = time_dif_abs.idxmin()
        print(vs_data_each[1], cur_time, out_data["Time(s)_total"][0], int(time.mktime(vs_data_timestamp.timetuple())))
        current_accelerometer_distance = math.exp(-out_data[self.label_axial_strain].iloc[measured_time_index] / 100) * accelerometer_distance_at_the_beginning_of_stage


        # ignore non-numeric record
        vs_data = vs_data.iloc[9:, :].astype(float)
        
        # FFT
        dt = vs_data.iloc[1, 0] - vs_data.iloc[0, 0]
        fn = 1 / (2*dt)

        data_num = len(vs_data)
        freq = fftfreq(data_num, dt)[:data_num//2]

        vs_data_fft = np.empty((0, data_num//2), int)

        for i in range(3):
            vs_data_fft = np.vstack((vs_data_fft, 2.0 / data_num * np.abs(fft(vs_data.iloc[:, i+1].values))[:data_num//2]))

        # ignore offset
        initial_stabilized_index = int(np.abs(vs_data.iloc[:, 0]).idxmin() / 2)
        initial_offset = vs_data.iloc[:initial_stabilized_index, 1:4].mean(axis=0).values
        vs_data.iloc[:, 1:4] = vs_data.iloc[:, 1:4] - initial_offset

        # frequency normalization 
        wp = low_pass_params["fp"] / fn
        ws = low_pass_params["fs"] / fn
        

        # low-pass filterization
        N, Wn = signal.buttord(wp, ws, low_pass_params["gpass"], low_pass_params["gstop"])
        b1, a1 = signal.butter(N, Wn, "low")
        vs_data["outputwavecol1_filtered"] = signal.filtfilt(b1, a1, vs_data.iloc[:, 2])
        vs_data["outputwavecol2_filtered"] = signal.filtfilt(b1, a1, vs_data.iloc[:, 3])


        # compute threshold value from maximum value
        max_value = np.abs(vs_data.iloc[:, 1:6]).max()
        threshold_value = max_value * ratio_thres        

        # detect rise index
        value_over_thres = np.where(np.abs(vs_data.iloc[:, 1:6]) > threshold_value)
        print(value_over_thres[1])
        
        rise_index = np.array([0, 0, 0, 0 ,0])
        for i in range(5):
            first_index_over_thres = np.where(value_over_thres[1] == i)[0][0]
            rise_index[i] = value_over_thres[0][first_index_over_thres]

        # compute duration
        duration_time = abs(vs_data.iloc[rise_index[3], 0] - vs_data.iloc[rise_index[4], 0])

        # compute shear wave velocity
        vs_rise = current_accelerometer_distance / duration_time / 1000

        # draw result figure
        fig, axes = setup_figure(num_row=3, height=10, hspace=0.3)
        
        # axes[0]: entire stretch
        axes[0].plot(vs_data.iloc[:, 0], vs_data.iloc[:, 4], "b", linewidth=0.5)
        axes[0].plot(vs_data.iloc[:, 0], vs_data.iloc[:, 5], "r", linewidth=0.5)
        axes[0].plot(vs_data.iloc[rise_index[3], 0], vs_data.iloc[rise_index[3], 4], marker="*", markersize=10, color="w", markerfacecolor="b")
        axes[0].plot(vs_data.iloc[rise_index[4], 0], vs_data.iloc[rise_index[4], 5], marker="*", markersize=10, color="w", markerfacecolor="r")
        axes[0].set_xlabel("Time (sec)")
        axes[0].set_ylabel("Volatage (V)")
        axes[0].grid(linestyle=":", linewidth=0.5, color="k")

        # axes[1]: expanded area
        axes[1].plot(vs_data.iloc[:, 0], vs_data.iloc[:, 2], marker=".", linewidth=0, markerfacecolor="b", markeredgewidth=0.25)
        axes[1].plot(vs_data.iloc[:, 0], vs_data.iloc[:, 3], marker=".", linewidth=0, markerfacecolor="r", markeredgewidth=0.25)
        axes[1].plot(vs_data.iloc[:, 0], vs_data.iloc[:, 4], "b")
        axes[1].plot(vs_data.iloc[:, 0], vs_data.iloc[:, 5], "r")
        axes[1].plot(vs_data.iloc[rise_index[3], 0], vs_data.iloc[rise_index[3], 4], marker="*", markersize=10, color="w", markerfacecolor="b")
        axes[1].plot(vs_data.iloc[rise_index[4], 0], vs_data.iloc[rise_index[4], 5], marker="*", markersize=10, color="w", markerfacecolor="r")
        axes[1].hlines(0, xmin=vs_data.iloc[0, 0], xmax=vs_data.iloc[-1, 0])
        axes[1].set_xlabel("Time (sec)")
        axes[1].set_ylabel("Volatage (V)")
        axes[1].grid(linestyle=":", linewidth=0.5, color="k")
        axes[1].set_xlim([-0.001, 0.0025])

        # axes[2]: frequecy range
        axes[2].plot(freq, vs_data_fft[1], "b", linewidth=0.5)
        axes[2].plot(freq, vs_data_fft[2], "r", linewidth=0.5)
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Normalize Amplitude (V)")
        axes[2].grid(linestyle=":", linewidth=0.5, color="k")
        axes[2].set_yscale("log")


        label = vs_data_each[1] + "\n" + "Vs: " + '{:.2f}'.format(vs_rise) + "m/s"
        axes[0].text(0.7, 0.1, label, fontsize=14, transform=axes[0].transAxes, bbox=dict(facecolor='w', edgecolor='k', pad=5.0, linewidth=0.75))

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

    dir_path = Path(r"E:\Shiga Dropbox\01_work\04_2021-_assistant professor\01_research\03_codes\04_Vs_analysis\consolidation")
    wave_dir = DataDir(path=dir_path, 
                       intial_accelerometer_distance=100, 
                       time_dif=0, 
                       stage="con", 
                       ratio_thres=0.15)
    
    wave_dir.analysis_vs_with_offset()


if __name__ == "__main__":
    main()