"""
This is to analyze the impact of irrigation and drainage styles for Hubei province, a typical Changjiang one-season rice-production province in China.
Written by Sisi Li,
on May 17, 2021
"""

import pandas as pd
import numpy as np
import main_run
import os

folder = "wf_new/Hubei/"
id_styles = ["totally decentralized", "partly centralized 1", "partly centralized 2", "totally centralized"]

# Create folders and create default parameters for each style
parameters = pd.read_json("{}data_config.json".format(folder), orient='index', typ='series', dtype=None)
station_info = pd.read_csv("input_data/station_info.csv", header=0, index_col=0, sep=",")
province_dic = {"Heilongjiang": [50873, 50953, 50756], "Jilin": [54064, 54186, 54266], "Jiangsu": [58354, 58241, 58038], "Anhui": [58319, 58326, 58215],
               "Hunan": [57673, 57779, 57874], "Sichuan": [57328, 56492, 56571], "Guizhou": [57731, 57825, 57910], "Yunnan": [56778, 56751, 56991],
               "Fujian": [58737, 58926, 58818], "Guangdong": [59306, 59097, 59478], "Guangxi": [57957, 59446, 59228], "Hainan": [59758, 59845, 59855],
               "Hubei": [57494, 57395, 57476], "Jiangxi": [57793, 58606, 58806], "Liaoning": [54342, 54470, 54353], "Zhejiang": [58448, 58562, 58464]}
parameters.at["latitude"] = str(station_info.loc[province_dic["Hubei"], "纬度"].mean().round(2))
parameters.at["altitude"] = str(station_info.loc[province_dic["Hubei"], "观测场拔海高度（米）"].mean().round(2))

parameters_dic = {}
for style in id_styles:
    if not os.path.exists("{}{}".format(folder, style)):
        os.makedirs("{}{}".format(folder, style))
    parameters_style = parameters.copy(deep=True)
    parameters_style.at["output_dir"] = "{}{}".format(folder, style)
    if style == "totally decentralized":
        parameters_style.at["recycle_irrg"] = "1"
        parameters_dic[style] = parameters_style
    elif style == "totally centralized":
        parameters_style.at["havePond"] = "False"
        parameters_dic[style] = parameters_style
    elif style == "partly centralized 1":
        parameters_style.at["havePond"] = "False"
        parameters_style.at["recycle_irrg"] = "1"
        parameters_dic[style] = parameters_style
    elif style == "partly centralized 2":
        parameters_dic[style] = parameters_style


# Read in random nutrient removal parameters
vf_N = pd.read_csv("input_data/vf_Nf_CJ_MC100.csv", sep=",", header=0, index_col=0)
vf_N_pd = np.loadtxt("input_data/vf_N_MC100.csv")
vf_P_pd = np.loadtxt("input_data/vf_P_MC100.csv")
ENC0_pd = np.loadtxt("input_data/ENC0_MC100.csv")
EPC0_pd = np.loadtxt("input_data/EPC0_MC100.csv")

# Create json file for each scenario and run model
for i in range(0, 100, 1):
    for style in id_styles:
        parameters1 = parameters_dic[style].copy(deep=True)
        parameters1.at["vf_N_pd"] = str(vf_N_pd[i])
        parameters1.at["vf_P_pd"] = str(vf_P_pd[i])
        parameters1.at["ENC0_dp"] = str(ENC0_pd[i])
        parameters1.at["EPC0_dp"] = str(EPC0_pd[i])
        parameters1.at["vf_N"] = ", ".join(str(x) for x in vf_N.loc[i].values)
        parameters1.at["vf_P"] = str(vf_P_pd[i] * 0.7)
        parameters_file = "{}{}/data_config{}.json".format(folder, style, str(i))
        parameters1.to_json(parameters_file, orient='index', index=True, indent=4)
        main_run.main(parameters_file, str(i))

