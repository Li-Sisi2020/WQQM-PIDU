"""
This is for scenario A1 analysis, discussing the impact of irrigation and drainage styles.
Written by Sisi Li,
on May 17, 2021
"""

import pandas as pd
import numpy as np
import main_runV2
import os

province = "Hubei"
folder = "wf_new1/{}/".format(province)
id_styles = ["totally decentralized", "partly centralized 1", "partly centralized 2", "totally centralized"]
TAWC_dic = {"Heilongjiang": 0.358, "Jilin": 0.362, "Jiangsu": 0.352, "Anhui": 0.348,
               "Hunan": 0.361, "Sichuan": 0.364, "Guizhou": 0.332, "Yunnan": 0.338,
               "Fujian": 0.346, "Guangdong": 0.340, "Guangxi": 0.316, "Hainan": 0.361,
               "Hubei": 0.350, "Jiangxi": 0.346, "Liaoning": 0.360, "Zhejiang": 0.350}
Irrg_rate_dic = {"Heilongjiang": [1.5, 1.6], "Jilin": [1.4, 1.5], "Jiangsu": [1.1, 1.3], "Anhui": [1.2, 1.5],
               "Hunan": [1.2, 1.4], "Sichuan": [0.6, 0.7], "Guizhou": [0.8, 0.9], "Yunnan": [0.5, 0.6],
               "Fujian": [0.8, 0.9], "Guangdong": [0.5, 0.55], "Guangxi": [0.6, 0.7], "Hainan": [0.4, 0.45],
               "Hubei": [1.1, 1.4], "Jiangxi": [0.9, 1.1], "Liaoning": [1.4, 1.7], "Zhejiang": [0.8, 1.0]}
province_dic = {"Heilongjiang": [50873, 50953, 50756], "Jilin": [54064, 54186, 54266], "Jiangsu": [58354, 58241, 58038], "Anhui": [58319, 58326, 58215],
               "Hunan": [57673, 57779, 57874], "Sichuan": [57328, 56492, 56571], "Guizhou": [57731, 57825, 57910], "Yunnan": [56778, 56751, 56991],
               "Fujian": [58737, 58926, 58818], "Guangdong": [59306, 59097, 59478], "Guangxi": [57957, 59446, 59228], "Hainan": [59758, 59845, 59855],
               "Hubei": [57494, 57395, 57476], "Jiangxi": [57793, 58606, 58806], "Liaoning": [54342, 54470, 54353], "Zhejiang": [58448, 58562, 58464]}

if ("1S" in province) | ("2S" in province):
    province1 = province[:-2]
else:
    province1 = province
TAWC = TAWC_dic[province1]
Irrg_rate = Irrg_rate_dic[province1]
# Create folders and create default parameters for each style
parameters = pd.read_json("{}data_config.json".format(folder), orient='index', typ='series', dtype=None)
station_info = pd.read_csv("input_data/station_info.csv", header=0, index_col=0, sep=",")
parameters.at["latitude"] = str(station_info.loc[province_dic[province1], "纬度"].mean().round(2))
parameters.at["altitude"] = str(station_info.loc[province_dic[province1], "观测场拔海高度（米）"].mean().round(2))


parameters_dic = {}
for style in id_styles:
    if not os.path.exists("{}/{}".format(folder, style)):
        os.makedirs("{}/{}".format(folder, style))
    parameters_style = parameters.copy(deep=True)
    parameters_style.at["output_dir"] = "{}/{}".format(folder, style)
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
        parameters1.at["vf_P"] = str(vf_P_pd[i])
        parameters_file = "{}{}/data_config{}.json".format(folder, style, str(i))
        parameters1.to_json(parameters_file, orient='index', index=True, indent=4)
        main_runV2.main(parameters_file, str(i), TAWC, Irrg_rate)

