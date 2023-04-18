# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pandas as pd
import numpy as np

def evapotranspiration_Hargreaves(T_max, T_min, latitude, day_of_year):
    """ This is the function to calculate the potential evapotranspiration by Hargreaves method"""
    import math
    T_mean = (T_max + T_min) / 2
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * day_of_year)
    sd = 0.409 * math.sin(2 * math.pi / 365 * day_of_year - 1.39)
    ws = math.acos(-math.tan(latitude * math.pi / 180) * math.tan(sd))
    Ra = 24 * 60 / math.pi * 0.0820 * dr * (ws * math.sin(latitude * math.pi / 180) * math.sin(sd) +
                                            math.cos(latitude * math.pi / 180) * math.cos(sd) * math.sin(ws))
    PET = 0.0023 * (T_mean + 17.8) * ((T_max - T_min) ** 0.5) * Ra / (2.5 - 0.00237 * T_mean)
    if PET < 0:
        PET = 0
    return PET


def evapotranspiration_FAO_PM(T_max, T_min, altitude, wind, wind_type, humidity,
                              day_of_year, latitude, solar, solar_type, albedo=0.23):
    """The function to calculate FAO Penman-Monteith potential evapotranspiration.
       Parameter wind_type is the type of wind data,
       1 表示10米风速， 一般国家基本、基准气象站数据，风速为地表10米处风速；
       2表示2米风速，一般自建微型/小型气象站，风速传感器安装在地表2米处。
       Parameter solar_type is the type of solar data,
       1表示日照时长，单位小时，一般来源于国家基本/基准气象站数据；
       2表示日均太阳辐射率，单位W/m2，一般来源于自建微型/小型气象站。
       """
    import math
    # Calculate parameters from temperature and altitude
    T_mean = (T_max + T_min) / 2
    P = 101.3 * (((293 - 0.0065 * altitude) / 293) ** 5.26)
    pschm_constant = 0.665 * 0.001 * P
    vpc_slope = 4098 * 0.6108 * math.exp(17.27 * T_mean / (T_mean + 237.3)) / ((T_mean + 237.3) ** 2)
    # Calculate vapor pressure deficit:
    e0_max = 0.6108 * math.exp(17.27 * T_max / (T_max + 237.3))
    e0_min = 0.6108 * math.exp(17.27 * T_min / (T_min + 237.3))
    e_sat = (e0_max + e0_min) / 2
    e_act = e_sat * humidity / 100.0
    e_deficit = e_sat - e_act
    # Calculate Radiation
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * day_of_year)
    sd = 0.409 * math.sin(2 * math.pi / 365 * day_of_year - 1.39)
    ws = math.acos(-math.tan(latitude * math.pi / 180) * math.tan(sd))
    Ra = 24 * 60 / math.pi * 0.0820 * dr * (ws * math.sin(latitude * math.pi / 180) * math.sin(sd) +
                                            math.cos(latitude * math.pi / 180) * math.cos(sd) * math.sin(ws))
    if solar_type == 1:
        N = 24 / math.pi * ws
        Rs = (0.25 + 0.50 * solar / N) * Ra
    elif solar_type == 2:
        Rs = solar * 3600 * 24 / 1000000  # Convert the unit from W/m2 to MJ/m2/day
    Rs0 = (0.75 + 0.00002 * altitude) * Ra
    Rns = (1 - albedo) * Rs
    Rs_rate = min(Rs / Rs0, 1.0)
    Rnl = 4.903 * 1e-09 * ((T_max + 273.16) ** 4 + (T_min + 273.16) ** 4) / 2 * (0.34 - 0.14 * (e_act ** 0.5)) * (
                1.35 * Rs_rate - 0.35)
    Rn = Rns - Rnl
    # Convert wind speed from 10m to 2m height
    if wind_type == 1:
        u2 = wind * 4.87 / math.log(67.8 * 10.0 - 5.42)
    elif wind_type == 2:
        u2 = wind
    # Calculate reference evapotranspiration from FAO Penman-Monteith equation
    PET = (0.408 * vpc_slope * Rn + pschm_constant * 900 / (T_mean + 273) * u2 * e_deficit) / (
                vpc_slope + pschm_constant * (1 + 0.34 * u2))
    if PET < 0:
        PET = 0
    return PET


def f_vhd(h, hDiff_FD, wd_up, wd_lw, l):
    """The depth-volume conversion function for the ditch
       h is the water depth in the ditch, in mm.
       hDiff_FD is the effective depth of the ditch, in mm,
       wd_up and wd_lw are the upper and lower widths of the ditch, in mm,
       l is the length of the ditch.
       The output is the volume in one segment of the ditch, in mm m2"""
    if h < 0:
        h = 0
    elif h > hDiff_FD:
        h = hDiff_FD
    wd = (wd_up - wd_lw) * h / hDiff_FD + wd_lw
    s = (wd_lw + wd) * h / 2  # in mm m
    v = s * l  # in mm m2
    return v


def f_vhp(h, hDiff_FP, area_p, slp_p):
    """The depth-volume conversion function for the pond.
       h is the water depth in the pond, in mm.
       The output is the volume in the pond, in mm m2"""
    import math
    if h < 0:
        h = 0
    elif h > hDiff_FP:
        h = hDiff_FP
    s_up = area_p  # in m2
    s_lw = (math.sqrt(s_up) - hDiff_FP / 1000 * slp_p * 2) ** 2  # in m2
    h1 = math.sqrt(s_lw) / 2 / slp_p  # in m
    s = s_lw * ((h1 + h / 1000) ** 2) / (h1 ** 2)  # in m2
    v = (s * (h1 + h / 1000) - s_lw * h1) / 3 * 1000  # in mm m2
    return v



def f_vhpd(h, v, h_v_sr, area_pd, h_yan):
    """The volume-depth conversion function for the ditch-pond system
       h is the water depth as in the first-order ditch, in mm.
       v is the increased water volume, in mm m2/d.
       h_v_sr is a pandas Series to store the paired depth-volume (H-V) relationship for the paddy IDU.
       area_pd is the total area of the paddy IDU, in m2.
       h_yan is the depth as in the 1st order ditch above which drainage out the system is naturally occur, in mm
       The output is a tuple (q_dp, hdp),
       where q_dp is the water flow out of the system, in m3/d,
       and hdp is the water depth in ditch-pond system as the value of depth in the first-order ditch, in mm"""
    h_min = h_v_sr.index.min()
    i = max(int(round(h, 0)), h_min)
    if h < h_min:
        current_volume = h_v_sr[h_min] + (h - h_min) * area_pd
    else:
        current_volume = h_v_sr[i]
    if v < 0:
        result_q_dp = 0  # in m3/d
        while h_v_sr[i] > (v + current_volume):
            if i > h_min:
                i -= 1
            else:
                i = h_min + (v + current_volume) / area_pd
                break
        result_hdp = float(i)
        del i
    else:
        max_volume = h_v_sr[h_yan] - current_volume
        if v > max_volume:
            result_q_dp = (v - max_volume) / 1000.0  # in m3/d
            result_hdp = h_yan  # in mm
        else:
            result_q_dp = 0  # in m3/d
            if (v + current_volume) < 0:
                result_hdp = (v + current_volume) / area_pd + h_min
            else:
                while h_v_sr[i] < (v + current_volume):
                    i += 1
                result_hdp = float(i)
                del i
    return (result_q_dp, result_hdp)


def draw_graphs(graphs_dfs, output_dir):
    """ This function draws default graphs for the simulated results, for visual show in GUI .
    Parameters:
        graphs_dfs, the pandas DataFrames that store data for drawing graphs, including:
        Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1,
        TN_cp_df1, TP_cp_df1, TN_fs_df1, TN_s_df1, TP_fs_df1, TP_s_df1, TN_in_out, TP_in_out.
        output_dir, the directory to store graphs.
    Outputs:
        The paths to three graphs.
    """
    Hf_df1 = graphs_dfs[0]
    Hdp_df1 = graphs_dfs[1]
    TN_cf_df1 = graphs_dfs[2]
    TP_cf_df1 = graphs_dfs[3]
    TN_cd1_df1 = graphs_dfs[4]
    TP_cd1_df1 = graphs_dfs[5]
    if isinstance(graphs_dfs[6], int):
        orderDitch = 1
    else:
        orderDitch = 2
        TN_cd2_df1 = graphs_dfs[6]
        TP_cd2_df1 = graphs_dfs[7]
    if isinstance(graphs_dfs[8], int):
        havePond = False
    else:
        havePond = True
        TN_cp_df1 = graphs_dfs[8]
        TP_cp_df1 = graphs_dfs[9]
    TN_fs_df1 = graphs_dfs[10]
    TN_s_df1 = graphs_dfs[11]
    TP_fs_df1 = graphs_dfs[12]
    TP_s_df1 = graphs_dfs[13]
    TN_in_out = graphs_dfs[14]
    TP_in_out = graphs_dfs[15]

    """ Drawing Graphs for visual results on the GUI """
    # 准备字体和负号
    myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 准备日期序列
    dates = pd.date_range(start="2018-" + TN_cf_df1.index[0], end="2018-" + TN_cf_df1.index[-1], freq="D")
    dates_doy = np.zeros(dates.size)
    for i, date in enumerate(dates):
        dates_doy[i] = date.dayofyear
    if dates[-1].month <= 11:
        end_date_plt = "2018-" + str(dates[-1].month + 1) + "-1"
    else:
        end_date_plt = "2019-1-1"
    dates_plt = pd.date_range(start="2018-" + str(dates[0].month) + "-1", end=end_date_plt, freq="MS")
    dates_doy_plt = np.zeros(dates_plt.size)
    for i, date in enumerate(dates_plt):
        dates_doy_plt[i] = date.dayofyear
    xtick_labels = list()
    for date in dates_plt:
        xtick_labels.append(str(date.month) + "-1")
    """ 田-沟-塘水量水质变化图 """
    # 重组long-term tidy DataFrame,以便绘图
    Hf_df2 = Hf_df1.iloc[:, 0]
    Hdp_df2 = Hdp_df1.iloc[:, 0]
    TN_cf_df2 = TN_cf_df1.iloc[:, 0]
    TN_cd1_df2 = TN_cd1_df1.iloc[:, 0]
    TP_cf_df2 = TP_cf_df1.iloc[:, 0]
    TP_cd1_df2 = TP_cd1_df1.iloc[:, 0]
    if orderDitch == 2:
        TN_cd2_df2 = TN_cd2_df1.iloc[:, 0]
        TP_cd2_df2 = TP_cd2_df1.iloc[:, 0]
    if havePond:
        TN_cp_df2 = TN_cp_df1.iloc[:, 0]
        TP_cp_df2 = TP_cp_df1.iloc[:, 0]
    for i in range(1, TN_cf_df1.shape[1]):
        Hf_df2 = pd.concat([Hf_df2, Hf_df1.iloc[:, i]], ignore_index=True)
        Hdp_df2 = pd.concat([Hdp_df2, Hdp_df1.iloc[:, i]], ignore_index=True)
        TN_cf_df2 = pd.concat([TN_cf_df2, TN_cf_df1.iloc[:, i]], ignore_index=True)
        TN_cd1_df2 = pd.concat([TN_cd1_df2, TN_cd1_df1.iloc[:, i]], ignore_index=True)
        TP_cf_df2 = pd.concat([TP_cf_df2, TP_cf_df1.iloc[:, i]], ignore_index=True)
        TP_cd1_df2 = pd.concat([TP_cd1_df2, TP_cd1_df1.iloc[:, i]], ignore_index=True)
        if orderDitch == 2:
            TN_cd2_df2 = pd.concat([TN_cd2_df2, TN_cd2_df1.iloc[:, i]], ignore_index=True)
            TP_cd2_df2 = pd.concat([TP_cd2_df2, TP_cd2_df1.iloc[:, i]], ignore_index=True)
        if havePond:
            TN_cp_df2 = pd.concat([TN_cp_df2, TN_cp_df1.iloc[:, i]], ignore_index=True)
            TP_cp_df2 = pd.concat([TP_cp_df2, TP_cp_df1.iloc[:, i]], ignore_index=True)
    H_ts = pd.concat([pd.DataFrame({"水位(mm)": Hf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
                      pd.DataFrame({"水位(mm)": Hdp_df2, "水体": "沟塘", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
                      ], ignore_index=True)
    if havePond & (orderDitch == 2):
        TN_ts = pd.concat(
            [pd.DataFrame({"TN浓度(mg/L)": TN_cf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TN浓度(mg/L)": TN_cd1_df2, "水体": "第一级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TN浓度(mg/L)": TN_cd2_df2, "水体": "第二级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TN浓度(mg/L)": TN_cp_df2, "水体": "塘", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
             ], ignore_index=True)
        TP_ts = pd.concat(
            [pd.DataFrame({"TP浓度(mg/L)": TP_cf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TP浓度(mg/L)": TP_cd1_df2, "水体": "第一级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TP浓度(mg/L)": TP_cd2_df2, "水体": "第二级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TP浓度(mg/L)": TP_cp_df2, "水体": "塘", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
             ], ignore_index=True)
    elif havePond & (orderDitch == 1):
        TN_ts = pd.concat(
            [pd.DataFrame({"TN浓度(mg/L)": TN_cf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TN浓度(mg/L)": TN_cd1_df2, "水体": "第一级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TN浓度(mg/L)": TN_cp_df2, "水体": "塘", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
             ], ignore_index=True)
        TP_ts = pd.concat(
            [pd.DataFrame({"TP浓度(mg/L)": TP_cf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TP浓度(mg/L)": TP_cd1_df2, "水体": "第一级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TP浓度(mg/L)": TP_cp_df2, "水体": "塘", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
             ], ignore_index=True)
    elif (havePond is False) & (orderDitch == 2):
        TN_ts = pd.concat(
            [pd.DataFrame({"TN浓度(mg/L)": TN_cf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TN浓度(mg/L)": TN_cd1_df2, "水体": "第一级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TN浓度(mg/L)": TN_cd2_df2, "水体": "第二级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
             ], ignore_index=True)
        TP_ts = pd.concat(
            [pd.DataFrame({"TP浓度(mg/L)": TP_cf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TP浓度(mg/L)": TP_cd1_df2, "水体": "第一级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TP浓度(mg/L)": TP_cd2_df2, "水体": "第二级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
             ], ignore_index=True)
    elif (havePond is False) & (orderDitch == 1):
        TN_ts = pd.concat(
            [pd.DataFrame({"TN浓度(mg/L)": TN_cf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TN浓度(mg/L)": TN_cd1_df2, "水体": "第一级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
             ], ignore_index=True)
        TP_ts = pd.concat(
            [pd.DataFrame({"TP浓度(mg/L)": TP_cf_df2, "水体": "田", "date": np.tile(dates_doy, TN_cf_df1.shape[1])}),
             pd.DataFrame({"TP浓度(mg/L)": TP_cd1_df2, "水体": "第一级沟", "date": np.tile(dates_doy, TN_cf_df1.shape[1])})
             ], ignore_index=True)
    # 绘制时间序列图
    sns.set(style="ticks", palette="colorblind", font=myfont.get_name())
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.0})
    f = plt.figure(figsize=(8, 9))
    ax1 = f.add_subplot(311)
    ax2 = f.add_subplot(312)
    ax3 = f.add_subplot(313)
    sns.lineplot(x="date", y="水位(mm)", data=H_ts, style="水体", hue="水体", estimator="mean", ci=99,
                 err_style="band", ax=ax1)
    ax1.set_xticks(dates_doy_plt)
    ax1.set_xticklabels([])
    ax1.set_xlabel("")
    sns.lineplot(x="date", y="TN浓度(mg/L)", data=TN_ts, style="水体", hue="水体", estimator="mean", ci=99,
                 err_style="band", ax=ax2)
    ax2.set_xticks(dates_doy_plt)
    ax2.set_xticklabels([])
    ax2.set_xlabel("")
    sns.lineplot(x="date", y="TP浓度(mg/L)", data=TP_ts, style="水体", hue="水体", estimator="mean", ci=99,
                 err_style="band", ax=ax3)
    ax3.set_xticks(dates_doy_plt)
    ax3.set_xticklabels(xtick_labels)
    ax3.set_xlabel("日期", fontproperties=myfont)
    plt.tight_layout()
    f.savefig(output_dir+"/水量水质时间序列图.png")

    """ 田块/灌排单元地表氮磷流失量对比图 """
    # 重组long-term tidy DataFrame,以便绘图
    TN_f1 = TN_fs_df1.iloc[:, 0]
    TN_sys1 = TN_s_df1.iloc[:, 0]
    TP_f1 = TP_fs_df1.iloc[:, 0]
    TP_sys1 = TP_s_df1.iloc[:, 0]
    for i in range(1, TN_fs_df1.shape[1]):
        TN_f1 = pd.concat([TN_f1, TN_fs_df1.iloc[:, i]], ignore_index=True)
        TN_sys1 = pd.concat([TN_sys1, TN_s_df1.iloc[:, i]], ignore_index=True)
        TP_f1 = pd.concat([TP_f1, TP_fs_df1.iloc[:, i]], ignore_index=True)
        TP_sys1 = pd.concat([TP_sys1, TP_s_df1.iloc[:, i]], ignore_index=True)
    TN_load_ts = pd.concat(
        [pd.DataFrame({"TN负荷(kg/ha)": TN_f1, "尺度": "田块", "date": np.tile(dates_doy, TN_fs_df1.shape[1])}),
         pd.DataFrame({"TN负荷(kg/ha)": TN_sys1, "尺度": "灌排单元", "date": np.tile(dates_doy, TN_fs_df1.shape[1])})
         ], ignore_index=True)
    TP_load_ts = pd.concat(
        [pd.DataFrame({"TP负荷(kg/ha)": TP_f1, "尺度": "田块", "date": np.tile(dates_doy, TN_fs_df1.shape[1])}),
         pd.DataFrame({"TP负荷(kg/ha)": TP_sys1, "尺度": "灌排单元", "date": np.tile(dates_doy, TN_fs_df1.shape[1])})
         ], ignore_index=True)
    # 绘制时间序列图
    sns.set(style="ticks", palette="colorblind", font=myfont.get_name())
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.0})
    f = plt.figure(figsize=(8, 6))
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)
    sns.lineplot(x="date", y="TN负荷(kg/ha)", data=TN_load_ts, style="尺度", hue="尺度", estimator="mean",
                 ci=99, err_style="band", ax=ax1)
    ax1.set_xticks(dates_doy_plt)
    ax1.set_xticklabels([])
    ax1.set_xlabel("")
    sns.lineplot(x="date", y="TP负荷(kg/ha)", data=TP_load_ts, style="尺度", hue="尺度", estimator="mean",
                 ci=99, err_style="band", ax=ax2)
    ax2.set_xticks(dates_doy_plt)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_xlabel("日期", fontproperties=myfont)
    plt.tight_layout()
    f.savefig(output_dir+"/田块_灌排单元地表氮磷流失量对比图.png")

    """ 氮磷流入流出负荷图 """
    # 重组long-term tidy Dataframe,以便绘图
    TN_in_out.loc[TN_in_out["scale"] == "field", "scale"] = "田块"
    TN_in_out.loc[TN_in_out["scale"] == "IDU", "scale"] = "灌排单元"
    TP_in_out.loc[TP_in_out["scale"] == "field", "scale"] = "田块"
    TP_in_out.loc[TP_in_out["scale"] == "IDU", "scale"] = "灌排单元"
    TN_in_out1 = pd.concat([
        pd.DataFrame({"TN输出量(kg/ha)": -TN_in_out["irrigation input"], "尺度": TN_in_out["scale"], "class": "灌溉"}),
        pd.DataFrame({"TN输出量(kg/ha)": -TN_in_out["precipitation input"], "尺度": TN_in_out["scale"], "class": "降水"}),
        pd.DataFrame({"TN输出量(kg/ha)": TN_in_out["runoff loss"], "尺度": TN_in_out["scale"], "class": "径流"}),
        pd.DataFrame({"TN输出量(kg/ha)": TN_in_out["leaching loss"], "尺度": TN_in_out["scale"], "class": "淋溶"}),
        pd.DataFrame({"TN输出量(kg/ha)": TN_in_out["net export"], "尺度": TN_in_out["scale"], "class": "净输出"})],
        ignore_index=True)
    TP_in_out1 = pd.concat([
        pd.DataFrame({"TP输出量(kg/ha)": -TP_in_out["irrigation input"], "尺度": TP_in_out["scale"], "class": "灌溉"}),
        pd.DataFrame({"TP输出量(kg/ha)": -TP_in_out["precipitation input"], "尺度": TP_in_out["scale"], "class": "降水"}),
        pd.DataFrame({"TP输出量(kg/ha)": TP_in_out["runoff loss"], "尺度": TP_in_out["scale"], "class": "径流"}),
        pd.DataFrame({"TP输出量(kg/ha)": TP_in_out["leaching loss"], "尺度": TP_in_out["scale"], "class": "淋溶"}),
        pd.DataFrame({"TP输出量(kg/ha)": TP_in_out["net export"], "尺度": TP_in_out["scale"], "class": "净输出"})],
        ignore_index=True)
    # 绘制流入流出负荷图
    sns.set(style="darkgrid", palette="colorblind", font=myfont.get_name())
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.0})
    f = plt.figure(figsize=(6, 9))
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)
    ax1 = sns.barplot(y="class", x="TN输出量(kg/ha)", hue="尺度", data=TN_in_out1, ci="sd", ax=ax1)
    ax2 = sns.barplot(y="class", x="TP输出量(kg/ha)", hue="尺度", data=TP_in_out1, ci="sd", ax=ax2)
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    plt.tight_layout()
    f.savefig(output_dir+"/氮磷流入流出负荷图.png")

    return (output_dir+"/水量水质时间序列图.png", output_dir+"/田块_灌排单元地表氮磷流失量对比图.png",
            output_dir+"/氮磷流入流出负荷图.png")

