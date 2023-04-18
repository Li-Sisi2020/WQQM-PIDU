# -*- coding:utf-8 -*-
import json
import os
from prettytable import PrettyTable

def main(parameter_file, run_id, TAW=None, Irrg_rate=None):
    """ 参数录入 """
    print(parameter_file)
    with open(parameter_file, 'rb+') as cf:
        data = cf.read().decode('utf-8')
    data = json.loads(data)
    """ 第一页，灌排单元结构相关参数 """
    # print("请输入稻田灌排单元结构相关参数\n")
    # 是否有塘（有，输入1；无，输入0）：
    if data["havePond"] == "True":
        havePond = True
    else:
        havePond = False
    # 排水沟渠级数（仅限1或2）：
    orderDitch = int(data["orderDitch"])
    # 田块面积（公顷ha）：
    area_f = float(data["area_f"])

    if orderDitch == 1:
        # print("\n请输入第一级排水沟渠的参数\n")
        # 总长度（m）：
        Ld1 = float(data["Ld1"])
        # 沟深（mm）：
        HDiff_FD1 = float(data["HDiff_FD1"])
        # 上底宽（m）：
        Wd1_up = float(data["Wd1_up"])
        # 下底宽（m）：
        Wd1_lw = float(data["Wd1_lw"])
        # 数量（条）：
        n1 = int(data["n1"])  # 参与计算的排水沟渠的条数
        # 底部坡降（无量纲，如0.001）：
        slp_d1 = float(data["slp_d1"])  # 沟底沿程坡降，垂直高差与水平距离的比值，默认值为0.001
        # 曼宁粗糙系数（无量纲，如0.020）：
        n_d1 = float(data["n_d1"])  # 默认值为0.020
        # 植被/作物系数（用于蒸散发计算）：
        kc_d1 = float(data["kc_d1"])  # 植被覆盖度越大，蒸发系数越大，范围：1.05-1.30，无植被覆盖取1.05。默认值为1.05
        # 初始化TN浓度（mg/L）：
        TN_cd10 = float(data["TN_cd10"])  # 水稻生育期开始前第一级排水沟渠的初始TN浓度，单位mg/L。默认值为1.50，即地表水IV类标准。
        # 初始化TP浓度（mg/L）：
        TP_cd10 = float(data["TP_cd10"])  # 水稻生育期开始前第一级排水沟渠的初始TP浓度，单位mg/L。默认值为0.30，即地表水IV类标准。
        # 水位初始值（mm）：
        Hd10 = float(data["Hd10"])  # 水稻生育期开始之前，第一级沟渠水位初始值，单位mm。默认值50.0
    else:
        # print("\n请输入第一级排水沟渠的参数\n")
        # 沟间距（m）：
        Wf1 = float(data["Wf1"])
        # 沟深（mm）：
        HDiff_FD1 = float(data["HDiff_FD1"])
        # 上底宽（m）：
        Wd1_up = float(data["Wd1_up"])
        # 下底宽（m）：
        Wd1_lw = float(data["Wd1_lw"])
        # 数量（条）：
        n1 = int(data["n1"])  # 参与计算的第一级排水沟渠的条数
        # 底部坡降（无量纲，如0.001）：
        slp_d1 = float(data["slp_d1"])  # 沟底沿程坡降，垂直高差与水平距离的比值，默认值为0.001
        # 曼宁粗糙系数（无量纲，如0.020）：
        n_d1 = float(data["n_d1"])  # 默认值为0.020
        # 植被/作物系数（用于蒸散发计算）：
        kc_d1 = float(data["kc_d1"])  # 植被覆盖度越大，蒸发系数越大，范围：1.05-1.30，无植被覆盖取1.05。默认值为1.05
        # 初始化TN浓度（mg/L）：
        TN_cd10 = float(data["TN_cd10"])  # 水稻生育期开始前第一级排水沟渠的初始TN浓度，单位mg/L。默认值为1.50，即地表水IV类标准。
        # 初始化TP浓度（mg/L）：
        TP_cd10 = float(data["TP_cd10"])  # 水稻生育期开始前第一级排水沟渠的初始TP浓度，单位mg/L。默认值为0.30，即地表水IV类标准。
        # 水位初始值（mm）：
        Hd10 = float(data["Hd10"])  # 水稻生育期开始之前，第一级沟渠水位初始值，单位mm。默认值50.0
        # print("\n请输入第二级排水沟渠的参数\n")
        # 沟间距（m）：
        Wf2 = float(data["Wf2"])
        # 沟深（mm）：
        HDiff_FD2 = float(data["HDiff_FD2"])
        # 上底宽（m）：
        Wd2_up = float(data["Wd2_up"])
        # 下底宽（m）：
        Wd2_lw = float(data["Wd2_lw"])
        # 数量（条）：
        n2 = int(data["n2"])  # 参与计算的第二级排水沟渠的条数
        # 底部坡降（无量纲，如0.001）：
        slp_d2 = float(data["slp_d2"])  # 沟底沿程坡降，垂直高差与水平距离的比值，默认值为0.001
        # 曼宁粗糙系数（无量纲，如0.020）：
        n_d2 = float(data["n_d2"])  # 默认值为0.020
        # 植被/作物系数（用于蒸散发计算）：
        kc_d2 = float(data["kc_d2"])  # 植被覆盖度越大，蒸发系数越大，范围：1.05-1.30，无植被覆盖取1.05。默认值为1.05
        # 初始化TN浓度（mg/L）：
        TN_cd20 = float(data["TN_cd20"])  # 水稻生育期开始前第二级排水沟渠的初始TN浓度，单位mg/L。默认值为1.50，即地表水IV类标准。
        # 初始化TP浓度（mg/L）：
        TP_cd20 = float(data["TP_cd20"])  # 水稻生育期开始前第二级排水沟渠的初始TP浓度，单位mg/L。默认值为0.30，即地表水IV类标准。
    if havePond:
        # print("\n请输入水塘的参数\n")
        # 塘田面积比（无量纲，如0.0191）：
        rate_pond = float(data["rate_pond"])
        # 有效深度（mm）：
        HDiff_FP = float(data["HDiff_FP"])
        # 边坡系数（边坡的水平距离与垂直高差的比值，如1.5）：
        slp_p = float(data["slp_p"])
        # 植被/作物系数（用于蒸散发计算）：
        kc_p = float(data["kc_p"])  # 植被覆盖度越大，蒸发系数越大，范围：1.05-1.30，无植被覆盖取1.05。默认值为1.05
        # 初始化TN浓度（mg/L）：
        TN_cp0 = float(data["TN_cp0"])  # 水稻生育期开始前，水塘的初始TN浓度，单位mg/L。默认值为1.50，即地表水IV类标准。
        # 初始化TP浓度（mg/L）：
        TP_cp0 = float(data["TP_cp0"])  # 水稻生育期开始前，水塘的初始TP浓度，单位mg/L。默认值为0.10，即地表水IV类标准（湖库）。

    """ 第二页，水稻生育周期及水肥管理 """
    # print("\n请输入水稻生育周期及水肥管理信息\n")
    # 水稻生育期起始日期（格式：5-22）：
    startDay = data["startDay"]  # 水稻生育期的起始日期，一般是施基肥或泡田当天。若种植双季稻，则输入早稻的起始日期。
    # 水稻生育期结束日期（格式：10-1）：
    endDay = data["endDay"]  # 水稻生育期的结束日期，一般是收获当天。若种植双季稻，则输入晚稻的结束日期。

    # print("\n施肥日期及田面水初始浓度\n")
    # 施肥日期序列（多次施肥日期之间用英文逗号隔开，如 5-22,6-9）：
    F0 = data["F0"]

    # 施肥种类（基肥，输入1；追肥，输入2；多次施肥之间用英文逗号隔开，如1,2）：
    fert_type = data["fert_type"]  # 基肥，输入1；追肥，输入2；GUI中可以“基肥/追肥”选项选择,未施肥日期填入0。

    # 施肥当天田面水TN初始浓度（多次施肥的初始浓度之间用英文逗号隔开，如 28.0，16.9）：
    TN0 = data["TN0"]  # 未施肥日期的初始浓度默认填入0.0

    # 施肥当天田面水TN初始浓度（多次施肥的初始浓度之间用英文逗号隔开，如 2.0，0.0）：
    TP0 = data["TP0"]  # 未施肥日期的初始浓度默认填入0.0

    F_dates = list()
    for F_date in F0.split(","):
        # 处理没有施肥时的空数据
        F_dates.append(F_date if F_date != "" else "NA")
    for x in range(6 - len(F0.split(","))):
        F_dates.append("NA")

    fert_type_list = list()
    for fertT in fert_type.split(","):
        fert_type_list.append(0 if fertT == "" else int(fertT))
    for x in range(6 - len(fert_type.split(","))):
        fert_type_list.append(0)

    TN0_list = list()
    for TN0_str in TN0.split(","):
        TN0_list.append(0.0 if TN0_str == "" else float(TN0_str))
    for x in range(6 - len(F0.split(","))):
        TN0_list.append(0.0)

    TP0_list = list()
    for TP0_str in TP0.split(","):
        TP0_list.append(0.0 if TP0_str == "" else float(TP0_str))
    for x in range(6 - len(F0.split(","))):
        TP0_list.append(0.0)


    del F0
    del TN0
    del TP0

    # print("\n水稻生育期水位管理\n")
    # 请输入水位管理的文件路径：
    wtr_mng_File = data["wtr_mng_File"]  # 采用三种水位管理模式模拟稻田灌溉排水。

    # 田面水位小于最低适宜水位（H_min）时灌溉，灌溉至最高适宜水位（H_max）；田面水位高于耐淹水位（H_p）时排水。
    # 文件是以TAB分隔的文本数据（.txt），第一列（date）是日期，中间三列是上述三种水位（单位mm）。
    # 最后一列Kc是作物系数，用于计算潜在蒸散发，一般泡田期为1.05，返青期开始逐渐增加，至晒田后抽穗开花期稳定在1.20，
    # 后成熟期开始降低，至收获时约为0.90。参见示例数据。

    """ 第三页，气象和土壤水文参数 """
    # print("\n请输入气象数据的文件路径\n")

    # 降水量：
    prcp_File = data["prcp_File"]  # 日降水量时间序列数据，单位mm，以TAB分隔的文本文件（.txt）。
    # 水稻生育期内不允许数据缺失。参见示例数据。
    # 最高气温：
    tem_max_File = data["tem_max_File"]  # 日最高气温时间序列数据，单位摄氏度，以TAB分隔的文本文件（.txt）。
    # 水稻生育期内不允许数据缺失。参见示例数据。
    # 最低气温：
    tem_min_File = data["tem_min_File"]  # 日最低气温时间序列数据，单位摄氏度，以TAB分隔的文本文件（.txt）。
    # 水稻生育期内不允许数据缺失。参见示例数据。
    # 相对湿度（若无该数据，输入0）：
    humidity_File = data["humidity_File"]  # 日平均相对湿度时间序列，单位%，以TAB分隔的文本文件（.txt）。
    # 若没有该数据，可输入0；允许数据缺失，缺失值表示为-99.9。
    # 平均风速（若无该数据，输入0）：
    wind_File = data["wind_File"]  # 日平均风速时间序列，单位m/s，以TAB分隔的文本文件（.txt）。

    # 若没有该数据，可输入0；允许数据缺失，缺失值表示为-99.9。
    if os.path.isfile(wind_File):
        # 风速数据类型（10米风速，输入1；2米风速，输入2）：
        wind_type = int(data["wind_type"])  # 风速数据类型，根据实际情况选择。1表示10米风速，
        #  一般国家基本、基准气象站数据，风速为地表10米处风速；2表示2米风速，一般自建微型/小型气象站，风速传感器安装在地表2米处。
    else:
        wind_type = None

    # 日照数据（若无该数据，输入0）：
    solar_File = data["solar_File"]  # 日照数据时间序列，可以是日照时长，也可以是日均太阳辐射率，以TAB分隔的文本文件（.txt）。

    # 若没有该数据，可输入0；允许数据缺失，缺失值表示为-99.9。
    if os.path.isfile(solar_File):
        # 日照数据类型（日照时长，输入1；日均太阳辐射率，输入2）：
        solar_type = int(data["solar_type"])  # 日照数据类型，根据实际情况选择。
    else:
        solar_type = None
    # 1表示日照时长，单位小时，一般来源于国家基本/基准气象站数据；2表示日均太阳辐射率，单位W/m2，一般来源于自建微型/小型气象站。

    # 灌排单元所在地纬度（单位：度，如31.34）：
    latitude = float(data["latitude"])  # 灌排单元所在地纬度，单位：度。例如：31.34
    # 灌排单元所在地海拔高程（单位：米）：
    altitude = float(data["altitude"])  # 灌排单元所在地海拔高程，单位：米。

    # print("\n请输入土壤水文参数\n")
    # 田面水的深层下渗率（mm/d）：
    Fc0_f = float(data["Fc0_f"])  # 田面水的深层下渗率，单位mm/d。不包括到沟渠的侧渗量，推荐值0-5.0，默认值2.5
    # 沟塘水的深层下渗率（mm/d）：
    Fc0_pd = float(data["Fc0_pd"])  # 沟塘水的深层下渗率，单位mm/d。推荐值0-5.0，默认值2.5
    # 土壤饱和水力传导系数（m/d）：
    Ks = float(data["Ks"])  # 土壤饱和水力传导系数，单位m/d。默认值0.658

    """ 第四页，氮磷去除和含量分布相关"""
    # print("\n请输入氮磷去除动力学相关参数\n")
    # 田面水的TN去除速率（m/d）：
    vf_N = data["vf_N"]  # 田面水的TN去除速率，单位m/d。推荐值0.010-0.050，默认值0.022,
    # 基肥，孽肥，穗肥等可输入不同值；vf为与T0等长的数组。
    vf_N_list = list()
    for vf_N_str in vf_N.split(","):
        vf_N_list.append(0.0 if vf_N_str == "" else float(vf_N_str))
    for x in range(6 - len(vf_N.split(","))):
        vf_N_list.append(0.0)
    del vf_N

    # 田面水的TP去除速率（m/d）：
    vf_P = float(data["vf_P"])  # 田面水的TP去除速率，单位m/d。推荐值0.010-0.250，默认值0.049
    # 沟塘水的TN去除速率（m/d）：
    vf_N_pd = float(data["vf_N_pd"])  # 沟塘水的TN去除速率，单位m/d。推荐值0.010-1.000，默认值0.110
    # 沟塘水的TP去除速率（m/d）：
    vf_P_pd = float(data["vf_P_pd"])  # 沟塘水的TP去除速率，单位m/d。推荐值0.010-1.000，默认值0.049
    # 田面水TN平衡浓度比（无量纲）：
    ENCR_f = float(data["ENCR_f"])  # 田面水TN平衡浓度比，无量纲。田面水TN平衡浓度和施肥日初始浓度的比值，范围0-1.0， 默认值0.12
    # 田面水TP平衡浓度比（无量纲）：
    EPCR_f = float(data["EPCR_f"])  # 田面水TP平衡浓度比，无量纲。田面水TP平衡浓度和施肥日初始浓度的比值，范围0-1.0， 默认值0.20
    # 沟塘水TN平衡浓度（mg/L）：
    ENC0_dp = float(data["ENC0_dp"])  # 沟塘水TN平衡浓度，单位mg/L。推荐值0.10-2.00，默认值1.00
    # 沟塘水TP平衡浓度（mg/L）：
    EPC0_dp = float(data["EPC0_dp"])  # 沟塘水TP平衡浓度，单位mg/L。推荐值0.01-0.20，默认值0.10

    # print("\n请输入降水和灌溉水的氮磷浓度\n")
    # 灌溉水的TN浓度（mg/L）：
    TN_I = float(data["TN_I"])  # 灌溉水的TN浓度，单位mg/L。默认值为1.50，即地表水IV类标准。
    # 灌溉水的TP浓度（mg/L）：
    TP_I = float(data["TP_I"])  # 灌溉水的TP浓度，单位mg/L。默认值为0.10，即地表水IV类标准。
    # 施肥后15天以内降水的TN浓度（mg/L）：
    TN_P0 = float(data["TN_P0"])  # 施肥后15天以内降水的TN浓度，单位mg/L。降水中TN浓度和区域施肥量、氨挥发相关，
    # 因此区分施肥后15天以内和以外，原则上施肥后15天以内浓度更高。默认值2.50
    # 施肥后15天以外降水的TN浓度（mg/L）：
    TN_P1 = float(data["TN_P1"])  # 施肥后15天以外降水的TN浓度，单位mg/L。降水中TN浓度和区域施肥量、氨挥发相关，
    # 因此区分施肥后15天以内和以外，原则上施肥后15天以外浓度更低。默认值2.00
    # 降水的TP浓度（mg/L）：
    TP_P = float(data["TP_P"])  # 降水的TP浓度，单位mg/L。默认值0.15

    # print("\n请输入渗漏水与地表水的浓度比例\n")
    # 下渗水TN浓度比（无量纲）：
    TN_rate_lch = float(data["TN_rate_lch"])  # 下渗水TN浓度比，无量纲。下渗水中TN浓度和地表水的比值，默认值0.20
    # 下渗水TP浓度比（无量纲）：
    TP_rate_lch = float(data["TP_rate_lch"])  # 下渗水TP浓度比，无量纲。下渗水中TP浓度和地表水的比值，默认值0.10
    # 侧渗水TN浓度比（无量纲）：
    TN_rate_lch_lat = float(data["TN_rate_lch_lat"])  # 侧渗水TN浓度比，无量纲。下渗水中TN浓度和地表水的比值，默认值0.50
    # 侧渗水TP浓度比（无量纲）：
    TP_rate_lch_lat = float(data["TP_rate_lch_lat"])  # 侧渗水TP浓度比，无量纲。下渗水中TP浓度和地表水的比值，默认值0.20

    """ 第五页，结果文件夹和模式选择"""
    # 请选择结果文件夹（默认输入“”）：
    output_dir = data["output_dir"]  # 请选择结果存放的文件夹，默认在程序运行所在的工作文件夹中新建一个results文件夹。
    if output_dir == "":
        output_dir = os.getcwd() + "/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 管理措施模式选择。常规（必选项）：“精准水位控制”；强化（可选项）：“风险期提前排水”和“循环灌溉”
    # 精准水位控制。说明：1. 田块排水采取精准水位控制，不多余排水；2. 田块排水优先蓄积在沟塘中，沟塘蓄满方对外排水。
    # 风险期提前排水。说明：风险期内，田块主动排水或遇大雨及以上降雨预报时，提前1天排放沟塘水至调控目标水位。目标水位见结果文件water_level_control.txt。
    # 注意：同一个风险期，提前排水最多只进行一次操作。

    # 是否风险期提前排水？（是输入1，否输入0）：
    pre_risk = int(data["pre_risk"])  # 圆点勾选

    if pre_risk == 1:
        # 风险期天数为施肥后多少天？（7或15）：
        risk_days = int(data["risk_days"])  # 可选择7天或15天
        min_level_prerisk = int(data["min_level_prerisk"])
        # 可能是沟、塘的生态最低水位，也可能是养殖水塘所需的最低水位。

    # 循环灌溉。说明：需灌溉时，优先取用沟、塘存水循环灌溉农田，沟、塘存水不足时再使用外部水源。
    # 是否循环灌溉？（是输入1，否输入0）：
    recycle_irrg = int(data["recycle_irrg"])  # 圆点勾选
    if recycle_irrg == 1:
        # 取水所致的最低保证水位（单位mm）：
        min_level_recycle = int(data["min_level_recycle"])  # 设置因循环灌溉取水所致的灌排单元末端水体的最低保证水位，单位mm。
        # 可能是沟、塘的生态最低水位，也可能是养殖水塘所需的最低水位。

    # 其他选项
    # 自动作图？（是输入1，否输入0）：
    graph = int(data["graph"])  # 说明：是否就当前选定管理措施下的水量水质变化及氮磷流失负荷自动作图？方式：圆点勾选
    if (pre_risk == 1) | (recycle_irrg == 1):  # 说明：是否评价与常规措施（精准水位控制）相比，所选强化管理措施的氮磷减排效果？
        # 方式：当上述两个强化措施至少有一个勾选时，可选，原点勾选。
        # 强化措施效果评价？（是如输入1， 否输入0）：
        bmp_compare = int(data["bmp_compare"])

    print("参数录入完毕")
    """ 参数录入完毕，开始运算"""
    import sys
    import pandas as pd
    import numpy as np
    import sd_functions
    import modulesV2

    # Read in water management data
    wtr_mng_d = pd.read_csv(wtr_mng_File, sep='\t', header=0, index_col=0, parse_dates=True)
    # 水分管理数据有缺失，退出运算，要求重新输入
    if wtr_mng_d.loc[startDay:endDay, :].isnull().values.any():
        sys.exit("水分管理数据中有缺失，请检查修改后重新运算！\n")
    # Read in climate data
    prcp_d = pd.read_csv(prcp_File, sep='\t', header=0, index_col=0, parse_dates=True, na_values=-99.9)
    tem_max_d = pd.read_csv(tem_max_File, sep='\t', header=0, index_col=0, parse_dates=True, na_values=-99.9)
    tem_min_d = pd.read_csv(tem_min_File, sep='\t', header=0, index_col=0, parse_dates=True, na_values=-99.9)
    # 降雨或气温数据有缺失时，退出运算，要求重新输入气象数据。
    yearStart = prcp_d.index[0].year
    yearEnd = prcp_d.index[-1].year + 1
    index = pd.MultiIndex.from_product([[year for year in range(yearStart, yearEnd)], prcp_d.columns],
                                       names=['year', 'station'])
    prcp_d1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    tem_max_d1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    tem_min_d1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    for year in range(yearStart, yearEnd):
        prcp_d1[year] = prcp_d.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
        tem_max_d1[year] = tem_max_d.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
        tem_min_d1[year] = tem_min_d.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values

    if prcp_d1.isnull().values.any():
        sys.exit("降雨数据中有缺失，请检查修改后重新运算！\n")
    if tem_max_d1.isnull().values.any():
        sys.exit("最高气温数据中有缺失，请检查修改后重新运算！\n")
    if tem_min_d1.isnull().values.any():
        sys.exit("最低气温数据中有缺失，请检查修改后重新运算！\n")

    if os.path.isfile(humidity_File):
        humidity_d = pd.read_csv(humidity_File, sep='\t', header=0, index_col=0, parse_dates=True, na_values=-99)
    else:
        humidity_d = 0
    if os.path.isfile(wind_File):
        wind_d = pd.read_csv(wind_File, sep='\t', header=0, index_col=0, parse_dates=True, na_values=-99.9)
    else:
        wind_d = 0
    if os.path.isfile(solar_File):
        solar_d = pd.read_csv(solar_File, sep='\t', header=0, index_col=0, parse_dates=True, na_values=-99.9)
    else:
        solar_d = 0

    # Convert dates
    F_doy_array = np.zeros(len(F_dates), dtype=np.int16)
    for i, item in enumerate(F_dates):
        if item != 'NA':
            F_doy_array[i] = pd.Timestamp(2018, int(item.split('-')[0]), int(item.split('-')[1])).dayofyear
    fert_type_array = np.array(fert_type_list)
    TN0_array = np.array(TN0_list)
    TP0_array = np.array(TP0_list)
    vf_N_array = np.array(vf_N_list)
    yearStart = prcp_d.index[0].year
    yearEnd = prcp_d.index[-1].year + 1

    # Calculate PET
    pet_d = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns, dtype=np.float32)
    for column in prcp_d.columns:
        for date in prcp_d.index:
            if (not os.path.isfile(wind_File)) | (not os.path.isfile(humidity_File)) | (not os.path.isfile(solar_File)):
                pet_d.loc[date, column] = sd_functions.evapotranspiration_Hargreaves(tem_max_d.loc[date, column],
                                                                                     tem_min_d.loc[date, column],
                                                                                     latitude, date.dayofyear)
            elif (wind_d.isnull().loc[date, column]) | (humidity_d.isnull().loc[date, column]) | (
            solar_d.isnull().loc[date, column]):
                pet_d.loc[date, column] = sd_functions.evapotranspiration_Hargreaves(tem_max_d.loc[date, column],
                                                                                     tem_min_d.loc[date, column],
                                                                                     latitude, date.dayofyear)
            else:
                pet_d.loc[date, column] = sd_functions.evapotranspiration_FAO_PM(tem_max_d.loc[date, column],
                                                                                 tem_min_d.loc[date, column],
                                                                                 altitude,
                                                                                 wind_d.loc[date, column], wind_type,
                                                                                 humidity_d.loc[date, column],
                                                                                 date.dayofyear, latitude,
                                                                                 solar_d.loc[date, column], solar_type,
                                                                                 albedo=0.23)
    pet_d.to_csv(output_dir + "/PET_d.txt", sep="\t", mode="w", header=True, index=True, float_format="%.2f")
    del tem_max_d
    del tem_min_d

    # Calculate the areas of fields, ditches, and pond, in m2
    if orderDitch == 1:
        area_f = area_f * 10000  # in m2
        area_d1 = Wd1_up * Ld1  # in m2
        Wf2 = Ld1 / n1  # in m
        Wf1 = area_f / Ld1  # in m
        area_d2 = 0
    else:
        area_f = Wf1 * Wf2 * n1 * n2  # in m2
        area_d1 = Wd1_up * Wf2 * n1 * n2
        area_d2 = Wd2_up * Wf1 * n1 * n2
    if havePond is True:
        area_p = area_f * rate_pond
    else:
        area_p = 0

    area_pd = area_d1 + area_d2 + area_p
    pdf_rate1 = area_pd / (area_f + area_pd) * 100

    """Prepare a pandas Series to store the paired depth-volume (H-V) relationship"""
    if (havePond is True) & (orderDitch == 2):
        H_array = np.array(range(int(HDiff_FD1 - HDiff_FP), int(HDiff_FD1) + 1, 1))
        V_array = np.zeros(H_array.shape)
        for i in range(H_array.size):
            H = H_array[i]
            V_array[i] = (sd_functions.f_vhd(float(H), HDiff_FD1, Wd1_up, Wd1_lw, Wf2) +
                          sd_functions.f_vhd(float(H) + HDiff_FD2 - HDiff_FD1, HDiff_FD2, Wd2_up, Wd2_lw,
                                             Wf1)) * n1 * n2 + \
                         sd_functions.f_vhp(float(H) + HDiff_FP - HDiff_FD1, HDiff_FP, area_p, slp_p)
        H_V_sr = pd.Series(data=V_array, index=H_array)
    elif (havePond is True) & (orderDitch == 1):
        H_array = np.array(range(int(HDiff_FD1 - HDiff_FP), int(HDiff_FD1) + 1, 1))
        V_array = np.zeros(H_array.shape)
        for i in range(H_array.size):
            H = H_array[i]
            V_array[i] = sd_functions.f_vhd(float(H), HDiff_FD1, Wd1_up, Wd1_lw, Wf2) * n1 + \
                         sd_functions.f_vhp(float(H) + HDiff_FP - HDiff_FD1, HDiff_FP, area_p, slp_p)
        H_V_sr = pd.Series(data=V_array, index=H_array)
    elif (havePond is False) & (orderDitch == 2):
        H_array = np.array(range(int(HDiff_FD1 - HDiff_FD2), int(HDiff_FD1) + 1, 1))
        V_array = np.zeros(H_array.shape)
        for i in range(H_array.size):
            H = H_array[i]
            V_array[i] = (sd_functions.f_vhd(float(H), HDiff_FD1, Wd1_up, Wd1_lw, Wf2) +
                          sd_functions.f_vhd(float(H) + HDiff_FD2 - HDiff_FD1, HDiff_FD2, Wd2_up, Wd2_lw,
                                             Wf1)) * n1 * n2
        H_V_sr = pd.Series(data=V_array, index=H_array)
    elif (havePond is False) & (orderDitch == 1):
        H_array = np.array(range(0, int(HDiff_FD1) + 1, 1))
        V_array = np.zeros(H_array.shape)
        for i in range(H_array.size):
            H = H_array[i]
            V_array[i] = sd_functions.f_vhd(float(H), HDiff_FD1, Wd1_up, Wd1_lw, Wf2) * n1
        H_V_sr = pd.Series(data=V_array, index=H_array)
    # The rate of total volume of ditches and pond (m3) to the area of fields (m2), in m
    pdf_rate = V_array.max() / 1000 / area_f
    print("area_rate: {:.3f}%".format(pdf_rate1))
    print("fdp_rate: {:.3f}".format(pdf_rate))

    """Create a pandas DataFrame to store the paired field discharge depth-ditch adjusted depth (Hf-Hd) relationship, for accurate water management (AWM)"""
    Hf_max = int(round(V_array.max() / area_f, 0))  # the maximum controllable field depth, in mm
    Hf_list = list(range(0, Hf_max, 5))
    Hf_list.append(Hf_max)
    Hd_array = np.zeros(len(Hf_list), dtype=np.int16)
    for i, hf in enumerate(Hf_list):
        v_adjusted = V_array.max() - float(hf) * area_f  # the adjusted volume, in mm.m2
        h = int(HDiff_FD1)
        while H_V_sr[h] > v_adjusted:
            h -= 1
            if h < H_array[0]:
                h = H_array[0]
                break
        Hd_array[i] = h  # the adjusted depth as in the 1st order ditch, in mm
    Hf_Hdp_sr = pd.Series(data=Hd_array, index=Hf_list)

    if havePond is True:
        Hp_array = Hd_array + int(HDiff_FP) - int(HDiff_FD1)
        Hp_array = np.where(Hp_array > 0, Hp_array, 0)
    if orderDitch == 2:
        Hd2_array = Hd_array + int(HDiff_FD2) - int(HDiff_FD1)
        Hd2_array = np.where(Hd2_array > 0, Hd2_array, 0)
    Hd1_array = np.where(Hd_array > 0, Hd_array, 0)
    if (havePond is True) & (orderDitch == 2):
        Hf_Hd_df = pd.DataFrame(data={'田块排水量（mm）': Hf_list, '第一级沟水位（mm）': Hd1_array,
                                      '第二级沟水位（mm）': Hd2_array, '塘水位（mm）': Hp_array})
    elif (havePond is True) & (orderDitch == 1):
        Hf_Hd_df = pd.DataFrame(data={'田块排水量（mm）': Hf_list, '第一级沟水位（mm）': Hd1_array,
                                      '塘水位（mm）': Hp_array})
    elif (havePond is False) & (orderDitch == 2):
        Hf_Hd_df = pd.DataFrame(data={'田块排水量（mm）': Hf_list, '第一级沟水位（mm）': Hd1_array,
                                      '第二级沟水位（mm）': Hd2_array})
    elif (havePond is False) & (orderDitch == 1):
        Hf_Hd_df = pd.DataFrame(data={'田块排水量（mm）': Hf_list, '第一级沟水位（mm）': Hd1_array})

    water_control_File = output_dir + "/water_level_control.txt"
    f = open(water_control_File, mode='w')
    f.write("沟塘容积和田块面积之比:{:.4f} m\n".format(pdf_rate, '.4f'))
    f.write("沟塘面积占比:{:.2f} %\n\n".format(pdf_rate1, '.2f'))
    f.write("依据不同的田块排水量，各级沟塘的调控水位（自最低水位起算的实际水位）如下所示：\n\n")
    y = PrettyTable(Hf_Hd_df.columns.to_list())
    y.align[""] = "l"
    for index in range(0, Hf_Hd_df.shape[0], 1):
        y.add_row(Hf_Hd_df.iloc[index, :].to_list())
    f.write(y.get_string())
    f.close()

    """ Prepare parameters for main program """
    ditch_par = list()
    ditch_par.append(orderDitch)
    if orderDitch == 1:
        ditch_par.append(
            list([Wf1, HDiff_FD1, Wd1_up, Wd1_lw, n1, slp_d1, n_d1, kc_d1, TN_cd10, TP_cd10, Hd10, area_d1]))
    else:
        ditch_par.append(
            list([Wf1, HDiff_FD1, Wd1_up, Wd1_lw, n1, slp_d1, n_d1, kc_d1, TN_cd10, TP_cd10, Hd10, area_d1]))
        ditch_par.append(list([Wf2, HDiff_FD2, Wd2_up, Wd2_lw, n2, slp_d2, n_d2, kc_d2, TN_cd20, TP_cd20, area_d2]))
    pond_par = list()
    pond_par.append(havePond)
    if havePond:
        pond_par.append(list([rate_pond, HDiff_FP, slp_p, kc_p, TN_cp0, TP_cp0, area_p]))

    """Start the main program, as the set mode"""
    if (TAW is None) | (Irrg_rate is None):
        if (pre_risk == 0) & (recycle_irrg == 0):
            output_normal = modulesV2.normal(prcp_d, startDay, endDay, wtr_mng_d, pet_d, ditch_par, pond_par,
                                             F_doy_array,
                                             fert_type_array,
                                             TN0_array, TP0_array, TN_I, TP_I, area_f, Ks, Fc0_f, Fc0_pd, TN_P0,
                                             TN_P1, TP_P,
                                             TN_rate_lch, TN_rate_lch_lat, TP_rate_lch, TP_rate_lch_lat, ENC0_dp,
                                             EPC0_dp,
                                             ENCR_f, EPCR_f,
                                             vf_N_array, vf_P, H_V_sr, vf_N_pd, vf_P_pd, run_id, output_dir,
                                             graph)
        elif (pre_risk == 0) & (recycle_irrg == 1):
            output_bmp = modulesV2.recycle_irrg(prcp_d, startDay, endDay, wtr_mng_d, pet_d, ditch_par, pond_par,
                                                F_doy_array,
                                                fert_type_array, TN0_array, TP0_array, TN_I, TP_I, area_f, Ks,
                                                Fc0_f, Fc0_pd,
                                                TN_P0, TN_P1, TP_P, TN_rate_lch, TN_rate_lch_lat, TP_rate_lch,
                                                TP_rate_lch_lat,
                                                ENC0_dp, EPC0_dp, ENCR_f, EPCR_f, vf_N_array, vf_P, H_V_sr,
                                                vf_N_pd, vf_P_pd,
                                                min_level_recycle, run_id, output_dir, graph)

        if ((pre_risk == 1) | (recycle_irrg == 1)):
            if bmp_compare == 1:
                normal_comp = modulesV2.normal(prcp_d, startDay, endDay, wtr_mng_d, pet_d, ditch_par, pond_par,
                                               F_doy_array,
                                               fert_type_array,
                                               TN0_array, TP0_array, TN_I, TP_I, area_f, Ks, Fc0_f, Fc0_pd, TN_P0,
                                               TN_P1,
                                               TP_P,
                                               TN_rate_lch, TN_rate_lch_lat, TP_rate_lch, TP_rate_lch_lat,
                                               ENC0_dp, EPC0_dp,
                                               ENCR_f, EPCR_f,
                                               vf_N_array, vf_P, H_V_sr, vf_N_pd, vf_P_pd, run_id, output_dir=0,
                                               graph=0)
                output_text = output_bmp[0]
                bmp_comp = output_bmp[1]
                output_text += "\n\n所选强化措施的效果如下：\n"
                x = PrettyTable(["", "TN流失负荷(kg/ha)", "TP流失负荷(kg/ha)", "径流TN浓度(mg/L)", "径流TP浓度(mg/L)"])
                x.align[""] = "l"
                x.add_row(["常规措施", normal_comp["TN_load"], normal_comp["TP_load"], normal_comp["TN_conc"],
                           normal_comp["TP_conc"]])
                x.add_row(["强化措施", bmp_comp["TN_load"], bmp_comp["TP_load"], bmp_comp["TN_conc"], bmp_comp["TP_conc"]])
                x.add_row(["强化效果(%)", (bmp_comp["TN_load"] - normal_comp["TN_load"]) / normal_comp["TN_load"] * 100,
                           (bmp_comp["TP_load"] - normal_comp["TP_load"]) / normal_comp["TP_load"] * 100,
                           (bmp_comp["TN_conc"] - normal_comp["TN_conc"]) / normal_comp["TN_conc"] * 100,
                           (bmp_comp["TP_conc"] - normal_comp["TP_conc"]) / normal_comp["TP_conc"] * 100])
                x.float_format = ".2"
                output_text += x.get_string()
            else:
                output_text = output_bmp[0]
            if graph == 1:
                graph_paths = sd_functions.draw_graphs(output_bmp[2], output_dir)

        else:
            output_text = output_normal[0]
            if graph == 1:
                graph_paths = sd_functions.draw_graphs(output_normal[2], output_dir)
        print(output_text)  # 文本结果输出。标题叫“结果小结”
        # 按顺序显示三张图片结果，图片文件路径在graph_paths中（list）.
        # 图片标题依次是：“水量水质变化时间序列图”“田块-灌排单元尺度氮磷流失量对比图”“田块-灌排单元尺度氮磷流入流出对比图”
    else:
        if (pre_risk == 0) & (recycle_irrg == 0):
            output_normal = modulesV2.normal_irrigation_restrict(prcp_d, startDay, endDay, wtr_mng_d, pet_d, ditch_par,
                                                                 pond_par, F_doy_array,
                                                                 fert_type_array, TN0_array, TP0_array, TN_I, TP_I,
                                                                 area_f, Ks, Fc0_f, Fc0_pd,
                                                                 TN_P0, TN_P1, TP_P, TN_rate_lch, TN_rate_lch_lat,
                                                                 TP_rate_lch, TP_rate_lch_lat, ENC0_dp, EPC0_dp,
                                                                 ENCR_f, EPCR_f, vf_N_array, vf_P, H_V_sr, vf_N_pd,
                                                                 vf_P_pd, run_id, TAW,
                                                                 Irrg_rate, output_dir, graph)
        elif (pre_risk == 0) & (recycle_irrg == 1):
            output_bmp = modulesV2.recycle_irrg_irrigation_restrict(prcp_d, startDay, endDay, wtr_mng_d, pet_d,
                                                                    ditch_par, pond_par, F_doy_array,
                                                                    fert_type_array, TN0_array, TP0_array, TN_I, TP_I,
                                                                    area_f, Ks, Fc0_f, Fc0_pd,
                                                                    TN_P0, TN_P1, TP_P, TN_rate_lch, TN_rate_lch_lat,
                                                                    TP_rate_lch,
                                                                    TP_rate_lch_lat,
                                                                    ENC0_dp, EPC0_dp, ENCR_f, EPCR_f, vf_N_array, vf_P,
                                                                    H_V_sr, vf_N_pd, vf_P_pd,
                                                                    min_level_recycle, run_id, TAW, Irrg_rate,
                                                                    output_dir, graph)

        if ((pre_risk == 1) | (recycle_irrg == 1)):
            if bmp_compare == 1:
                normal_comp = modulesV2.normal_irrigation_restrict(prcp_d, startDay, endDay, wtr_mng_d, pet_d,
                                                                   ditch_par, pond_par, F_doy_array,
                                                                   fert_type_array,
                                                                   TN0_array, TP0_array, TN_I, TP_I, area_f, Ks, Fc0_f,
                                                                   Fc0_pd, TN_P0, TN_P1,
                                                                   TP_P,
                                                                   TN_rate_lch, TN_rate_lch_lat, TP_rate_lch,
                                                                   TP_rate_lch_lat, ENC0_dp, EPC0_dp,
                                                                   ENCR_f, EPCR_f,
                                                                   vf_N_array, vf_P, H_V_sr, vf_N_pd, vf_P_pd, run_id,
                                                                   TAW,
                                                                   Irrg_rate, output_dir=0, graph=0)
                output_text = output_bmp[0]
                bmp_comp = output_bmp[1]
                output_text += "\n\n所选强化措施的效果如下：\n"
                x = PrettyTable(["", "TN流失负荷(kg/ha)", "TP流失负荷(kg/ha)", "径流TN浓度(mg/L)", "径流TP浓度(mg/L)"])
                x.align[""] = "l"
                x.add_row(["常规措施", normal_comp["TN_load"], normal_comp["TP_load"], normal_comp["TN_conc"],
                           normal_comp["TP_conc"]])
                x.add_row(["强化措施", bmp_comp["TN_load"], bmp_comp["TP_load"], bmp_comp["TN_conc"], bmp_comp["TP_conc"]])
                x.add_row(["强化效果(%)", (bmp_comp["TN_load"] - normal_comp["TN_load"]) / normal_comp["TN_load"] * 100,
                           (bmp_comp["TP_load"] - normal_comp["TP_load"]) / normal_comp["TP_load"] * 100,
                           (bmp_comp["TN_conc"] - normal_comp["TN_conc"]) / normal_comp["TN_conc"] * 100,
                           (bmp_comp["TP_conc"] - normal_comp["TP_conc"]) / normal_comp["TP_conc"] * 100])
                x.float_format = ".2"
                output_text += x.get_string()
            else:
                output_text = output_bmp[0]
            if graph == 1:
                graph_paths = sd_functions.draw_graphs(output_bmp[2], output_dir)

        else:
            output_text = output_normal[0]
            if graph == 1:
                graph_paths = sd_functions.draw_graphs(output_normal[2], output_dir)
        print(output_text)  # 文本结果输出。标题叫“结果小结”
        # 按顺序显示三张图片结果，图片文件路径在graph_paths中（list）.
        # 图片标题依次是：“水量水质变化时间序列图”“田块-灌排单元尺度氮磷流失量对比图”“田块-灌排单元尺度氮磷流入流出对比图”

