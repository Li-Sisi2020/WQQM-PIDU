# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import calendar
import math
import sd_functions
from prettytable import PrettyTable


def create_common_df(prcp_d, startDay, endDay, wtr_mng_d):
    """ Create common empty pandas DataFrames to store results.
    Parameters:
        prcp_d, pandas DataFrame stores daily precipitation data, in mm;
        startDay, the start day of the rice growing period, in "MM-DD";
        endDay, the end day of the rice growing period, in "MM-DD";
        wtr_mng_d, pandas DataFrame stores water management parameters and Kc for each day during the growing period.
    Outputs:
        The created common pandas DataFrames in this order:
        Hf_df, Hdp_df, D_df, F_lat_df, Fc_f_df, Fc_pd_df, Qdp_df, I_df, TN_cf_df, TP_cf_df, TN_cd1_df, TP_cd1_df,
        TN_cd2_df, TP_cd2_df, TN_cp_df, TP_cp_df, TN_cfb_df, TP_cfb_df, TN_cfb_lat_df, TP_cfb_lat_df, TN_b_df, TP_b_df,
        TN_out_df, TP_out_df, Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1,
        TN_cp_df1, TP_cp_df1, TN_fs_df1, TP_fs_df1, TN_s_df1, TP_s_df1"""
    Hf_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns, dtype=np.float32)  # water level in fields, in mm
    Hdp_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                          dtype=np.float32)  # water level of ditches and pond, in mm
    D_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                        dtype=np.float32)  # Surface drainage from fields, in mm/d
    F_lat_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                            dtype=np.float32)  # Subsurface percolation from fields, in mm/d
    Fc_f_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                           dtype=np.float32)  # Subsurface percolation from fields, in mm/d
    Fc_pd_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                            dtype=np.float32)  # Subsurface percolation from fields, in mm/d
    Qdp_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                          dtype=np.float32)  # Runoff out of the system unit, in mm/d
    I_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns, dtype=np.float32)  # Irrigated water, in mm/d
    TP_cf_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                            dtype=np.float32)  # TP concentration in surface water of fields, in mg/L
    TN_cf_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                            dtype=np.float32)  # TN concentration in surface water of fields, in mg/L
    TN_cd1_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                             dtype=np.float32)  # TN concentration in 1st order ditch, in mg/L
    TP_cd1_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                             dtype=np.float32)  # TP concentration in 1st order ditch, in mg/L
    TN_cd2_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                             dtype=np.float32)  # TN concentration in 2nd order ditch, in mg/L
    TP_cd2_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                             dtype=np.float32)  # TP concentration in 2nd order ditch, in mg/L
    TN_cp_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                            dtype=np.float32)  # TN concentration in pond, in mg/L
    TP_cp_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                            dtype=np.float32)  # TP concentration in pond, in mg/L
    TN_cfb_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                             dtype=np.float32)  # TN concentration of leaching water from fields, in mg/L
    TP_cfb_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                             dtype=np.float32)  # TP concentration of leaching water from fields, in mg/L
    TN_cfb_lat_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                                 dtype=np.float32)  # TN concentration of leaching water from fields, in mg/L
    TP_cfb_lat_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                                 dtype=np.float32)  # TP concentration of leaching water from fields, in mg/L
    TN_b_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                           dtype=np.float32)  # TN load from system leaching water, in kg/ha/d
    TP_b_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                           dtype=np.float32)  # TP load from system leaching water, in kg/ha/d
    TN_out_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                             dtype=np.float32)  # TN concentration of outflow from the IDU system, in mg/L
    TP_out_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                             dtype=np.float32)  # TP concentration of outflow from the IDU system, in mg/L
    yearStart = prcp_d.index[0].year
    yearEnd = prcp_d.index[-1].year + 1
    index = pd.MultiIndex.from_product([[year for year in range(yearStart, yearEnd)], prcp_d.columns],
                                       names=['year', 'station'])
    Hf_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    Hdp_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TP_cf_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TN_cf_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TP_cd1_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TN_cd1_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TP_cd2_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TN_cd2_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TP_cp_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TN_cp_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TN_fs_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TP_fs_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TN_s_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    TP_s_df1 = pd.DataFrame(index=wtr_mng_d.loc[startDay:endDay, :].index, columns=index, dtype=np.float32)
    return (Hf_df, Hdp_df, D_df, F_lat_df, Fc_f_df, Fc_pd_df, Qdp_df, I_df, TN_cf_df, TP_cf_df, TN_cd1_df, TP_cd1_df,
            TN_cd2_df, TP_cd2_df, TN_cp_df, TP_cp_df, TN_cfb_df, TP_cfb_df, TN_cfb_lat_df, TP_cfb_lat_df, TN_b_df,
            TP_b_df, TN_out_df, TP_out_df, Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1,
            TP_cd2_df1, TN_cp_df1, TP_cp_df1, TN_fs_df1, TP_fs_df1, TN_s_df1, TP_s_df1)


def normal(prcp_d, startDay, endDay, wtr_mng_d, pet_d, ditch_par, pond_par, F_doy_array, fert_type_array,
           TN0_array, TP0_array, TN_I, TP_I, area_f, Ks, Fc0_f, Fc0_pd, TN_P0, TN_P1, TP_P, TN_rate_lch, TN_rate_lch_lat,
           TP_rate_lch, TP_rate_lch_lat, ENC0_dp, EPC0_dp, ENCR_f, EPCR_f, vf_N_array, vf_P, h_v_sr, vf_N_pd, vf_P_pd, run_id,
           output_dir=0, graph=0):
    """ The normal mode, for only accurate water level management.
    Parameters:
        prcp_d, pandas DataFrame stores daily precipitation data, in mm;
    Outputs:
        Hf"""
    Hf_df, Hdp_df, D_df, F_lat_df, Fc_f_df, Fc_pd_df, Qdp_df, I_df, TN_cf_df, TP_cf_df, TN_cd1_df, TP_cd1_df, \
    TN_cd2_df, TP_cd2_df, TN_cp_df, TP_cp_df, TN_cfb_df, TP_cfb_df, TN_cfb_lat_df, TP_cfb_lat_df, TN_b_df, TP_b_df, \
    TN_out_df, TP_out_df, Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1, \
    TN_cp_df1, TP_cp_df1, TN_fs_df1, TP_fs_df1, TN_s_df1, TP_s_df1 = create_common_df(
        prcp_d, startDay, endDay, wtr_mng_d)
    TN_b_df_4wf = TN_b_df.copy(deep=True)
    ET_df = Hf_df.copy(deep=True)

    orderDitch = ditch_par[0]
    if orderDitch == 1:
        Wf1 = ditch_par[1][0]
        HDiff_FD1 = ditch_par[1][1]
        Wd1_up = ditch_par[1][2]
        Wd1_lw = ditch_par[1][3]
        n1 = ditch_par[1][4]
        slp_d1 = ditch_par[1][5]
        n_d1 = ditch_par[1][6]
        kc_d1 = ditch_par[1][7]
        TN_cd10 = ditch_par[1][8]
        TP_cd10 = ditch_par[1][9]
        Hd10 = ditch_par[1][10]
        area_d1 = ditch_par[1][11]
        area_d2 = 0
        Wf2 = area_f / n1 / Wf1
    else:
        Wf1 = ditch_par[1][0]
        HDiff_FD1 = ditch_par[1][1]
        Wd1_up = ditch_par[1][2]
        Wd1_lw = ditch_par[1][3]
        n1 = ditch_par[1][4]
        slp_d1 = ditch_par[1][5]
        n_d1 = ditch_par[1][6]
        kc_d1 = ditch_par[1][7]
        TN_cd10 = ditch_par[1][8]
        TP_cd10 = ditch_par[1][9]
        Hd10 = ditch_par[1][10]
        area_d1 = ditch_par[1][11]
        Wf2 = ditch_par[2][0]
        HDiff_FD2 = ditch_par[2][1]
        Wd2_up = ditch_par[2][2]
        Wd2_lw = ditch_par[2][3]
        n2 = ditch_par[2][4]
        slp_d2 = ditch_par[2][5]
        n_d2 = ditch_par[2][6]
        kc_d2 = ditch_par[2][7]
        TN_cd20 = ditch_par[2][8]
        TP_cd20 = ditch_par[2][9]
        area_d2 = ditch_par[2][10]
    havePond = pond_par[0]
    if havePond:
        HDiff_FP = pond_par[1][1]
        slp_p = pond_par[1][2]
        kc_p = pond_par[1][3]
        TN_cp0 = pond_par[1][4]
        TP_cp0 = pond_par[1][5]
        area_p = pond_par[1][6]
    else:
        area_p = 0.0

    # The rate of area of ditches and pond to the total area, in %
    area_pd = area_d1 + area_d2 + area_p

    startday_doy = pd.Timestamp(2018, int(startDay.split('-')[0]), int(startDay.split('-')[1])).dayofyear
    endday_doy = pd.Timestamp(2018, int(endDay.split('-')[0]), int(endDay.split('-')[1])).dayofyear
    for column in prcp_d.columns:
        date_n = 0
        while date_n < len(prcp_d.index):
            dayofYear = prcp_d.index[date_n].dayofyear
            monthDay = str(prcp_d.index[date_n].month) + "-" + str(prcp_d.index[date_n].day)
            if calendar.isleap(prcp_d.index[date_n].year):
                dayofYear = dayofYear - 1
            if dayofYear == startday_doy:
                Hf = 0.0
                Hdp = Hd10
                Hd1 = Hd10
                TP_cd1 = TP_cd10
                TN_cd1 = TN_cd10
                if orderDitch == 2:
                    Hd2 = Hd1 + HDiff_FD2 - HDiff_FD1
                    TP_cd2 = TP_cd20
                    TN_cd2 = TN_cd20
                if havePond:
                    Hp = Hd1 + HDiff_FP - HDiff_FD1
                    TP_cp = TP_cp0
                    TN_cp = TN_cp0
                irrg_date = wtr_mng_d["H_min"].ne(0).idxmax()
                irrg_doy = pd.Timestamp(2018, int(irrg_date.split('-')[0]), int(irrg_date.split('-')[1])).dayofyear
                if F_doy_array[0] <= irrg_doy:  # 施肥日在灌溉日之前，田面水浓度初始值为施肥初始浓度
                    TP_cf = TP0_array[0]
                    TN_cf = TN0_array[0]
                else:  # 施肥日在灌溉日之后，田面水浓度初始值为灌溉水浓度
                    TP_cf = TP_I
                    TN_cf = TN_I
                fert_lag = 0

            if (dayofYear < startday_doy) | (dayofYear > endday_doy):
                prcp_d.iloc[date_n] = 0.0  # Change the precipitation in non-growing period to 0, for statistics
            else:
                """ Water balance in fields """
                # Get the precipitation and PET data of the simulation day
                prcp = prcp_d[column].iloc[date_n]
                PET = pet_d[column].iloc[date_n]
                # Calculate ET
                if Hf > 0:
                    ET = PET * wtr_mng_d.loc[monthDay, 'Kc']
                else:
                    ET = PET * 0.60
                ET_df[column].iloc[date_n] = ET
                # Do irrigation or not
                if (wtr_mng_d.loc[monthDay, 'H_min'] != 0) & (Hf < wtr_mng_d.loc[monthDay, 'H_min']) & (
                        prcp + Hf - ET < wtr_mng_d.loc[monthDay, 'H_min']):
                    irrg = wtr_mng_d.loc[monthDay, 'H_max'] - Hf
                else:
                    irrg = 0
                I_df[column].iloc[date_n] = irrg
                # Calculate lateral seepage to ditches
                if orderDitch == 2:
                    F_lat_d1 = max(0, Ks * (Hf - Hd1 + HDiff_FD1) / (0.25 * Wf1 * 1000) * Wf2 * 2 * (
                            Hf + HDiff_FD1))  # in m2.mm/d
                    F_lat_d2 = max(0, Ks * (Hf - Hd2 + HDiff_FD2) / (0.25 * Wf2 * 1000) * Wf1 * 2 * (
                            Hf + HDiff_FD2))  # in m2.mm/d
                    F_lat = (F_lat_d1 + F_lat_d2) / (Wf1 * Wf2)  # in mm/d
                elif orderDitch == 1:
                    F_lat_d1 = max(0, Ks * (Hf - Hd1 + HDiff_FD1) / (0.25 * Wf1 * 1000) * Wf2 * 2 * (
                            Hf + HDiff_FD1))
                    F_lat = F_lat_d1 / (Wf1 * Wf2)  # in mm/d
                F_lat_df[column].iloc[date_n] = F_lat
                # Calculate downward percolation
                if (Hf - ET) > Fc0_f:
                    Fc_f = Fc0_f
                else:
                    Fc_f = 0
                Fc_f_df[column].iloc[date_n] = Fc_f
                # Update water level in fields, Hf, first-round
                Hf1 = Hf + prcp + irrg - ET - Fc_f - F_lat
                # Do drainage or not
                if Hf1 > wtr_mng_d.loc[monthDay, 'H_p']:
                    D = Hf1 - wtr_mng_d.loc[monthDay, 'H_p']  # in mm/d
                    Hf1 = Hf1 - D  # Update again, second-round
                else:
                    D = 0
                D_df[column].iloc[date_n] = D

                """ Water quality in fields"""
                # Initial TN and TP concentrations for the fertilization days, and
                # adjust TP and TN with irrigation, precipitation and ETs.
                flag_N = 0
                flag_P = 0
                TN_P = TN_P1
                for index, F_doy in enumerate(F_doy_array[TN0_array > 0]):
                    if (dayofYear >= F_doy) & (dayofYear - 15 <= F_doy):
                        TN_P = TN_P0
                for index, item in enumerate(F_doy_array):
                    if (dayofYear - fert_lag == item) & (D == 0):
                        fert_lag = 0
                        if TN0_array[index] != 0:
                            TN_cf = TN0_array[index]
                            flag_N = 1
                        if TP0_array[index] != 0:
                            TP_cf = TP0_array[index]
                            flag_P = 1
                    elif (dayofYear - fert_lag == item) & (D > 0):
                        if fert_lag < 5:
                            fert_lag += 1
                        else:
                            fert_lag = 0
                            if TN0_array[index] != 0:
                                TN_cf = TN0_array[index]
                                flag_N = 1
                            if TP0_array[index] != 0:
                                TP_cf = TP0_array[index]
                                flag_P = 1
                # Mixed concentration based on material balance
                if (Hf1 >= 10) & (Hf > 10):
                    if flag_N == 0:
                        TN_cf = (TN_cf * Hf + TN_I * irrg + TN_P * prcp) / (Hf + irrg + prcp - ET)
                    if flag_P == 0:
                        TP_cf = (TP_cf * Hf + TP_I * irrg + TP_P * prcp) / (Hf + irrg + prcp - ET)
                elif (Hf1 >= 10) & (Hf <= 10):
                    temp = max(min(10 + Hf, 10.0), 1)
                    if flag_N == 0:
                        TN_cf = (TN_cf * 2.0 * temp + TN_I * irrg + TN_P * prcp) / (temp + irrg + prcp - ET)
                    if flag_P == 0:
                        TP_cf = (TP_cf * 4.0 * temp + TP_I * irrg + TP_P * prcp) / (temp + irrg + prcp - ET)
                    del temp
                # Adjust TN concentration for 2-7 days after tillering/filling fertilization
                for x, doy in enumerate(F_doy_array[fert_type_array == 2]):
                    if (dayofYear - 2 >= doy) & (dayofYear - 3 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.3
                    elif (dayofYear - 4 >= doy) & (dayofYear - 5 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.2
                    elif (dayofYear - 6 >= doy) & (dayofYear - 7 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.1
                    else:
                        TN_cf = TN_cf

                # Record percolated concentrations
                TN_cfb = TN_cf * TN_rate_lch
                TN_cfb_lat = max(TN_cf * TN_rate_lch_lat, ENC0_dp)
                TP_cfb = TP_cf * TP_rate_lch
                TP_cfb_lat = max(TP_cf * TP_rate_lch_lat, EPC0_dp)
                TP_cfb_df[column].iloc[date_n] = TP_cfb
                TN_cfb_df[column].iloc[date_n] = TN_cfb
                TP_cfb_lat_df[column].iloc[date_n] = TP_cfb_lat
                TN_cfb_lat_df[column].iloc[date_n] = TN_cfb_lat
                # Account for areal dynamics in the water-soil interface
                flag_N = 0
                flag_P = 0
                for index, F_doy in enumerate(F_doy_array[TN0_array > 0]):
                    if index + 1 < np.count_nonzero(TN0_array):
                        if (dayofYear > F_doy) & (dayofYear < F_doy_array[TN0_array > 0][index + 1]) & (Hf > 0) & (
                                Hf1 > 0) & (TN_cf > (ENCR_f * TN0_array[index])) & (flag_N == 0):
                            TN_cf1 = max(TN_cf * math.exp(- vf_N_array[index] * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         TN0_array[index] * ENCR_f)
                            flag_N = 1
                    else:
                        if (dayofYear > F_doy) & (Hf > 0) & (Hf1 > 0) & (TN_cf > (ENCR_f * TN0_array[index])) & (
                                flag_N == 0):
                            TN_cf1 = max(TN_cf * math.exp(- vf_N_array[index] * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         TN0_array[index] * ENCR_f)
                            flag_N = 1
                for index, F_doy in enumerate(F_doy_array[TP0_array > 0]):
                    if index + 1 < np.count_nonzero(TP0_array):
                        if (dayofYear > F_doy) & (dayofYear < F_doy_array[TP0_array > 0][index + 1]) & (Hf > 0) & (
                                Hf1 > 0) & (TP_cf > (EPCR_f * TP0_array[index])) & (flag_P == 0):
                            TP_cf1 = max(TP_cf * math.exp(- vf_P * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         EPCR_f * TP0_array[index])
                            flag_P = 1
                    else:
                        if (dayofYear > F_doy) & (Hf > 0) & (Hf1 > 0) & (TP_cf > (EPCR_f * TP0_array[index])) & (
                                flag_P == 0):
                            TP_cf1 = max(TP_cf * math.exp(- vf_P * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         EPCR_f * TP0_array[index])
                            flag_P = 1
                if flag_P == 0:
                    TP_cf1 = TP_cf
                if flag_N == 0:
                    TN_cf1 = TN_cf
                # Record water table and water quality in fields
                TP_cf_df[column].iloc[date_n] = (TP_cf + TP_cf1) / 2
                TN_cf_df[column].iloc[date_n] = (TN_cf + TN_cf1) / 2
                if Hf1 >= 0:
                    Hf_df[column].iloc[date_n] = Hf1
                else:
                    Hf_df[column].iloc[date_n] = 0.0
                Hf = Hf1

                """ Water balance in ditches and the pond """
                # Calculate ET and Fc from ditches and the pond
                if Hd1 > 0:
                    Wd1_d = (Wd1_up - Wd1_lw) * Hd1 / HDiff_FD1 + Wd1_lw
                    if orderDitch == 2:
                        area_d1_d = Wd1_d * Wf2 * n1 * n2
                    elif orderDitch == 1:
                        area_d1_d = Wd1_d * Wf2 * n1
                    if Hd1 - Fc0_pd > (kc_d1 * PET):
                        Fc_d1_v = Fc0_pd * area_d1_d
                    else:
                        Fc_d1_v = 0.0
                else:
                    area_d1_d = 0
                    Fc_d1_v = 0.0

                if orderDitch == 2:
                    if Hd2 > 0:
                        Wd2_d = (Wd2_up - Wd2_lw) * Hd2 / HDiff_FD2 + Wd2_lw
                        area_d2_d = Wd2_d * Wf1 * n1 * n2
                        if Hd2 - Fc0_pd > (kc_d2 * PET):
                            Fc_d2_v = Fc0_pd * area_d2_d
                        else:
                            Fc_d2_v = 0.0
                    else:
                        area_d2_d = 0
                        Fc_d2_v = 0.0

                if havePond == True:
                    if Hp > 0:
                        area_p_lw = (math.sqrt(area_p) - HDiff_FP / 1000 * slp_p * 2) ** 2  # in m2
                        hp1 = math.sqrt(area_p_lw) / 2 / slp_p  # in m
                        area_p_d = area_p_lw * ((hp1 + Hp / 1000) ** 2) / (hp1 ** 2)  # in m2
                        if Hp - Fc0_pd > (kc_p * PET):
                            Fc_p_v = Fc0_pd * area_p_d
                        else:
                            Fc_p_v = 0.0
                    else:
                        area_p_d = 0
                        Fc_p_v = 0.0
                # ET, ET_pd;, downward percoloation, Fc_pd
                if (havePond is True) & (orderDitch == 2):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_d2 * PET * area_d2_d + kc_p * PET * area_p_d + \
                              0.60 * PET * (area_pd - area_d1_d - area_d2_d - area_p_d)
                    Fc_pd_v = Fc_d1_v + Fc_d2_v + Fc_p_v
                elif (havePond is True) & (orderDitch == 1):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_p * PET * area_p_d + 0.60 * PET * (
                            area_pd - area_d1_d - area_p_d)
                    Fc_pd_v = Fc_d1_v + Fc_p_v
                elif (havePond is False) & (orderDitch == 2):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_d2 * PET * area_d2_d + 0.60 * PET * (
                            area_pd - area_d1_d - area_d2_d)
                    Fc_pd_v = Fc_d1_v + Fc_d2_v
                elif (havePond is False) & (orderDitch == 1):
                    ET_pd_v = kc_d1 * PET * area_d1_d + 0.60 * PET * (
                            area_pd - area_d1_d)
                    Fc_pd_v = Fc_d1_v

                # Update water level in ditches and ponds, Hdp; and calculate outflow Q_dp of the system from the pond
                volume_add = prcp * area_pd - Fc_pd_v - ET_pd_v + (F_lat + D) * area_f
                Q_dp = sd_functions.f_vhpd(Hdp, volume_add, h_v_sr, area_pd, HDiff_FD1)[0]  # in m3/d
                Hdp1 = sd_functions.f_vhpd(Hdp, volume_add, h_v_sr, area_pd, HDiff_FD1)[1]  # in mm
                Qdp_df[column].iloc[date_n] = Q_dp / (area_f + area_pd) * 1000  # in mm/d
                Hdp_df[column].iloc[date_n] = Hdp1

                """ Water quality in ditches """
                # 1st order ditch
                # Record percolation concentration
                TN_cd1b = TN_cd1 * TN_rate_lch
                TP_cd1b = TP_cd1 * TP_rate_lch
                # Calculate mixed concentration
                if orderDitch == 2:
                    TN_cd1_out = (TN_cf * D * Wf1 * Wf2 + TN_cfb_lat * F_lat_d1 + TN_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TN_P * prcp * area_d1 / n1 / n2
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1 / n2)
                    TP_cd1_out = (TP_cf * D * Wf1 * Wf2 + TP_cfb_lat * F_lat_d1 + TP_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TP_P * prcp * area_d1 / n1 / n2
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1 / n2)
                elif orderDitch == 1:
                    TN_cd1_out = (TN_cf * D * Wf1 * Wf2 + TN_cfb_lat * F_lat_d1 + TN_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TN_P * prcp * area_d1 / n1
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1)
                    TP_cd1_out = (TP_cf * D * Wf1 * Wf2 + TP_cfb_lat * F_lat_d1 + TP_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TP_P * prcp * area_d1 / n1
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1)
                if TN_cd1_out <= 0:
                    TN_cd1_out = TN_cd1
                if TP_cd1_out <= 0:
                    TP_cd1_out = TP_cd1
                # Calculate retained concentration
                if Hdp1 > 0:
                    if TN_cd1 > ENC0_dp:
                        TN_cd1 = max(TN_cd1_out * math.exp(- vf_N_pd * 1 / max(0.05, Hdp1 / 1000)), ENC0_dp)
                    else:
                        TN_cd1 = TN_cd1_out
                    if TP_cd1_out > EPC0_dp:
                        TP_cd1 = max(TP_cd1_out * math.exp(- vf_P_pd * 1 / max(0.05, Hdp1 / 1000)), EPC0_dp)
                    else:
                        TP_cd1 = TP_cd1_out
                else:
                    TN_cd1 = TN_cd1_out
                    TP_cd1 = TP_cd1_out
                # Record concentrations
                TN_cd1_df[column].iloc[date_n] = (TN_cd1_out + TN_cd1) / 2
                TP_cd1_df[column].iloc[date_n] = (TP_cd1_out + TP_cd1) / 2
                TN_cf = TN_cf1
                TP_cf = TP_cf1
                # Prepare flow rate (m3/s) leaving the 1st order ditch for pond retention calculation
                if (Q_dp > 0) & (orderDitch == 1):
                    A_d1 = (Wd1_up + Wd1_lw) * HDiff_FD1 / 1000 / 2  # in m2，截面积
                    if D > 0:
                        Q_f0 = D * area_f / n1 / 24 / 3600 / 1000  # in m3/s,入流量
                        v_d1_out = (Q_f0 ** 0.4) * (slp_d1 ** 0.3) / (n_d1 ** 0.6) / (Wd1_up ** 0.4)  # in m/s，流速
                        Q_d1_out = v_d1_out * A_d1 * n1  # in m3/s,出流量
                        del v_d1_out
                    else:
                        Pw_d1 = Wd1_lw + (((Wd1_up - Wd1_lw) / 2) ** 2 + (HDiff_FD1 / 1000) ** 2) ** 0.5 * 2  # Wet perimeter, in m
                        Q_d1_out = (A_d1 ** 1.67) * (slp_d1 ** 0.5) / n_d1 / (Pw_d1 ** 0.67) * n1  # in m3/s
                        del Pw_d1
                    del A_d1

                # 2nd order ditch
                if orderDitch == 2:
                    # Record percolation concentration
                    TN_cd2b = TN_cd2 * TN_rate_lch
                    TP_cd2b = TP_cd2 * TP_rate_lch
                    # Prepare flow rate (m3/s) of the 2nd order ditch
                    if Hd2 > 0:
                        Pw_d2 = Wd2_lw + (((Wd2_up - Wd2_lw) / 2) ** 2 + (HDiff_FD2 / 1000) ** 2) ** 0.5 * \
                                Hd2 / HDiff_FD2 * 2  # Wet perimeter, in m
                        Wd2 = Wd2_lw + (Wd2_up - Wd2_lw) * Hd2 / HDiff_FD2  # water surface width, in m
                        A_d2 = (Wd2 + Wd2_lw) * Hd2 / 2000  # in m2
                        Q_d2 = (A_d2 ** 1.67) * (slp_d2 ** 0.5) / n_d2 / (Pw_d2 ** 0.67)  # in m3/s
                        del Pw_d2
                        del Wd2
                        del A_d2
                    else:
                        Q_d2 = 0.01
                    if D > 0:
                        # Calculate concentrations at the outlet of the 2nd order ditch by one-dimentional water quality model
                        # Prepare flow rate (m3/s) of 1st and 2nd order ditches for mixed concentration estimation #
                        Q_d1_0 = (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)
                                  + (prcp * area_d1 - kc_d1 * PET * area_d1_d - Fc_d1_v) / n1 / n2 - sd_functions.f_vhd(
                                    Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) / 1000  # in m3
                        TN_cd2_mix = (TN_cfb_lat * F_lat_d2 + TN_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TN_P * prcp * area_d2 / n1 / n2 + TN_cd1_out *
                                      Q_d1_0 * 1000) / (F_lat_d2 + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (prcp * area_d2 - kc_d2 * PET * area_d2_d
                                                                    ) / n1 / n2 + Q_d1_0 * 1000)
                        TP_cd2_mix = (TP_cfb_lat * F_lat_d2 + TP_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TP_P * prcp * area_d2 / n1 / n2 + TP_cd1_out *
                                      Q_d1_0 * 1000) / (F_lat_d2 + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (prcp * area_d2 - kc_d2 * PET * area_d2_d
                                                                    ) / n1 / n2 + Q_d1_0 * 1000)
                        # water surface width, in m
                        if Q_dp > 0:
                            Wd2 = Wd2_up
                            Hd2_d = HDiff_FD2 / 1000  # in m
                        else:
                            Hd2_d = max(0.01, (Hd2 + Hdp1 + HDiff_FD2 - HDiff_FD1) / 2000)  # in m
                            Wd2 = Wd2_lw + (Wd2_up - Wd2_lw) * Hd2_d * 1000 / HDiff_FD2
                        v_d2 = ((Q_d1_0 / 24 / 3600) ** 0.4) * (slp_d2 ** 0.3) / (n_d2 ** 0.6) / (
                                    Wd2 ** 0.4)  # in m/s
                        # Calculate concentrations at the outlet
                        TN_rate_array = np.zeros(n1)
                        TP_rate_array = np.zeros(n1)
                        for i in range(0, n1):
                            TN_rate_array[i] = math.exp(- vf_N_pd / Hd2_d * (Wf1 * (i + 0.5) / v_d2 / 3600 / 24))
                            TP_rate_array[i] = math.exp(- vf_P_pd / Hd2_d * Wf1 * (i + 0.5) / v_d2 / 3600 / 24)
                        if TN_cd2_mix > ENC0_dp:
                            TN_cd2_out = TN_cd2_mix * TN_rate_array.mean()
                        else:
                            TN_cd2_out = TN_cd2_mix
                        if TP_cd2_mix > EPC0_dp:
                            TP_cd2_out = TP_cd2_mix * TP_rate_array.mean()
                        else:
                            TP_cd2_out = TP_cd2_mix
                        # Calculate the flow discharge at the outlet of the second-order ditch
                        v_d2_out = ((Q_d1_0 * n1 / 24 / 3600) ** 0.4) * (slp_d2 ** 0.3) / (n_d2 ** 0.6) / \
                                   (Wd2 ** 0.4)  # in m/s
                        A_d2 = (Wd2 + Wd2_lw) * Hd2_d / 2  # in m2
                        Q_d2_out = v_d2_out * A_d2 * n2  # in m3/s
                        del Wd2
                        del Hd2_d
                        del v_d2_out
                        del v_d2
                        del A_d2
                        del TN_rate_array
                        del TP_rate_array
                    else:
                        # Calculate mixed concentration with lateral flow, precipitation and ET
                        TN_cd2_mix = (TN_cfb_lat * F_lat_d2 + TN_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TN_P * prcp * area_d2 / n1 / n2) / (
                                             F_lat_d2 + sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (
                                             prcp * area_d2 - kc_d2 * PET * area_d2_d) / n1 / n2)
                        TP_cd2_mix = (TP_cfb_lat * F_lat_d2 + TP_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TP_P * prcp * area_d2 / n1 / n2) / (
                                             F_lat_d2 + sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (
                                             prcp * area_d2 - kc_d2 * PET * area_d2_d) / n1 / n2)
                        if TN_cd2_mix <= 0:
                            TN_cd2_mix = TN_cd2
                        if TP_cd2_mix <= 0:
                            TP_cd2_mix = TP_cd2
                        TN_cd2_out = TN_cd2_mix
                        TP_cd2_out = TP_cd2_mix
                        # Calculate the flow discharge at the outlet of the second-order ditch
                        Q_d2_out = Q_d2 * n2
                    # Calculate the retained concentration within the second-order ditch
                    if (Hdp1 + HDiff_FD2 - HDiff_FD1) > 0:
                        if TN_cd2_mix > ENC0_dp:
                            TN_cd2 = max(TN_cd2_mix * math.exp(
                                - vf_N_pd * 1 / max(0.05, (Hdp1 + HDiff_FD2 - HDiff_FD1) / 1000)), ENC0_dp)
                        else:
                            TN_cd2 = TN_cd2_mix
                        if TP_cd2_mix > EPC0_dp:
                            TP_cd2 = max(TP_cd2_mix * math.exp(
                                - vf_P_pd * 1 / max(0.05, (Hdp1 + HDiff_FD2 - HDiff_FD1) / 1000)), EPC0_dp)
                        else:
                            TP_cd2 = TP_cd2_mix
                    else:
                        TN_cd2 = TN_cd2_mix
                        TP_cd2 = TP_cd2_mix
                    # Record concentrations
                    TN_cd2_df[column].iloc[date_n] = (TN_cd2_out + TN_cd2) / 2
                    TP_cd2_df[column].iloc[date_n] = (TP_cd2_out + TP_cd2) / 2

                """ Record percolation TN and TP losses from the IDU """
                if havePond is True:
                    # Record percolation concentrations
                    TN_cpb = TN_cp * TN_rate_lch
                    TP_cpb = TP_cp * TP_rate_lch
                if (havePond is True) & (orderDitch == 2):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v + TN_cd2b * Fc_d2_v +
                                                    TN_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v + TP_cd2b * Fc_d2_v +
                                                    TP_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cd2b <= 10.0:
                        TN_cd2b_4wf = 0
                    else:
                        TN_cd2b_4wf = TN_cd2b
                    if TN_cpb <= 10.0:
                        TN_cpb_4wf = 0
                    else:
                        TN_cpb_4wf = TN_cpb
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cd2b_4wf * Fc_d2_v + TN_cpb_4wf * Fc_p_v) / (
                                                               area_f + area_pd) / 100
                elif (havePond is True) & (orderDitch == 1):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v +
                                                    TN_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v +
                                                    TP_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cpb <= 10.0:
                        TN_cpb_4wf = 0
                    else:
                        TN_cpb_4wf = TN_cpb
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cpb_4wf * Fc_p_v) / (
                                                               area_f + area_pd) / 100
                elif (havePond is False) & (orderDitch == 2):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v + TN_cd2b * Fc_d2_v
                                                    ) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v + TP_cd2b * Fc_d2_v
                                                    ) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cd2b <= 10.0:
                        TN_cd2b_4wf = 0
                    else:
                        TN_cd2b_4wf = TN_cd2b
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cd2b_4wf * Fc_d2_v) / (
                                                               area_f + area_pd) / 100
                elif (havePond is False) & (orderDitch == 1):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v) / (
                            area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v) / (
                            area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v) / (
                            area_f + area_pd) / 100

                """ Water quality in the pond """
                if havePond == True:
                    # Calculate mixed concentration first, and then calculate first-order retention/release
                    # Mixed concentration
                    if orderDitch == 2:
                        # total volume in ditches before balance, total_volume_d; and after balance, total_volume_d_bln; in mm.m2
                        total_volume_d = (D + F_lat) * area_f + n1 * n2 * (sd_functions.f_vhd(
                            Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) + prcp * (area_d1 + area_d2) \
                                         - kc_d1 * PET * area_d1_d - kc_d2 * PET * area_d2_d - Fc_d1_v - Fc_d2_v
                        total_volume_d_bln = n1 * n2 * (sd_functions.f_vhd(
                            Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + sd_functions.f_vhd(
                            Hdp1 + HDiff_FD2 - HDiff_FD1, HDiff_FD2, Wd2_up, Wd2_lw, Wf1))
                        # mix flow from ditches with the water in ponds and precipitation water, in mg/L
                        if total_volume_d > total_volume_d_bln:
                            TN_cp_mix = (TN_cd2_out * (total_volume_d - total_volume_d_bln) + TN_cp *
                                         sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cd2_out * (total_volume_d - total_volume_d_bln) + TP_cp *
                                         sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                        else:
                            TN_cp_mix = (TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (sd_functions.f_vhp(
                                Hp, HDiff_FP, area_p, slp_p) + prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (sd_functions.f_vhp(
                                Hp, HDiff_FP, area_p, slp_p) + prcp * area_p - kc_p * PET * area_p_d)
                    elif orderDitch == 1:
                        # total volume in ditches before balance, total_volume_d; and after balance, total_volume_d_bln; in mm.m2
                        total_volume_d = (D + F_lat) * area_f + n1 * sd_functions.f_vhd(
                            Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + prcp * area_d1 - kc_d1 * PET * area_d1_d - Fc_d1_v
                        total_volume_d_bln = n1 * sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)
                        # mix flow from ditches with the water in ponds and precipitation water, in mg/L
                        if total_volume_d > total_volume_d_bln:
                            TN_cp_mix = (TN_cd1_out * (total_volume_d - total_volume_d_bln) +
                                         TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cd1_out * (total_volume_d - total_volume_d_bln) +
                                         TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                        else:
                            TN_cp_mix = (TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                    if TN_cp_mix <= 0:
                        TN_cp_mix = TN_cp
                    if TP_cp_mix <= 0:
                        TP_cp_mix = TP_cp

                    # Calculate the retained concentration at the end of a day in the pond
                    if (Hdp1 + HDiff_FP - HDiff_FD1) > 0:
                        Hp_ave = (Hp + Hdp1 + HDiff_FP - HDiff_FD1) / 2000  # in m
                    if ((Hdp1 + HDiff_FP - HDiff_FD1) > 0) & (TN_cp_mix > ENC0_dp):
                        TN_cp = TN_cp_mix * math.exp(- vf_N_pd * 1 / max(0.05, Hp_ave))
                    else:
                        TN_cp = TN_cp_mix
                    if ((Hdp1 + HDiff_FP - HDiff_FD1) > 0) & (TP_cp_mix > EPC0_dp):
                        TP_cp = TP_cp_mix * math.exp(- vf_P_pd * 1 / max(0.05, Hp_ave))
                    else:
                        TP_cp = TP_cp_mix

                    # Calculate the concentrations at the outflow of the pond
                    if Q_dp > 0:
                        # Prepare the average depth in pond during the retained time
                        Hp_rt = (Hp + HDiff_FP) / 2000  # in m
                        if orderDitch == 2:
                            # Prepare the retained time in pond before outflow
                            TT = (sd_functions.f_vhp(HDiff_FP, HDiff_FP, area_p, slp_p) -
                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p)) / 1000 / Q_d2_out / 3600 / 24  # in d
                            # Calculate with first-order dynamic equation
                            if TN_cp_mix > ENC0_dp:
                                TN_cp_out = TN_cp_mix * math.exp(- vf_N_pd * TT / Hp_rt) * math.exp(
                                    - vf_N_pd * area_p / (Q_d2_out * 3600 * 24))
                            else:
                                TN_cp_out = TN_cp_mix
                            if TP_cp_mix > EPC0_dp:
                                TP_cp_out = TP_cp_mix * math.exp(- vf_P_pd * TT / Hp_rt) * math.exp(
                                    - vf_P_pd * area_p / (Q_d2_out * 3600 * 24))
                            else:
                                TP_cp_out = TP_cp_mix
                        elif orderDitch == 1:
                            # Prepare the retained time in pond before outflow
                            TT = (sd_functions.f_vhp(HDiff_FP, HDiff_FP, area_p, slp_p) -
                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p)) / 1000 / Q_d1_out / 3600 / 24  # in d
                            # Calculate with first-order dynamic equation
                            if TN_cp_mix > ENC0_dp:
                                TN_cp_out = TN_cp_mix * math.exp(- vf_N_pd * TT / Hp_rt) * math.exp(
                                    - vf_N_pd * area_p / (Q_d1_out * 3600 * 24))
                            else:
                                TN_cp_out = TN_cp_mix
                            if TP_cp_mix > EPC0_dp:
                                TP_cp_out = TP_cp_mix * math.exp(- vf_P_pd * TT / Hp_rt) * math.exp(
                                    - vf_P_pd * area_p / (Q_d1_out * 3600 * 24))
                            else:
                                TP_cp_out = TP_cp_mix
                        del Hp_rt
                        del TT
                        TN_out_df[column].iloc[date_n] = TN_cp_out
                        TP_out_df[column].iloc[date_n] = TP_cp_out
                    # Record TN and TP concentration
                    TN_cp_df[column].iloc[date_n] = (TN_cp_mix + TN_cp) / 2
                    TP_cp_df[column].iloc[date_n] = (TP_cp_mix + TP_cp) / 2

                elif orderDitch == 2:
                    TN_out_df[column].iloc[date_n] = TN_cd2_out
                    TP_out_df[column].iloc[date_n] = TP_cd2_out
                else:
                    TN_out_df[column].iloc[date_n] = TN_cd1_out
                    TP_out_df[column].iloc[date_n] = TP_cd1_out

                # Update water level in 1st, 2nd order ditches and in pond, in mm
                Hdp = Hdp1
                Hd1 = Hdp1
                if orderDitch == 2:
                    Hd2 = Hdp1 + HDiff_FD2 - HDiff_FD1
                if havePond is True:
                    Hp = Hdp1 + HDiff_FP - HDiff_FD1

            date_n += 1

    """ Calculate daily concentrations in surface flow from fields and from the IDU outlet """
    TN_cf_out = TN_cf_df.copy()
    TN_cf_out[D_df == 0] = np.NAN
    TP_cf_out = TP_cf_df.copy()
    TP_cf_out[D_df == 0] = np.NAN
    TN_out_df[Qdp_df == 0] = np.NAN
    TP_out_df[Qdp_df == 0] = np.NAN
    """ Calculate the total TN and TP loads, in kg/ha/d """
    # from fields
    TN_fs_df = TN_cf_df * D_df / 100
    TP_fs_df = TP_cf_df * D_df / 100
    TN_fb_lat_df = TN_cfb_lat_df * F_lat_df / 100
    TP_fb_lat_df = TP_cfb_lat_df * F_lat_df / 100
    TN_fb_df = TN_cfb_df * Fc_f_df / 100 + TN_fb_lat_df
    TP_fb_df = TP_cfb_df * Fc_f_df / 100 + TP_fb_lat_df
    TN_f_df = TN_fs_df + TN_fb_df
    TP_f_df = TP_fs_df + TP_fb_df
    # from system unit
    TN_s_df = TN_out_df.multiply(Qdp_df, fill_value=0.0) / 100
    TP_s_df = TP_out_df.multiply(Qdp_df, fill_value=0.0) / 100
    TN_sys_df = TN_s_df + TN_b_df
    TP_sys_df = TP_s_df + TP_b_df
    del TN_cfb_df
    del TP_cfb_df
    # for water footprint calculation
    TN_s_df_4wf = TN_s_df.where(TN_out_df > 2.0, 0.0)
    TP_s_df_4wf = TP_s_df.where(TP_out_df > 0.4, 0.0)
    TN_fs_df_4wf = TN_fs_df.where(TN_cf_df > 2.0, 0.0)
    TP_fs_df_4wf = TP_fs_df.where(TP_cf_df > 0.4, 0.0)
    TN_fb_lat_df_4wf = TN_fb_lat_df.where(TN_cfb_lat_df > 2.0, 0.0)
    TP_fb_lat_df_4wf = TP_fb_lat_df.where(TP_cfb_lat_df > 0.4, 0.0)
    TN_fs_A_4wf = (TN_fs_df_4wf + TN_fb_lat_df_4wf).resample("A", kind="period").sum()
    TP_fs_A_4wf = (TP_fs_df_4wf + TP_fb_lat_df_4wf).resample("A", kind="period").sum()
    TN_s_A_4wf = TN_s_df_4wf.resample("A", kind="period").sum()
    TP_s_A_4wf = TP_s_df_4wf.resample("A", kind="period").sum()
    TN_b_A_4wf = TN_b_df_4wf.resample("A", kind="period").sum()
    loads_4gwf = pd.DataFrame({"TN_runoff": TN_s_A_4wf.values.reshape(-1),
                               "TP_runoff": TP_s_A_4wf.values.reshape(-1),
                               "TN_leaching": TN_b_A_4wf.values.reshape(-1),
                               "TN_fs": TN_fs_A_4wf.values.reshape(-1),
                               "TP_fs": TP_fs_A_4wf.values.reshape(-1)})
    del TN_cfb_lat_df
    del TP_cfb_lat_df
    """ Calculate the proportions of surface and subsurface loads"""
    # from system unit
    TN_s_A = TN_s_df.resample('A', kind='period').sum()
    TN_b_A = TN_b_df.resample('A', kind='period').sum()
    TN_sys_A = TN_sys_df.resample('A', kind='period').sum()
    TP_s_A = TP_s_df.resample('A', kind='period').sum()
    TP_b_A = TP_b_df.resample('A', kind='period').sum()
    TP_sys_A = TP_sys_df.resample('A', kind='period').sum()
    TN_s_pr = TN_s_A / TN_sys_A
    TN_b_pr = TN_b_A / TN_sys_A
    TP_s_pr = TP_s_A / TP_sys_A
    TP_b_pr = TP_b_A / TP_sys_A
    result_sys_pr_df = pd.DataFrame({'TN_load(kg/ha/yr)': TN_sys_A.values.reshape(-1),
                                     'TN_surface(kg/ha/yr)': TN_s_A.values.reshape(-1),
                                     'TN_surface(rate)': TN_s_pr.values.reshape(-1),
                                     'TN_subsurface(kg/ha/yr)': TN_b_A.values.reshape(-1),
                                     'TN_subsurface(rate)': TN_b_pr.values.reshape(-1),
                                     'TP_load(kg/ha/yr)': TP_sys_A.values.reshape(-1),
                                     'TP_surface(kg/ha/yr)': TP_s_A.values.reshape(-1),
                                     'TP_surface(rate)': TP_s_pr.values.reshape(-1),
                                     'TP_subsurface(kg/ha/yr)': TP_b_A.values.reshape(-1),
                                     'TP_subsurface(rate)': TP_b_pr.values.reshape(-1)})
    result_sys_pr = result_sys_pr_df.describe()
    # from fields
    TN_fs_A = TN_fs_df.resample('A', kind='period').sum()
    TN_fb_A = TN_fb_df.resample('A', kind='period').sum()
    TN_f_A = TN_f_df.resample('A', kind='period').sum()
    TP_fs_A = TP_fs_df.resample('A', kind='period').sum()
    TP_fb_A = TP_fb_df.resample('A', kind='period').sum()
    TP_f_A = TP_f_df.resample('A', kind='period').sum()
    TN_fs_pr = TN_fs_A / TN_f_A
    TN_fb_pr = TN_fb_A / TN_f_A
    TP_fs_pr = TP_fs_A / TP_f_A
    TP_fb_pr = TP_fb_A / TP_f_A
    result_f_pr_df = pd.DataFrame({'TN_load(kg/ha/yr)': TN_f_A.values.reshape(-1),
                                   'TN_surface(kg/ha/yr)': TN_fs_A.values.reshape(-1),
                                   'TN_surface(rate)': TN_fs_pr.values.reshape(-1),
                                   'TN_subsurface(kg/ha/yr)': TN_fb_A.values.reshape(-1),
                                   'TN_subsurface(rate)': TN_fb_pr.values.reshape(-1),
                                   'TP_load(kg/ha/yr)': TP_f_A.values.reshape(-1),
                                   'TP_surface(kg/ha/yr)': TP_fs_A.values.reshape(-1),
                                   'TP_surface(rate)': TP_fs_pr.values.reshape(-1),
                                   'TP_subsurface(kg/ha/yr)': TP_fb_A.values.reshape(-1),
                                   'TP_subsurface(rate)': TP_fb_pr.values.reshape(-1)})
    result_f_pr = result_f_pr_df.describe()
    """ Calculate the annual average flow-weighted concentrations, in mg/L """
    # Concentrations in surface flow from fields
    TN_cs_f = TN_fs_A * 100 / (D_df.resample('A', kind='period').sum())
    TP_cs_f = TP_fs_A * 100 / (D_df.resample('A', kind='period').sum())
    # Concentrations in surface flow from system unit outlet
    TN_cs_sys = TN_s_A * 100 / (Qdp_df.resample('A', kind='period').sum())
    TP_cs_sys = TP_s_A * 100 / (Qdp_df.resample('A', kind='period').sum())

    """ Calculate the total nutrient inputs to and outputs from the field-ditch-pond system unit.
        The inputs are from irrigation and precipitation, the outputs are as surface runoff and leaching."""
    irrg_A = I_df.resample('A', kind='period').sum()
    prcp_A = prcp_d.resample('A', kind='period').sum()
    # field scale
    irrg_TN_input_f = irrg_A * TN_I / 100  # in kg/ha/yr
    irrg_TP_input_f = irrg_A * TP_I / 100  # in kg/ha/yr
    prcp_TN_input_f = prcp_A * TN_P / 100  # in kg/ha/yr
    prcp_TP_input_f = prcp_A * TP_P / 100  # in kg/ha/yr
    # IDU scale
    irrg_TN_input_sys = irrg_A * TN_I / 100 * area_f / (area_f + area_pd)  # in kg/ha/yr
    irrg_TP_input_sys = irrg_A * TP_I / 100 * area_f / (area_f + area_pd)  # in kg/ha/yr
    prcp_TN_input_sys = prcp_A * TN_P / 100  # in kg/ha/yr
    prcp_TP_input_sys = prcp_A * TP_P / 100  # in kg/ha/yr
    # Calculate the net nutrient export from fields and from the system unit, respectively.
    TN_export_f = TN_f_A - irrg_TN_input_f - prcp_TN_input_f
    TP_export_f = TP_f_A - irrg_TP_input_f - prcp_TP_input_f
    TN_export_sys = TN_sys_A - irrg_TN_input_sys - prcp_TN_input_sys
    TP_export_sys = TP_sys_A - irrg_TP_input_sys - prcp_TP_input_sys
    # The input-output dataframes
    TN_in_out = pd.concat([pd.DataFrame({"irrigation input": irrg_TN_input_f.values.reshape(-1),
                                         "precipitation input": prcp_TN_input_f.values.reshape(-1),
                                         "runoff loss": TN_fs_A.values.reshape(-1),
                                         "leaching loss": TN_fb_A.values.reshape(-1),
                                         "net export": TN_export_f.values.reshape(-1),
                                         "scale": "field"}),
                           pd.DataFrame({"irrigation input": irrg_TN_input_sys.values.reshape(-1),
                                         "precipitation input": prcp_TN_input_sys.values.reshape(-1),
                                         "runoff loss": TN_s_A.values.reshape(-1),
                                         "leaching loss": TN_b_A.values.reshape(-1),
                                         "net export": TN_export_sys.values.reshape(-1),
                                         "scale": "IDU"})],
                          ignore_index=True)
    TP_in_out = pd.concat([pd.DataFrame({"irrigation input": irrg_TP_input_f.values.reshape(-1),
                                         "precipitation input": prcp_TP_input_f.values.reshape(-1),
                                         "runoff loss": TP_fs_A.values.reshape(-1),
                                         "leaching loss": TP_fb_A.values.reshape(-1),
                                         "net export": TP_export_f.values.reshape(-1),
                                         "scale": "field"}),
                           pd.DataFrame({"irrigation input": irrg_TP_input_sys.values.reshape(-1),
                                         "precipitation input": prcp_TP_input_sys.values.reshape(-1),
                                         "runoff loss": TP_s_A.values.reshape(-1),
                                         "leaching loss": TP_b_A.values.reshape(-1),
                                         "net export": TP_export_sys.values.reshape(-1),
                                         "scale": "IDU"})],
                          ignore_index=True)
    # Calculate the export reduction by ditches and ponds
    TN_reduction = (TN_fs_A * area_f - TN_s_A * (area_f + area_pd)) / (TN_fs_A * area_f) * 100  # in %
    TP_reduction = (TP_fs_A * area_f - TP_s_A * (area_f + area_pd)) / (TP_fs_A * area_f) * 100  # in %
    TN_reduction = TN_reduction.where(TN_reduction > 0, 0)
    TN_reduction = TN_reduction.where(TN_reduction < 100, 100)
    TP_reduction = TP_reduction.where(TP_reduction > 0, 0)
    TP_reduction = TP_reduction.where(TP_reduction < 100, 100)
    result_export_df = pd.DataFrame({'TN_reduction(%)': TN_reduction.values.reshape(-1),
                                     'TP_reduction(%)': TP_reduction.values.reshape(-1)})
    result_export = result_export_df.describe()

    # Prepare data for bmp comparison
    comp_data = {"TN_load": result_sys_pr.loc["mean", "TN_load(kg/ha/yr)"],
                 "TP_load": result_sys_pr.loc["mean", "TP_load(kg/ha/yr)"],
                 "TN_conc": round(np.nanmean(TN_cs_sys.values), 3),
                 "TP_conc": round(np.nanmean(TP_cs_sys.values), 3)}

    """ Write the results into text files """
    if output_dir != 0:
        # Write runoff and leaching loads for water footprint calculation
        with open(output_dir + '/loads_4gwf{}.txt'.format(run_id), mode='w') as output_f:
            output_f.write(
                'The simulated nutrient loads (kg/ha) from IDUs for water footprint calculation are below:\n')
            output_f.close()
        loads_4gwf.to_csv(output_dir + '/loads_4gwf{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                          float_format='%.4f')
        # Write ET_d into files
        with open(output_dir + '/ET_d.txt', mode='w') as output_f:
            output_f.write(
                'The simulated ET (mm/d) from paddy fields are below:\n')
            output_f.close()
        ET_df.to_csv(output_dir + '/ET_d.txt', mode='a', header=True, index=True, sep='\t',
                     float_format='%.4f')
        # Write output summary file
        with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='w') as output_f:
            output_f.write('The simulated nutrient loads from the fields are summarized as below:\n')
            output_f.close()
        result_f_pr.to_csv(output_dir +'/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
        with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
            output_f.write('\nThe simulated nutrient loads from the system unit are summarized as below:\n')
            output_f.close()
        result_sys_pr.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                             float_format='%.2f')
        with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
            output_f.write('\nThe reduction rate by ditches and ponds are summarized as below:\n')
            output_f.close()
        result_export.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                             float_format='%.2f')

        # Write average concentrations
        with open(output_dir + '/TN_cs_sys{}.txt'.format(run_id), mode='w') as output_f:
            output_f.write(
                'The simulated flow-weighted annual average TN concentrations (mg/L) from system unit outlet are:\n')
            output_f.close()
        TN_cs_sys.to_csv(output_dir + '/TN_cs_sys{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                         float_format='%.3f')
        with open(output_dir + '/TP_cs_sys{}.txt'.format(run_id), mode='w') as output_f:
            output_f.write(
                'The simulated flow-weighted annual average TP concentrations (mg/L) from system unit outlet are:\n')
            output_f.close()
        TP_cs_sys.to_csv(output_dir + '/TP_cs_sys{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                         float_format='%.3f')

        # Irrigated water content
        with open(output_dir + '/I_out{}.txt'.format(run_id), 'w') as f:
            f.write("The irrigated water content is simulated as below, in mm.\n")
            f.close()
        irrg_A.to_csv(output_dir + '/I_out{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
        # precipitation water content
        with open(output_dir + '/prcp_A.txt', 'w') as f:
            f.write("The precipitation water content during the growing season is simulated as below, in mm.\n")
            f.close()
        prcp_A.to_csv(output_dir + '/prcp_A.txt', mode='a', header=True, index=True, sep='\t', float_format='%.2f')
        # Annual load from system unit
        with open(output_dir + '/TN_load_sysA{}.txt'.format(run_id), 'w') as output_f:
            output_f.write("The annual TN loads (kg/ha) from the system unit are simulated as:\n")
            output_f.close()
        TN_sys_A.to_csv(output_dir + '/TN_load_sysA{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                        float_format='%.3f')
        with open(output_dir + '/TP_load_sysA{}.txt'.format(run_id), 'w') as output_f:
            output_f.write("The annual TP loads (kg/ha) from the system unit are simulated as:\n")
            output_f.close()
        TP_sys_A.to_csv(output_dir + '/TP_load_sysA{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                        float_format='%.3f')
        # TN reduction percentages
        TN_reduction.to_csv(output_dir + '/TN_reduction_rate{}.txt'.format(run_id), header=True, index=True, sep='\t',
                            float_format='%.2f')
        # TP reduction percentages
        TP_reduction.to_csv(output_dir + '/TP_reduction_rate{}.txt'.format(run_id), header=True, index=True, sep='\t',
                            float_format='%.2f')
        # Nutrient input output results.
        with open(output_dir + '/TN_in_out{}.txt'.format(run_id), mode='w') as f:
            f.write("The TN flux to and from the field V.S. IDU scale is as below, in kg/ha. \n")
            f.close()
        TN_in_out.to_csv(output_dir + '/TN_in_out{}.txt'.format(run_id), mode='a', header=True, index=False, sep='\t',
                         float_format='%.4f')
        with open(output_dir + '/TP_in_out{}.txt'.format(run_id), mode='w') as f:
            f.write("The TP flux to and from the field V.S. IDU scale is as below, in kg/ha. \n")
            f.close()
        TP_in_out.to_csv(output_dir + '/TP_in_out{}.txt'.format(run_id), mode='a', header=True, index=False, sep='\t',
                         float_format='%.4f')

        """ Prepare results summary for GUI """
        output_text = f"""||模拟结果小结||
            \n计算时期内，水稻季的年平均模拟结果如下：
            \n（1）田块尺度\n"""
        output_text += "TN年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
            result_f_pr.loc["mean", "TN_load(kg/ha/yr)"], result_f_pr.loc["mean", "TN_surface(rate)"] * 100,
                                                          result_f_pr.loc["mean", "TN_subsurface(rate)"] * 100)
        output_text += "TP年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
            result_f_pr.loc["mean", "TP_load(kg/ha/yr)"], result_f_pr.loc["mean", "TP_surface(rate)"] * 100,
                                                          result_f_pr.loc["mean", "TP_subsurface(rate)"] * 100)
        output_text += "地表径流中TN年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TN_cs_f.values))
        output_text += "地表径流中TP年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TP_cs_f.values))
        output_text += "\n（2）灌排单元尺度\n"
        output_text += "TN年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
            result_sys_pr.loc["mean", "TN_load(kg/ha/yr)"], result_sys_pr.loc["mean", "TN_surface(rate)"] * 100,
                                                            result_sys_pr.loc["mean", "TN_subsurface(rate)"] * 100)
        output_text += "TP年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
            result_sys_pr.loc["mean", "TP_load(kg/ha/yr)"], result_sys_pr.loc["mean", "TP_surface(rate)"] * 100,
                                                            result_sys_pr.loc["mean", "TP_subsurface(rate)"] * 100)
        output_text += "地表径流中TN年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TN_cs_sys.values))
        output_text += "地表径流中TP年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TP_cs_sys.values))


    if graph == 1:
        """ Convert outputs to time series """
        yearStart = prcp_d.index[0].year
        yearEnd = prcp_d.index[-1].year + 1
        Hdp_df = Hdp_df - HDiff_FD1
        for year in range(yearStart, yearEnd):
            Hf_df1[year] = Hf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            Hdp_df1[year] = Hdp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_cf_df1[year] = TP_cf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_cd1_df1[year] = TP_cd1_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_cf_df1[year] = TN_cf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_cd1_df1[year] = TN_cd1_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            if havePond:
                TP_cp_df1[year] = TP_cp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
                TN_cp_df1[year] = TN_cp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            if orderDitch == 2:
                TP_cd2_df1[year] = TP_cd2_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
                TN_cd2_df1[year] = TN_cd2_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_fs_df1[year] = TN_fs_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_s_df1[year] = TN_s_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_fs_df1[year] = TP_fs_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_s_df1[year] = TP_s_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values

        """Write Time series outputs """
        # water level in fields
        Hf_df1.to_csv(output_dir + '/H_f_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # water level in ditches and pond
        Hdp_df1.to_csv(output_dir + '/H_dp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN concentration in fields
        TN_cf_df1.to_csv(output_dir + '/TN_cf_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP concentration in fields
        TP_cf_df1.to_csv(output_dir + '/TP_cf_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN concentration in first-order ditch
        TN_cd1_df1.to_csv(output_dir + '/TN_cd1_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP concentration in first-order ditch
        TP_cd1_df1.to_csv(output_dir + '/TP_cd1_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        if orderDitch == 2:
            # TN concentration in second-order ditch
            TN_cd2_df1.to_csv(output_dir + '/TN_cd2_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
            # TP concentration in second-order ditch
            TP_cd2_df1.to_csv(output_dir + '/TP_cd2_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        if havePond:
            # TN concentration in pond
            TN_cp_df1.to_csv(output_dir + '/TN_cp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
            # TP concentration in pond
            TP_cp_df1.to_csv(output_dir + '/TP_cp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN load from field with surface runoff
        TN_fs_df1.to_csv(output_dir + '/TN_load_fs_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN load from system with surface runoff
        TN_s_df1.to_csv(output_dir + '/TN_load_syss_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP load from field with surface runoff
        TP_fs_df1.to_csv(output_dir + '/TP_load_fs_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP load from system with surface runoff
        TP_s_df1.to_csv(output_dir + '/TP_load_syss_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')

        """ Prepare dataframe list for drawing graphs """
        if orderDitch == 1:
            TN_cd2_df1 = 0
            TP_cd2_df1 = 0
        if not havePond:
            TN_cp_df1 = 0
            TP_cp_df1 = 0
        graph_dfs = list([Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1,
        TN_cp_df1, TP_cp_df1, TN_fs_df1, TN_s_df1, TP_fs_df1, TP_s_df1, TN_in_out, TP_in_out])

    if (output_dir != 0) & (graph == 1):
        return (output_text, comp_data, graph_dfs)
    elif (output_dir != 0) & (graph == 0):
        return (output_text, comp_data)
    else:
        return comp_data


def pre_risk(prcp_d, startDay, endDay, wtr_mng_d, pet_d, ditch_par, pond_par, F_doy_array, fert_type_array,
             TN0_array, TP0_array, TN_I, TP_I, area_f, Ks, Fc0_f, Fc0_pd, TN_P0, TN_P1, TP_P, TN_rate_lch, TN_rate_lch_lat,
             TP_rate_lch, TP_rate_lch_lat, ENC0_dp, EPC0_dp, ENCR_f, EPCR_f, vf_N_array, vf_P, h_v_sr, vf_N_pd, vf_P_pd,
             risk_days, min_level, Hf_Hdp_sr, run_id, output_dir=0, graph=0):
    """ The pre-risk mode, for only accurate water level management.
        Parameters:
            prcp_d, pandas DataFrame stores daily precipitation data, in mm;
        Outputs:
            Hf"""
    pre_risk_log = PrettyTable(["日期", "预估农田排水量(mm)", "排水前沟塘水位(mm)", "排水后沟塘水位(mm)", "排水量(m3)",
                                "排水平均TN浓度(mg/L)", "排水平均TP浓度(mg/L)"])
    pre_risk_log.align[""] = "l"

    Hf_df, Hdp_df, D_df, F_lat_df, Fc_f_df, Fc_pd_df, Qdp_df, I_df, TN_cf_df, TP_cf_df, TN_cd1_df, TP_cd1_df, \
    TN_cd2_df, TP_cd2_df, TN_cp_df, TP_cp_df, TN_cfb_df, TP_cfb_df, TN_cfb_lat_df, TP_cfb_lat_df, TN_b_df, TP_b_df, \
    TN_out_df, TP_out_df, Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1, \
    TN_cp_df1, TP_cp_df1, TN_fs_df1, TP_fs_df1, TN_s_df1, TP_s_df1 = create_common_df(
        prcp_d, startDay, endDay, wtr_mng_d)
    ET_df = Hf_df.copy(deep=True)

    orderDitch = ditch_par[0]
    if orderDitch == 1:
        Wf1 = ditch_par[1][0]
        HDiff_FD1 = ditch_par[1][1]
        Wd1_up = ditch_par[1][2]
        Wd1_lw = ditch_par[1][3]
        n1 = ditch_par[1][4]
        slp_d1 = ditch_par[1][5]
        n_d1 = ditch_par[1][6]
        kc_d1 = ditch_par[1][7]
        TN_cd10 = ditch_par[1][8]
        TP_cd10 = ditch_par[1][9]
        Hd10 = ditch_par[1][10]
        area_d1 = ditch_par[1][11]
        area_d2 = 0
        Wf2 = area_f / n1 / Wf1
    else:
        Wf1 = ditch_par[1][0]
        HDiff_FD1 = ditch_par[1][1]
        Wd1_up = ditch_par[1][2]
        Wd1_lw = ditch_par[1][3]
        n1 = ditch_par[1][4]
        slp_d1 = ditch_par[1][5]
        n_d1 = ditch_par[1][6]
        kc_d1 = ditch_par[1][7]
        TN_cd10 = ditch_par[1][8]
        TP_cd10 = ditch_par[1][9]
        Hd10 = ditch_par[1][10]
        area_d1 = ditch_par[1][11]
        Wf2 = ditch_par[2][0]
        HDiff_FD2 = ditch_par[2][1]
        Wd2_up = ditch_par[2][2]
        Wd2_lw = ditch_par[2][3]
        n2 = ditch_par[2][4]
        slp_d2 = ditch_par[2][5]
        n_d2 = ditch_par[2][6]
        kc_d2 = ditch_par[2][7]
        TN_cd20 = ditch_par[2][8]
        TP_cd20 = ditch_par[2][9]
        area_d2 = ditch_par[2][10]
    havePond = pond_par[0]
    if havePond:
        HDiff_FP = pond_par[1][1]
        slp_p = pond_par[1][2]
        kc_p = pond_par[1][3]
        TN_cp0 = pond_par[1][4]
        TP_cp0 = pond_par[1][5]
        area_p = pond_par[1][6]
    else:
        area_p = 0.0

    # The rate of area of ditches and pond to the total area, in %
    area_pd = area_d1 + area_d2 + area_p

    startday_doy = pd.Timestamp(2018, int(startDay.split('-')[0]), int(startDay.split('-')[1])).dayofyear
    endday_doy = pd.Timestamp(2018, int(endDay.split('-')[0]), int(endDay.split('-')[1])).dayofyear
    for column in prcp_d.columns:
        date_n = 0
        while date_n < len(prcp_d.index):
            dayofYear = prcp_d.index[date_n].dayofyear
            monthDay = str(prcp_d.index[date_n].month) + "-" + str(prcp_d.index[date_n].day)
            if calendar.isleap(prcp_d.index[date_n].year):
                dayofYear = dayofYear - 1
            if dayofYear == startday_doy:
                water_adj = 0  # flag whether water level adjustment has done, 1 for yes, and 0 for no
                Hf = 0.0
                Hdp = Hd10
                Hd1 = Hd10
                TP_cd1 = TP_cd10
                TN_cd1 = TN_cd10
                if orderDitch == 2:
                    Hd2 = Hd1 + HDiff_FD2 - HDiff_FD1
                    TP_cd2 = TP_cd20
                    TN_cd2 = TN_cd20
                if havePond:
                    Hp = Hd1 + HDiff_FP - HDiff_FD1
                    TP_cp = TP_cp0
                    TN_cp = TN_cp0
                irrg_date = wtr_mng_d["H_min"].ne(0).idxmax()
                irrg_doy = pd.Timestamp(2018, int(irrg_date.split('-')[0]), int(irrg_date.split('-')[1])).dayofyear
                if F_doy_array[0] <= irrg_doy:  # 施肥日在灌溉日之前，田面水浓度初始值为施肥初始浓度
                    TP_cf = TP0_array[0]
                    TN_cf = TN0_array[0]
                else:  # 施肥日在灌溉日之后，田面水浓度初始值为灌溉水浓度
                    TP_cf = TP_I
                    TN_cf = TN_I
                fert_lag = 0

            if (dayofYear < startday_doy) | (dayofYear > endday_doy):
                prcp_d.iloc[date_n] = 0.0  # Change the precipitation in non-growing period to 0, for statistics
            else:
                """ Water balance in fields """
                # Get the precipitation and PET data of the simulation day
                prcp = prcp_d[column].iloc[date_n]
                PET = pet_d[column].iloc[date_n]
                # Calculate ET
                if Hf > 0:
                    ET = PET * wtr_mng_d.loc[monthDay, 'Kc']
                else:
                    ET = PET * 0.60
                ET_df[column].iloc[date_n] = ET
                # Do irrigation or not
                if (wtr_mng_d.loc[monthDay, 'H_min'] != 0) & (Hf < wtr_mng_d.loc[monthDay, 'H_min']) & (
                        prcp + Hf - ET < wtr_mng_d.loc[monthDay, 'H_min']):
                    irrg = wtr_mng_d.loc[monthDay, 'H_max'] - Hf
                else:
                    irrg = 0
                I_df[column].iloc[date_n] = irrg
                # Calculate lateral seepage to ditches
                if orderDitch == 2:
                    F_lat_d1 = max(0, Ks * (Hf - Hd1 + HDiff_FD1) / (0.25 * Wf1 * 1000) * Wf2 * 2 * (
                            Hf + HDiff_FD1))  # in m2.mm/d
                    F_lat_d2 = max(0, Ks * (Hf - Hd2 + HDiff_FD2) / (0.25 * Wf2 * 1000) * Wf1 * 2 * (
                            Hf + HDiff_FD2))  # in m2.mm/d
                    F_lat = (F_lat_d1 + F_lat_d2) / (Wf1 * Wf2)  # in mm/d
                elif orderDitch == 1:
                    F_lat_d1 = max(0, Ks * (Hf - Hd1 + HDiff_FD1) / (0.25 * Wf1 * 1000) * Wf2 * 2 * (
                            Hf + HDiff_FD1))
                    F_lat = F_lat_d1 / (Wf1 * Wf2)  # in mm/d
                F_lat_df[column].iloc[date_n] = F_lat
                # Calculate downward percolation
                if (Hf - ET) > Fc0_f:
                    Fc_f = Fc0_f
                else:
                    Fc_f = 0
                Fc_f_df[column].iloc[date_n] = Fc_f
                # Update water level in fields, Hf, first-round
                Hf1 = Hf + prcp + irrg - ET - Fc_f - F_lat
                # Do drainage or not
                if Hf1 > wtr_mng_d.loc[monthDay, 'H_p']:
                    D = Hf1 - wtr_mng_d.loc[monthDay, 'H_p']  # in mm/d
                    Hf1 = Hf1 - D  # Update again, second-round
                else:
                    D = 0
                if D > 0:
                    water_adj = 1
                D_df[column].iloc[date_n] = D

                """ Water quality in fields"""
                # Initial TN and TP concentrations for the fertilization days, and
                # adjust TP and TN with irrigation, precipitation and ETs.
                flag_N = 0
                flag_P = 0
                TN_P = TN_P1
                for index, F_doy in enumerate(F_doy_array[TN0_array > 0]):
                    if (dayofYear >= F_doy) & (dayofYear - 15 <= F_doy):
                        TN_P = TN_P0
                for index, item in enumerate(F_doy_array):
                    if (dayofYear - fert_lag == item) & (D == 0):
                        fert_lag = 0
                        if TN0_array[index] != 0:
                            TN_cf = TN0_array[index]
                            flag_N = 1
                        if TP0_array[index] != 0:
                            TP_cf = TP0_array[index]
                            flag_P = 1
                    elif (dayofYear - fert_lag == item) & (D > 0):
                        if fert_lag < 5:
                            fert_lag += 1
                        else:
                            fert_lag = 0
                            if TN0_array[index] != 0:
                                TN_cf = TN0_array[index]
                                flag_N = 1
                            if TP0_array[index] != 0:
                                TP_cf = TP0_array[index]
                                flag_P = 1
                # Mixed concentration based on material balance
                if (Hf1 >= 10) & (Hf > 10):
                    if flag_N == 0:
                        TN_cf = (TN_cf * Hf + TN_I * irrg + TN_P * prcp) / (Hf + irrg + prcp - ET)
                    if flag_P == 0:
                        TP_cf = (TP_cf * Hf + TP_I * irrg + TP_P * prcp) / (Hf + irrg + prcp - ET)
                elif (Hf1 >= 10) & (Hf <= 10):
                    temp = max(min(10 + Hf, 10.0), 1)
                    if flag_N == 0:
                        TN_cf = (TN_cf * 2.0 * temp + TN_I * irrg + TN_P * prcp) / (temp + irrg + prcp - ET)
                    if flag_P == 0:
                        TP_cf = (TP_cf * 4.0 * temp + TP_I * irrg + TP_P * prcp) / (temp + irrg + prcp - ET)
                    del temp
                # Adjust TN concentration for 2-7 days after tillering/filling fertilization
                for x, doy in enumerate(F_doy_array[fert_type_array == 2]):
                    if (dayofYear - 2 >= doy) & (dayofYear - 3 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.3
                    elif (dayofYear - 4 >= doy) & (dayofYear - 5 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.2
                    elif (dayofYear - 6 >= doy) & (dayofYear - 7 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.1
                    else:
                        TN_cf = TN_cf

                # Record percolated concentrations
                TN_cfb = TN_cf * TN_rate_lch
                TN_cfb_lat = max(TN_cf * TN_rate_lch_lat, ENC0_dp)
                TP_cfb = TP_cf * TP_rate_lch
                TP_cfb_lat = max(TP_cf * TP_rate_lch_lat, EPC0_dp)
                TP_cfb_df[column].iloc[date_n] = TP_cfb
                TN_cfb_df[column].iloc[date_n] = TN_cfb
                TP_cfb_lat_df[column].iloc[date_n] = TP_cfb_lat
                TN_cfb_lat_df[column].iloc[date_n] = TN_cfb_lat
                # Account for areal dynamics in the water-soil interface
                flag_N = 0
                flag_P = 0
                for index, F_doy in enumerate(F_doy_array[TN0_array > 0]):
                    if index + 1 < np.count_nonzero(TN0_array):
                        if (dayofYear > F_doy) & (dayofYear < F_doy_array[TN0_array > 0][index + 1]) & (Hf > 0) & (
                                Hf1 > 0) & (TN_cf > (ENCR_f * TN0_array[index])) & (flag_N == 0):
                            TN_cf1 = max(TN_cf * math.exp(- vf_N_array[index] * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         TN0_array[index] * ENCR_f)
                            flag_N = 1
                    else:
                        if (dayofYear > F_doy) & (Hf > 0) & (Hf1 > 0) & (TN_cf > (ENCR_f * TN0_array[index])) & (
                                flag_N == 0):
                            TN_cf1 = max(TN_cf * math.exp(- vf_N_array[index] * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         TN0_array[index] * ENCR_f)
                            flag_N = 1
                for index, F_doy in enumerate(F_doy_array[TP0_array > 0]):
                    if index + 1 < np.count_nonzero(TP0_array):
                        if (dayofYear > F_doy) & (dayofYear < F_doy_array[TP0_array > 0][index + 1]) & (Hf > 0) & (
                                Hf1 > 0) & (TP_cf > (EPCR_f * TP0_array[index])) & (flag_P == 0):
                            TP_cf1 = max(TP_cf * math.exp(- vf_P * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         EPCR_f * TP0_array[index])
                            flag_P = 1
                    else:
                        if (dayofYear > F_doy) & (Hf > 0) & (Hf1 > 0) & (TP_cf > (EPCR_f * TP0_array[index])) & (
                                flag_P == 0):
                            TP_cf1 = max(TP_cf * math.exp(- vf_P * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         EPCR_f * TP0_array[index])
                            flag_P = 1
                if flag_P == 0:
                    TP_cf1 = TP_cf
                if flag_N == 0:
                    TN_cf1 = TN_cf
                # Record water table and water quality in fields
                TP_cf_df[column].iloc[date_n] = (TP_cf + TP_cf1) / 2
                TN_cf_df[column].iloc[date_n] = (TN_cf + TN_cf1) / 2
                if Hf1 >= 0:
                    Hf_df[column].iloc[date_n] = Hf1
                else:
                    Hf_df[column].iloc[date_n] = 0.0
                Hf = Hf1

                """ Water balance in ditches and the pond """
                # Decide whether water level adjustment is needed
                if water_adj == 1:
                    for F_doy in F_doy_array:
                        if dayofYear == F_doy:
                            water_adj = 0
                if water_adj == 0:
                    Hf_adj = 0
                    for F_doy in F_doy_array:
                        if (dayofYear >= F_doy) & (dayofYear - risk_days <= F_doy):
                            # 估算农田预计排水量，mm
                            iloc_day = wtr_mng_d.index.get_loc(monthDay)
                            H_p_next = min(wtr_mng_d['H_p'].iloc[iloc_day + 1], wtr_mng_d['H_p'].iloc[iloc_day + 2],
                                           wtr_mng_d['H_p'].iloc[iloc_day + 3])
                            # Adjust water depth based on weather forecast and water requirement of fields
                            prcp_next = prcp_d[column].iloc[date_n + 1]
                            if prcp_next > 100:
                                prcp_next1 = 100
                            elif prcp_next > 50:
                                prcp_next1 = 70
                            elif prcp_next > 25:
                                prcp_next1 = 35
                            else:
                                prcp_next1 = 0
                            if (prcp_next1 > 0) & (Hf + prcp_next1 - H_p_next > 10):  # 降雨前排水
                                Hf_adj = math.ceil((Hf + prcp_next1 - H_p_next - 10) / 5) * 5
                                Hf_adj1 = min(Hf_adj, Hf_Hdp_sr.index.max())
                            elif (prcp_next1 == 0) & (Hf - wtr_mng_d['H_p'].iloc[iloc_day + 1] > 10):  # 提前一天主动排水
                                Hf_adj = math.ceil((Hf - wtr_mng_d['H_p'].iloc[iloc_day + 1] - 10) / 5) * 5
                                Hf_adj1 = min(Hf_adj, Hf_Hdp_sr.index.max())
                    # Do water level adjustment
                    if havePond is True:
                        if (Hf_adj > 0) & (Hdp + HDiff_FP - HDiff_FD1 > min_level):
                            Hdp_adj = max(Hf_Hdp_sr[Hf_adj1], min_level - HDiff_FP + HDiff_FD1)
                            if Hdp_adj < Hdp:
                                water_adj_potential = 1  # 是否实施提前排水，是1， 否0
                                #  计算提前排水量
                                Q_dp_adj = (h_v_sr[Hdp] - h_v_sr[Hdp_adj]) / 1000  # in m3/d
                                if orderDitch == 2:
                                    # 计算提前排水量的平均水质浓度，更新沟塘水位
                                    vd1_prerisk = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                                   sd_functions.f_vhd(Hdp_adj, HDiff_FD1, Wd1_up, Wd1_lw,
                                                                      Wf2)) * n1 * n2
                                    vd2_prerisk = (sd_functions.f_vhd(Hd2, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                                   sd_functions.f_vhd(Hdp_adj + HDiff_FD2 - HDiff_FD1,
                                                                      HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) * n1 * n2
                                    vp_prerisk = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - \
                                                 sd_functions.f_vhp(Hdp_adj + HDiff_FP - HDiff_FD1, HDiff_FP,
                                                                    area_p, slp_p)
                                    TN_prerisk = (TN_cd1 * vd1_prerisk + TN_cd2 * vd2_prerisk + TN_cp * vp_prerisk) / (
                                                         vd1_prerisk + vd2_prerisk + vp_prerisk)
                                    TP_prerisk = (TP_cd1 * vd1_prerisk + TP_cd2 * vd2_prerisk + TP_cp * vp_prerisk) / (
                                                         vd1_prerisk + vd2_prerisk + vp_prerisk)
                                    Hd1 = Hdp_adj
                                    Hd2 = Hdp_adj + HDiff_FD2 - HDiff_FD1
                                    Hp = Hdp_adj + HDiff_FP - HDiff_FD1
                                else:
                                    vd1_prerisk = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                                   sd_functions.f_vhd(Hdp_adj, HDiff_FD1, Wd1_up, Wd1_lw,
                                                                      Wf2)) * n1
                                    vp_prerisk = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - \
                                                 sd_functions.f_vhp(Hdp_adj + HDiff_FP - HDiff_FD1, HDiff_FP,
                                                                    area_p, slp_p)
                                    TN_prerisk = (TN_cd1 * vd1_prerisk + TN_cp * vp_prerisk) / (
                                            vd1_prerisk + vp_prerisk)
                                    TP_prerisk = (TP_cd1 * vd1_prerisk + TP_cp * vp_prerisk) / (
                                            vd1_prerisk + vp_prerisk)
                                    Hd1 = Hdp_adj
                                    Hp = Hdp_adj + HDiff_FP - HDiff_FD1
                            else:
                                water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                        else:
                            water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                    elif orderDitch == 2:
                        if (Hf_adj > 0) & (Hdp + HDiff_FD2 - HDiff_FD1 > min_level):
                            Hdp_adj = max(Hf_Hdp_sr[Hf_adj1], min_level - HDiff_FD2 + HDiff_FD1)
                            if Hdp_adj < Hdp:
                                water_adj_potential = 1  # 是否实施提前排水，是1， 否0
                                #  计算提前排水量
                                Q_dp_adj = (h_v_sr[Hdp] - h_v_sr[Hdp_adj]) / 1000  # in m3/d
                                # 计算提前排水量的平均水质浓度，更新沟塘水位
                                vd1_prerisk = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                               sd_functions.f_vhd(Hdp_adj, HDiff_FD1, Wd1_up, Wd1_lw,
                                                                  Wf2)) * n1 * n2
                                vd2_prerisk = (sd_functions.f_vhd(Hd2, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                               sd_functions.f_vhd(Hdp_adj + HDiff_FD2 - HDiff_FD1,
                                                                  HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) * n1 * n2
                                TN_prerisk = (TN_cd1 * vd1_prerisk + TN_cd2 * vd2_prerisk) / (
                                        vd1_prerisk + vd2_prerisk)
                                TP_prerisk = (TP_cd1 * vd1_prerisk + TP_cd2 * vd2_prerisk) / (
                                        vd1_prerisk + vd2_prerisk)
                                Hd1 = Hdp_adj
                                Hd2 = Hdp_adj + HDiff_FD2 - HDiff_FD1
                            else:
                                water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                        else:
                            water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                    elif orderDitch == 1:
                        if (Hf_adj > 0) & (Hdp > min_level):
                            Hdp_adj = max(Hf_Hdp_sr[Hf_adj1], min_level)
                            if Hdp_adj < Hdp:
                                water_adj_potential = 1  # 是否实施提前排水，是1， 否0
                                # 计算提前排水量
                                Q_dp_adj = (h_v_sr[Hdp] - h_v_sr[Hdp_adj]) / 1000  # in m3/d
                                # 计算提前排水量的平均水质浓度，更新沟塘水位
                                TN_prerisk = TN_cd1
                                TP_prerisk = TP_cd1
                                Hd1 = Hdp_adj
                            else:
                                water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                        else:
                            water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                    if water_adj_potential == 1:
                        pre_risk_log.add_row(
                            [prcp_d.index[date_n], Hf_adj, round(Hdp), round(Hdp_adj), round(Q_dp_adj),
                             round(TN_prerisk, 2), round(TP_prerisk, 2)])
                        Hdp = Hdp_adj
                        del Hf_adj
                        del Hf_adj1
                        del Hdp_adj
                else:
                    water_adj_potential = 0

                # Calculate ET and Fc from ditches and the pond
                if Hd1 > 0:
                    Wd1_d = (Wd1_up - Wd1_lw) * Hd1 / HDiff_FD1 + Wd1_lw
                    if orderDitch == 2:
                        area_d1_d = Wd1_d * Wf2 * n1 * n2
                    elif orderDitch == 1:
                        area_d1_d = Wd1_d * Wf2 * n1
                    if Hd1 - Fc0_pd > (kc_d1 * PET):
                        Fc_d1_v = Fc0_pd * area_d1_d
                    else:
                        Fc_d1_v = 0.0
                else:
                    area_d1_d = 0
                    Fc_d1_v = 0.0

                if orderDitch == 2:
                    if Hd2 > 0:
                        Wd2_d = (Wd2_up - Wd2_lw) * Hd2 / HDiff_FD2 + Wd2_lw
                        area_d2_d = Wd2_d * Wf1 * n1 * n2
                        if Hd2 - Fc0_pd > (kc_d2 * PET):
                            Fc_d2_v = Fc0_pd * area_d2_d
                        else:
                            Fc_d2_v = 0.0
                    else:
                        area_d2_d = 0
                        Fc_d2_v = 0.0

                if havePond == True:
                    if Hp > 0:
                        area_p_lw = (math.sqrt(area_p) - HDiff_FP / 1000 * slp_p * 2) ** 2  # in m2
                        hp1 = math.sqrt(area_p_lw) / 2 / slp_p  # in m
                        area_p_d = area_p_lw * ((hp1 + Hp / 1000) ** 2) / (hp1 ** 2)  # in m2
                        if Hp - Fc0_pd > (kc_p * PET):
                            Fc_p_v = Fc0_pd * area_p_d
                        else:
                            Fc_p_v = 0.0
                    else:
                        area_p_d = 0
                        Fc_p_v = 0.0
                # ET, ET_pd;, downward percoloation, Fc_pd
                if (havePond is True) & (orderDitch == 2):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_d2 * PET * area_d2_d + kc_p * PET * area_p_d + \
                              0.60 * PET * (area_pd - area_d1_d - area_d2_d - area_p_d)
                    Fc_pd_v = Fc_d1_v + Fc_d2_v + Fc_p_v
                elif (havePond is True) & (orderDitch == 1):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_p * PET * area_p_d + 0.60 * PET * (
                            area_pd - area_d1_d - area_p_d)
                    Fc_pd_v = Fc_d1_v + Fc_p_v
                elif (havePond is False) & (orderDitch == 2):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_d2 * PET * area_d2_d + 0.60 * PET * (
                            area_pd - area_d1_d - area_d2_d)
                    Fc_pd_v = Fc_d1_v + Fc_d2_v
                elif (havePond is False) & (orderDitch == 1):
                    ET_pd_v = kc_d1 * PET * area_d1_d + 0.60 * PET * (
                            area_pd - area_d1_d)
                    Fc_pd_v = Fc_d1_v

                # Update water level in ditches and ponds, Hdp; and calculate outflow Q_dp of the system from the pond
                volume_add = prcp * area_pd - Fc_pd_v - ET_pd_v + (F_lat + D) * area_f
                Q_dp = sd_functions.f_vhpd(Hdp, volume_add, h_v_sr, area_pd, HDiff_FD1)[0]  # in m3/d
                Hdp1 = sd_functions.f_vhpd(Hdp, volume_add, h_v_sr, area_pd, HDiff_FD1)[1]  # in mm
                if water_adj_potential == 1:
                    Qdp_df[column].iloc[date_n] = (Q_dp + Q_dp_adj) / (area_f + area_pd) * 1000  # in mm/d
                else:
                    Qdp_df[column].iloc[date_n] = Q_dp / (area_f + area_pd) * 1000  # in mm/d
                Hdp_df[column].iloc[date_n] = Hdp1

                """ Water quality in ditches """
                # 1st order ditch
                # Record percolation concentration
                TN_cd1b = TN_cd1 * TN_rate_lch
                TP_cd1b = TP_cd1 * TP_rate_lch
                # Calculate mixed concentration
                if orderDitch == 2:
                    TN_cd1_out = (TN_cf * D * Wf1 * Wf2 + TN_cfb_lat * F_lat_d1 + TN_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TN_P * prcp * area_d1 / n1 / n2
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1 / n2)
                    TP_cd1_out = (TP_cf * D * Wf1 * Wf2 + TP_cfb_lat * F_lat_d1 + TP_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TP_P * prcp * area_d1 / n1 / n2
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1 / n2)
                elif orderDitch == 1:
                    TN_cd1_out = (TN_cf * D * Wf1 * Wf2 + TN_cfb_lat * F_lat_d1 + TN_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TN_P * prcp * area_d1 / n1
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1)
                    TP_cd1_out = (TP_cf * D * Wf1 * Wf2 + TP_cfb_lat * F_lat_d1 + TP_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TP_P * prcp * area_d1 / n1
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1)
                if TN_cd1_out <= 0:
                    TN_cd1_out = TN_cd1
                if TP_cd1_out <= 0:
                    TP_cd1_out = TP_cd1
                # Calculate retained concentration
                if Hdp1 > 0:
                    if TN_cd1 > ENC0_dp:
                        TN_cd1 = max(TN_cd1_out * math.exp(- vf_N_pd * 1 / max(0.05, Hdp1 / 1000)), ENC0_dp)
                    else:
                        TN_cd1 = TN_cd1_out
                    if TP_cd1_out > EPC0_dp:
                        TP_cd1 = max(TP_cd1_out * math.exp(- vf_P_pd * 1 / max(0.05, Hdp1 / 1000)), EPC0_dp)
                    else:
                        TP_cd1 = TP_cd1_out
                else:
                    TN_cd1 = TN_cd1_out
                    TP_cd1 = TP_cd1_out
                # Record concentrations
                TN_cd1_df[column].iloc[date_n] = (TN_cd1_out + TN_cd1) / 2
                TP_cd1_df[column].iloc[date_n] = (TP_cd1_out + TP_cd1) / 2
                TN_cf = TN_cf1
                TP_cf = TP_cf1
                # Prepare flow rate (m3/s) leaving the 1st order ditch for pond retention calculation
                if (Q_dp > 0) & (orderDitch == 1):
                    A_d1 = (Wd1_up + Wd1_lw) * HDiff_FD1 / 1000 / 2  # in m2，截面积
                    if D > 0:
                        Q_f0 = D * area_f / n1 / 24 / 3600 / 1000  # in m3/s,入流量
                        v_d1_out = (Q_f0 ** 0.4) * (slp_d1 ** 0.3) / (n_d1 ** 0.6) / (Wd1_up ** 0.4)  # in m/s，流速
                        Q_d1_out = v_d1_out * A_d1 * n1  # in m3/s,出流量
                        del v_d1_out
                    else:
                        Pw_d1 = Wd1_lw + (((Wd1_up - Wd1_lw) / 2) ** 2 + (HDiff_FD1 / 1000) ** 2) ** 0.5 * 2  # Wet perimeter, in m
                        Q_d1_out = (A_d1 ** 1.67) * (slp_d1 ** 0.5) / n_d1 / (Pw_d1 ** 0.67) * n1  # in m3/s
                        del Pw_d1
                    del A_d1

                # 2nd order ditch
                if orderDitch == 2:
                    # Record percolation concentration
                    TN_cd2b = TN_cd2 * TN_rate_lch
                    TP_cd2b = TP_cd2 * TP_rate_lch
                    # Prepare flow rate (m3/s) of the 2nd order ditch
                    if Hd2 > 0:
                        Pw_d2 = Wd2_lw + (((Wd2_up - Wd2_lw) / 2) ** 2 + (HDiff_FD2 / 1000) ** 2) ** 0.5 * \
                                Hd2 / HDiff_FD2 * 2  # Wet perimeter, in m
                        Wd2 = Wd2_lw + (Wd2_up - Wd2_lw) * Hd2 / HDiff_FD2  # water surface width, in m
                        A_d2 = (Wd2 + Wd2_lw) * Hd2 / 2000  # in m2
                        Q_d2 = (A_d2 ** 1.67) * (slp_d2 ** 0.5) / n_d2 / (Pw_d2 ** 0.67)  # in m3/s
                        del Pw_d2
                        del Wd2
                        del A_d2
                    else:
                        Q_d2 = 0.01
                    if D > 0:
                        # Calculate concentrations at the outlet of the 2nd order ditch by one-dimentional water quality model
                        # Prepare flow rate (m3/s) of 1st and 2nd order ditches for mixed concentration estimation #
                        Q_d1_0 = (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)
                                  + (prcp * area_d1 - kc_d1 * PET * area_d1_d - Fc_d1_v) / n1 / n2 - sd_functions.f_vhd(
                                    Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) / 1000  # in m3
                        TN_cd2_mix = (TN_cfb_lat * F_lat_d2 + TN_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TN_P * prcp * area_d2 / n1 / n2 + TN_cd1_out *
                                      Q_d1_0 * 1000) / (F_lat_d2 + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (prcp * area_d2 - kc_d2 * PET * area_d2_d
                                                                    ) / n1 / n2 + Q_d1_0 * 1000)
                        TP_cd2_mix = (TP_cfb_lat * F_lat_d2 + TP_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TP_P * prcp * area_d2 / n1 / n2 + TP_cd1_out *
                                      Q_d1_0 * 1000) / (F_lat_d2 + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (prcp * area_d2 - kc_d2 * PET * area_d2_d
                                                                    ) / n1 / n2 + Q_d1_0 * 1000)
                        # water surface width, in m
                        if Q_dp > 0:
                            Wd2 = Wd2_up
                            Hd2_d = HDiff_FD2 / 1000  # in m
                        else:
                            Hd2_d = max(0.01, (Hd2 + Hdp1 + HDiff_FD2 - HDiff_FD1) / 2000)  # in m
                            Wd2 = Wd2_lw + (Wd2_up - Wd2_lw) * Hd2_d * 1000 / HDiff_FD2
                        v_d2 = ((Q_d1_0 / 24 / 3600) ** 0.4) * (slp_d2 ** 0.3) / (n_d2 ** 0.6) / (
                                Wd2 ** 0.4)  # in m/s
                        # Calculate concentrations at the outlet
                        TN_rate_array = np.zeros(n1)
                        TP_rate_array = np.zeros(n1)
                        for i in range(0, n1):
                            TN_rate_array[i] = math.exp(- vf_N_pd / Hd2_d * (Wf1 * (i + 0.5) / v_d2 / 3600 / 24))
                            TP_rate_array[i] = math.exp(- vf_P_pd / Hd2_d * Wf1 * (i + 0.5) / v_d2 / 3600 / 24)
                        if TN_cd2_mix > ENC0_dp:
                            TN_cd2_out = TN_cd2_mix * TN_rate_array.mean()
                        else:
                            TN_cd2_out = TN_cd2_mix
                        if TP_cd2_mix > EPC0_dp:
                            TP_cd2_out = TP_cd2_mix * TP_rate_array.mean()
                        else:
                            TP_cd2_out = TP_cd2_mix
                        # Calculate the flow discharge at the outlet of the second-order ditch
                        v_d2_out = ((Q_d1_0 * n1 / 24 / 3600) ** 0.4) * (slp_d2 ** 0.3) / (n_d2 ** 0.6) / \
                                   (Wd2 ** 0.4)  # in m/s
                        A_d2 = (Wd2 + Wd2_lw) * Hd2_d / 2  # in m2
                        Q_d2_out = v_d2_out * A_d2 * n2  # in m3/s
                        del Wd2
                        del Hd2_d
                        del v_d2_out
                        del v_d2
                        del A_d2
                        del TN_rate_array
                        del TP_rate_array
                    else:
                        # Calculate mixed concentration with lateral flow, precipitation and ET
                        TN_cd2_mix = (TN_cfb_lat * F_lat_d2 + TN_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TN_P * prcp * area_d2 / n1 / n2) / (
                                             F_lat_d2 + sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (
                                             prcp * area_d2 - kc_d2 * PET * area_d2_d) / n1 / n2)
                        TP_cd2_mix = (TP_cfb_lat * F_lat_d2 + TP_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TP_P * prcp * area_d2 / n1 / n2) / (
                                             F_lat_d2 + sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (
                                             prcp * area_d2 - kc_d2 * PET * area_d2_d) / n1 / n2)
                        if TN_cd2_mix <= 0:
                            TN_cd2_mix = TN_cd2
                        if TP_cd2_mix <= 0:
                            TP_cd2_mix = TP_cd2
                        TN_cd2_out = TN_cd2_mix
                        TP_cd2_out = TP_cd2_mix
                        # Calculate the flow discharge at the outlet of the second-order ditch
                        Q_d2_out = Q_d2 * n2
                    # Calculate the retained concentration within the second-order ditch
                    if (Hdp1 + HDiff_FD2 - HDiff_FD1) > 0:
                        if TN_cd2_mix > ENC0_dp:
                            TN_cd2 = max(TN_cd2_mix * math.exp(
                                - vf_N_pd * 1 / max(0.05, (Hdp1 + HDiff_FD2 - HDiff_FD1) / 1000)), ENC0_dp)
                        else:
                            TN_cd2 = TN_cd2_mix
                        if TP_cd2_mix > EPC0_dp:
                            TP_cd2 = max(TP_cd2_mix * math.exp(
                                - vf_P_pd * 1 / max(0.05, (Hdp1 + HDiff_FD2 - HDiff_FD1) / 1000)), EPC0_dp)
                        else:
                            TP_cd2 = TP_cd2_mix
                    else:
                        TN_cd2 = TN_cd2_mix
                        TP_cd2 = TP_cd2_mix
                    # Record concentrations
                    TN_cd2_df[column].iloc[date_n] = (TN_cd2_out + TN_cd2) / 2
                    TP_cd2_df[column].iloc[date_n] = (TP_cd2_out + TP_cd2) / 2

                """ Record percolation TN and TP losses from the IDU """
                if havePond is True:
                    # Record percolation concentrations
                    TN_cpb = TN_cp * TN_rate_lch
                    TP_cpb = TP_cp * TP_rate_lch
                if (havePond is True) & (orderDitch == 2):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v + TN_cd2b * Fc_d2_v +
                                                    TN_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v + TP_cd2b * Fc_d2_v +
                                                    TP_cpb * Fc_p_v) / (area_f + area_pd) / 100
                elif (havePond is True) & (orderDitch == 1):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v +
                                                    TN_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v +
                                                    TP_cpb * Fc_p_v) / (area_f + area_pd) / 100
                elif (havePond is False) & (orderDitch == 2):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v + TN_cd2b * Fc_d2_v
                                                    ) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v + TP_cd2b * Fc_d2_v
                                                    ) / (area_f + area_pd) / 100
                elif (havePond is False) & (orderDitch == 1):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v) / (
                            area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v) / (
                            area_f + area_pd) / 100

                """ Water quality in the pond """
                if havePond == True:
                    # Calculate mixed concentration first, and then calculate first-order retention/release
                    # Mixed concentration
                    if orderDitch == 2:
                        # total volume in ditches before balance, total_volume_d; and after balance, total_volume_d_bln; in mm.m2
                        total_volume_d = (D + F_lat) * area_f + n1 * n2 * (sd_functions.f_vhd(
                            Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) + prcp * (area_d1 + area_d2) \
                                         - kc_d1 * PET * area_d1_d - kc_d2 * PET * area_d2_d - Fc_d1_v - Fc_d2_v
                        total_volume_d_bln = n1 * n2 * (sd_functions.f_vhd(
                            Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + sd_functions.f_vhd(
                            Hdp1 + HDiff_FD2 - HDiff_FD1, HDiff_FD2, Wd2_up, Wd2_lw, Wf1))
                        # mix flow from ditches with the water in ponds and precipitation water, in mg/L
                        if total_volume_d > total_volume_d_bln:
                            TN_cp_mix = (TN_cd2_out * (total_volume_d - total_volume_d_bln) + TN_cp *
                                         sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cd2_out * (total_volume_d - total_volume_d_bln) + TP_cp *
                                         sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                        else:
                            TN_cp_mix = (TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (sd_functions.f_vhp(
                                Hp, HDiff_FP, area_p, slp_p) + prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (sd_functions.f_vhp(
                                Hp, HDiff_FP, area_p, slp_p) + prcp * area_p - kc_p * PET * area_p_d)
                    elif orderDitch == 1:
                        # total volume in ditches before balance, total_volume_d; and after balance, total_volume_d_bln; in mm.m2
                        total_volume_d = (D + F_lat) * area_f + n1 * sd_functions.f_vhd(
                            Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + prcp * area_d1 - kc_d1 * PET * area_d1_d - Fc_d1_v
                        total_volume_d_bln = n1 * sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)
                        # mix flow from ditches with the water in ponds and precipitation water, in mg/L
                        if total_volume_d > total_volume_d_bln:
                            TN_cp_mix = (TN_cd1_out * (total_volume_d - total_volume_d_bln) +
                                         TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cd1_out * (total_volume_d - total_volume_d_bln) +
                                         TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                        else:
                            TN_cp_mix = (TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                    if TN_cp_mix <= 0:
                        TN_cp_mix = TN_cp
                    if TP_cp_mix <= 0:
                        TP_cp_mix = TP_cp

                    # Calculate the retained concentration at the end of a day in the pond
                    if (Hdp1 + HDiff_FP - HDiff_FD1) > 0:
                        Hp_ave = (Hp + Hdp1 + HDiff_FP - HDiff_FD1) / 2000  # in m
                    if ((Hdp1 + HDiff_FP - HDiff_FD1) > 0) & (TN_cp_mix > ENC0_dp):
                        TN_cp = TN_cp_mix * math.exp(- vf_N_pd * 1 / max(0.05, Hp_ave))
                    else:
                        TN_cp = TN_cp_mix
                    if ((Hdp1 + HDiff_FP - HDiff_FD1) > 0) & (TP_cp_mix > EPC0_dp):
                        TP_cp = TP_cp_mix * math.exp(- vf_P_pd * 1 / max(0.05, Hp_ave))
                    else:
                        TP_cp = TP_cp_mix

                    # Calculate the concentrations at the outflow of the pond
                    if Q_dp > 0:
                        # Prepare the average depth in pond during the retained time
                        Hp_rt = (Hp + HDiff_FP) / 2000  # in m
                        if orderDitch == 2:
                            # Prepare the retained time in pond before outflow
                            TT = (sd_functions.f_vhp(HDiff_FP, HDiff_FP, area_p, slp_p) -
                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p)) / 1000 / Q_d2_out / 3600 / 24  # in d
                            # Calculate with first-order dynamic equation
                            if TN_cp_mix > ENC0_dp:
                                TN_cp_out = TN_cp_mix * math.exp(- vf_N_pd * TT / Hp_rt) * math.exp(
                                    - vf_N_pd * area_p / (Q_d2_out * 3600 * 24))
                            else:
                                TN_cp_out = TN_cp_mix
                            if TP_cp_mix > EPC0_dp:
                                TP_cp_out = TP_cp_mix * math.exp(- vf_P_pd * TT / Hp_rt) * math.exp(
                                    - vf_P_pd * area_p / (Q_d2_out * 3600 * 24))
                            else:
                                TP_cp_out = TP_cp_mix
                        elif orderDitch == 1:
                            # Prepare the retained time in pond before outflow
                            TT = (sd_functions.f_vhp(HDiff_FP, HDiff_FP, area_p, slp_p) -
                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p)) / 1000 / Q_d1_out / 3600 / 24  # in d
                            # Calculate with first-order dynamic equation
                            if TN_cp_mix > ENC0_dp:
                                TN_cp_out = TN_cp_mix * math.exp(- vf_N_pd * TT / Hp_rt) * math.exp(
                                    - vf_N_pd * area_p / (Q_d1_out * 3600 * 24))
                            else:
                                TN_cp_out = TN_cp_mix
                            if TP_cp_mix > EPC0_dp:
                                TP_cp_out = TP_cp_mix * math.exp(- vf_P_pd * TT / Hp_rt) * math.exp(
                                    - vf_P_pd * area_p / (Q_d1_out * 3600 * 24))
                            else:
                                TP_cp_out = TP_cp_mix
                        del Hp_rt
                        del TT
                        if water_adj_potential == 1:
                            TN_out_df[column].iloc[date_n] = (TN_cp_out * Q_dp + TN_prerisk * Q_dp_adj) / (
                                    Q_dp + Q_dp_adj)
                            TP_out_df[column].iloc[date_n] = (TP_cp_out * Q_dp + TP_prerisk * Q_dp_adj) / (
                                    Q_dp + Q_dp_adj)
                            del Q_dp_adj
                            del TN_prerisk
                            del TP_prerisk
                        else:
                            TN_out_df[column].iloc[date_n] = TN_cp_out
                            TP_out_df[column].iloc[date_n] = TP_cp_out
                    elif water_adj_potential == 1:
                        TN_out_df[column].iloc[date_n] = TN_prerisk
                        TP_out_df[column].iloc[date_n] = TP_prerisk
                        del Q_dp_adj
                        del TN_prerisk
                        del TP_prerisk
                    # Record TN and TP concentration
                    TN_cp_df[column].iloc[date_n] = (TN_cp_mix + TN_cp) / 2
                    TP_cp_df[column].iloc[date_n] = (TP_cp_mix + TP_cp) / 2

                elif orderDitch == 2:
                    if water_adj_potential == 1:
                        TN_out_df[column].iloc[date_n] = (TN_cd2_out * Q_dp + TN_prerisk * Q_dp_adj) / (Q_dp + Q_dp_adj)
                        TP_out_df[column].iloc[date_n] = (TP_cd2_out * Q_dp + TP_prerisk * Q_dp_adj) / (Q_dp + Q_dp_adj)
                        del Q_dp_adj
                        del TN_prerisk
                        del TP_prerisk
                    else:
                        TN_out_df[column].iloc[date_n] = TN_cd2_out
                        TP_out_df[column].iloc[date_n] = TP_cd2_out
                else:
                    if water_adj_potential == 1:
                        TN_out_df[column].iloc[date_n] = (TN_cd1_out * Q_dp + TN_prerisk * Q_dp_adj) / (Q_dp + Q_dp_adj)
                        TP_out_df[column].iloc[date_n] = (TP_cd1_out * Q_dp + TP_prerisk * Q_dp_adj) / (Q_dp + Q_dp_adj)
                        del Q_dp_adj
                        del TN_prerisk
                        del TP_prerisk
                    else:
                        TN_out_df[column].iloc[date_n] = TN_cd1_out
                        TP_out_df[column].iloc[date_n] = TP_cd1_out

                # Update water level in 1st, 2nd order ditches and in pond, in mm
                Hdp = Hdp1
                Hd1 = Hdp1
                if orderDitch == 2:
                    Hd2 = Hdp1 + HDiff_FD2 - HDiff_FD1
                if havePond is True:
                    Hp = Hdp1 + HDiff_FP - HDiff_FD1

            date_n += 1

    """ Calculate daily concentrations in surface flow from fields and from the IDU outlet """
    TN_cf_out = TN_cf_df.copy()
    TN_cf_out[D_df == 0] = np.NAN
    TP_cf_out = TP_cf_df.copy()
    TP_cf_out[D_df == 0] = np.NAN
    TN_out_df[Qdp_df == 0] = np.NAN
    TP_out_df[Qdp_df == 0] = np.NAN
    """ Calculate the total TN and TP loads, in kg/ha/d """
    # from fields
    TN_fs_df = TN_cf_df * D_df / 100
    TP_fs_df = TP_cf_df * D_df / 100
    TN_fb_lat_df = TN_cfb_lat_df * F_lat_df / 100
    TP_fb_lat_df = TP_cfb_lat_df * F_lat_df / 100
    TN_fb_df = TN_cfb_df * Fc_f_df / 100 + TN_fb_lat_df
    TP_fb_df = TP_cfb_df * Fc_f_df / 100 + TP_fb_lat_df
    TN_f_df = TN_fs_df + TN_fb_df
    TP_f_df = TP_fs_df + TP_fb_df
    # from system unit
    TN_s_df = TN_out_df.multiply(Qdp_df, fill_value=0.0) / 100
    TP_s_df = TP_out_df.multiply(Qdp_df, fill_value=0.0) / 100
    TN_sys_df = TN_s_df + TN_b_df
    TP_sys_df = TP_s_df + TP_b_df
    del TN_cfb_df
    del TP_cfb_df
    del TN_cfb_lat_df
    del TP_cfb_lat_df
    """ Calculate the proportions of surface and subsurface loads"""
    # from system unit
    TN_s_A = TN_s_df.resample('A', kind='period').sum()
    TN_b_A = TN_b_df.resample('A', kind='period').sum()
    TN_sys_A = TN_sys_df.resample('A', kind='period').sum()
    TP_s_A = TP_s_df.resample('A', kind='period').sum()
    TP_b_A = TP_b_df.resample('A', kind='period').sum()
    TP_sys_A = TP_sys_df.resample('A', kind='period').sum()
    TN_s_pr = TN_s_A / TN_sys_A
    TN_b_pr = TN_b_A / TN_sys_A
    TP_s_pr = TP_s_A / TP_sys_A
    TP_b_pr = TP_b_A / TP_sys_A
    result_sys_pr_df = pd.DataFrame({'TN_load(kg/ha/yr)': TN_sys_A.values.reshape(-1),
                                     'TN_surface(kg/ha/yr)': TN_s_A.values.reshape(-1),
                                     'TN_surface(rate)': TN_s_pr.values.reshape(-1),
                                     'TN_subsurface(kg/ha/yr)': TN_b_A.values.reshape(-1),
                                     'TN_subsurface(rate)': TN_b_pr.values.reshape(-1),
                                     'TP_load(kg/ha/yr)': TP_sys_A.values.reshape(-1),
                                     'TP_surface(kg/ha/yr)': TP_s_A.values.reshape(-1),
                                     'TP_surface(rate)': TP_s_pr.values.reshape(-1),
                                     'TP_subsurface(kg/ha/yr)': TP_b_A.values.reshape(-1),
                                     'TP_subsurface(rate)': TP_b_pr.values.reshape(-1)})
    result_sys_pr = result_sys_pr_df.describe()
    # from fields
    TN_fs_A = TN_fs_df.resample('A', kind='period').sum()
    TN_fb_A = TN_fb_df.resample('A', kind='period').sum()
    TN_f_A = TN_f_df.resample('A', kind='period').sum()
    TP_fs_A = TP_fs_df.resample('A', kind='period').sum()
    TP_fb_A = TP_fb_df.resample('A', kind='period').sum()
    TP_f_A = TP_f_df.resample('A', kind='period').sum()
    TN_fs_pr = TN_fs_A / TN_f_A
    TN_fb_pr = TN_fb_A / TN_f_A
    TP_fs_pr = TP_fs_A / TP_f_A
    TP_fb_pr = TP_fb_A / TP_f_A
    result_f_pr_df = pd.DataFrame({'TN_load(kg/ha/yr)': TN_f_A.values.reshape(-1),
                                   'TN_surface(kg/ha/yr)': TN_fs_A.values.reshape(-1),
                                   'TN_surface(rate)': TN_fs_pr.values.reshape(-1),
                                   'TN_subsurface(kg/ha/yr)': TN_fb_A.values.reshape(-1),
                                   'TN_subsurface(rate)': TN_fb_pr.values.reshape(-1),
                                   'TP_load(kg/ha/yr)': TP_f_A.values.reshape(-1),
                                   'TP_surface(kg/ha/yr)': TP_fs_A.values.reshape(-1),
                                   'TP_surface(rate)': TP_fs_pr.values.reshape(-1),
                                   'TP_subsurface(kg/ha/yr)': TP_fb_A.values.reshape(-1),
                                   'TP_subsurface(rate)': TP_fb_pr.values.reshape(-1)})
    result_f_pr = result_f_pr_df.describe()
    """ Calculate the annual average flow-weighted concentrations, in mg/L """
    # Concentrations in surface flow from fields
    TN_cs_f = TN_fs_A * 100 / (D_df.resample('A', kind='period').sum())
    TP_cs_f = TP_fs_A * 100 / (D_df.resample('A', kind='period').sum())
    # Concentrations in surface flow from system unit outlet
    TN_cs_sys = TN_s_A * 100 / (Qdp_df.resample('A', kind='period').sum())
    TP_cs_sys = TP_s_A * 100 / (Qdp_df.resample('A', kind='period').sum())

    """ Calculate the total nutrient inputs to and outputs from the field-ditch-pond system unit.
        The inputs are from irrigation and precipitation, the outputs are as surface runoff and leaching."""
    irrg_A = I_df.resample('A', kind='period').sum()
    prcp_A = prcp_d.resample('A', kind='period').sum()
    # field scale
    irrg_TN_input_f = irrg_A * TN_I / 100  # in kg/ha/yr
    irrg_TP_input_f = irrg_A * TP_I / 100  # in kg/ha/yr
    prcp_TN_input_f = prcp_A * TN_P / 100  # in kg/ha/yr
    prcp_TP_input_f = prcp_A * TP_P / 100  # in kg/ha/yr
    # IDU scale
    irrg_TN_input_sys = irrg_A * TN_I / 100 * area_f / (area_f + area_pd)  # in kg/ha/yr
    irrg_TP_input_sys = irrg_A * TP_I / 100 * area_f / (area_f + area_pd)  # in kg/ha/yr
    prcp_TN_input_sys = prcp_A * TN_P / 100  # in kg/ha/yr
    prcp_TP_input_sys = prcp_A * TP_P / 100  # in kg/ha/yr
    # Calculate the net nutrient export from fields and from the system unit, respectively.
    TN_export_f = TN_f_A - irrg_TN_input_f - prcp_TN_input_f
    TP_export_f = TP_f_A - irrg_TP_input_f - prcp_TP_input_f
    TN_export_sys = TN_sys_A - irrg_TN_input_sys - prcp_TN_input_sys
    TP_export_sys = TP_sys_A - irrg_TP_input_sys - prcp_TP_input_sys
    # The input-output dataframes
    TN_in_out = pd.concat([pd.DataFrame({"irrigation input": irrg_TN_input_f.values.reshape(-1),
                                         "precipitation input": prcp_TN_input_f.values.reshape(-1),
                                         "runoff loss": TN_fs_A.values.reshape(-1),
                                         "leaching loss": TN_fb_A.values.reshape(-1),
                                         "net export": TN_export_f.values.reshape(-1),
                                         "scale": "field"}),
                           pd.DataFrame({"irrigation input": irrg_TN_input_sys.values.reshape(-1),
                                         "precipitation input": prcp_TN_input_sys.values.reshape(-1),
                                         "runoff loss": TN_s_A.values.reshape(-1),
                                         "leaching loss": TN_b_A.values.reshape(-1),
                                         "net export": TN_export_sys.values.reshape(-1),
                                         "scale": "IDU"})],
                          ignore_index=True)
    TP_in_out = pd.concat([pd.DataFrame({"irrigation input": irrg_TP_input_f.values.reshape(-1),
                                         "precipitation input": prcp_TP_input_f.values.reshape(-1),
                                         "runoff loss": TP_fs_A.values.reshape(-1),
                                         "leaching loss": TP_fb_A.values.reshape(-1),
                                         "net export": TP_export_f.values.reshape(-1),
                                         "scale": "field"}),
                           pd.DataFrame({"irrigation input": irrg_TP_input_sys.values.reshape(-1),
                                         "precipitation input": prcp_TP_input_sys.values.reshape(-1),
                                         "runoff loss": TP_s_A.values.reshape(-1),
                                         "leaching loss": TP_b_A.values.reshape(-1),
                                         "net export": TP_export_sys.values.reshape(-1),
                                         "scale": "IDU"})],
                          ignore_index=True)
    # Calculate the export reduction by ditches and ponds
    TN_reduction = (TN_fs_A * area_f - TN_s_A * (area_f + area_pd)) / (TN_fs_A * area_f) * 100  # in %
    TP_reduction = (TP_fs_A * area_f - TP_s_A * (area_f + area_pd)) / (TP_fs_A * area_f) * 100  # in %
    TN_reduction = TN_reduction.where(TN_reduction > 0, 0)
    TN_reduction = TN_reduction.where(TN_reduction < 100, 100)
    TP_reduction = TP_reduction.where(TP_reduction > 0, 0)
    TP_reduction = TP_reduction.where(TP_reduction < 100, 100)
    result_export_df = pd.DataFrame({'TN_reduction(%)': TN_reduction.values.reshape(-1),
                                     'TP_reduction(%)': TP_reduction.values.reshape(-1)})
    result_export = result_export_df.describe()

    # Prepare data for bmp comparison
    comp_data = {"TN_load": result_sys_pr.loc["mean", "TN_load(kg/ha/yr)"],
                 "TP_load": result_sys_pr.loc["mean", "TP_load(kg/ha/yr)"],
                 "TN_conc": round(np.nanmean(TN_cs_sys.values), 3),
                 "TP_conc": round(np.nanmean(TP_cs_sys.values), 3)}

    """ Write the results into text files """
    # Write ET_d into files
    with open(output_dir + '/ET_d.txt', mode='w') as output_f:
        output_f.write(
            'The simulated ET (mm/d) from paddy fields are below:\n')
        output_f.close()
    ET_df.to_csv(output_dir + '/ET_d.txt', mode='a', header=True, index=True, sep='\t',
                 float_format='%.4f')
    # Write output summary file
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write('The simulated nutrient loads from the fields are summarized as below:\n')
        output_f.close()
    result_f_pr.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                       float_format='%.2f')
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
        output_f.write('\nThe simulated nutrient loads from the system unit are summarized as below:\n')
        output_f.close()
    result_sys_pr.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                         float_format='%.2f')
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
        output_f.write('\nThe reduction rate by ditches and ponds are summarized as below:\n')
        output_f.close()
    result_export.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                         float_format='%.2f')

    # Write average concentrations
    with open(output_dir + '/TN_cs_sys{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write(
            'The simulated flow-weighted annual average TN concentrations (mg/L) from system unit outlet are:\n')
        output_f.close()
    TN_cs_sys.to_csv(output_dir + '/TN_cs_sys{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                     float_format='%.3f')
    with open(output_dir + '/TP_cs_sys{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write(
            'The simulated flow-weighted annual average TP concentrations (mg/L) from system unit outlet are:\n')
        output_f.close()
    TP_cs_sys.to_csv(output_dir + '/TP_cs_sys{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                     float_format='%.3f')

    # Irrigated water content
    with open(output_dir + '/I_out{}.txt'.format(run_id), 'w') as f:
        f.write("The irrigated water content is simulated as below, in mm.\n")
        f.close()
    irrg_A.to_csv(output_dir + '/I_out{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    # precipitation water content
    with open(output_dir + '/prcp_A.txt', 'w') as f:
        f.write("The precipitation water content during the growing season is simulated as below, in mm.\n")
        f.close()
    prcp_A.to_csv(output_dir + '/prcp_A.txt', mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    # Annual load from system unit
    with open(output_dir + '/TN_load_sysA{}.txt'.format(run_id), 'w') as output_f:
        output_f.write("The annual TN loads (kg/ha) from the system unit are simulated as:\n")
        output_f.close()
    TN_sys_A.to_csv(output_dir + '/TN_load_sysA{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                    float_format='%.3f')
    with open(output_dir + '/TP_load_sysA{}.txt'.format(run_id), 'w') as output_f:
        output_f.write("The annual TP loads (kg/ha) from the system unit are simulated as:\n")
        output_f.close()
    TP_sys_A.to_csv(output_dir + '/TP_load_sysA{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                    float_format='%.3f')
    # TN reduction percentages
    TN_reduction.to_csv(output_dir + '/TN_reduction_rate{}.txt'.format(run_id), header=True, index=True, sep='\t',
                        float_format='%.2f')
    # TP reduction percentages
    TP_reduction.to_csv(output_dir + '/TP_reduction_rate{}.txt'.format(run_id), header=True, index=True, sep='\t',
                        float_format='%.2f')
    # Nutrient input output results.
    with open(output_dir + '/TN_in_out{}.txt'.format(run_id), mode='w') as f:
        f.write("The TN flux to and from the field V.S. IDU scale is as below, in kg/ha. \n")
        f.close()
    TN_in_out.to_csv(output_dir + '/TN_in_out{}.txt'.format(run_id), mode='a', header=True, index=False, sep='\t',
                     float_format='%.4f')
    with open(output_dir + '/TP_in_out{}.txt'.format(run_id), mode='w') as f:
        f.write("The TP flux to and from the field V.S. IDU scale is as below, in kg/ha. \n")
        f.close()
    TP_in_out.to_csv(output_dir + '/TP_in_out{}.txt'.format(run_id), mode='a', header=True, index=False, sep='\t',
                     float_format='%.4f')
    with open(output_dir + "/pre-risk_log{}.txt".format(run_id), mode='w') as f:
        f.write("The pre-risk drainage management was simulated to be conducted in dates below:\n")
        f.write(pre_risk_log.get_string())
        f.close()

    """ Prepare results summary for GUI """
    output_text = f"""||模拟结果小结||
                    \n计算时期内，水稻季的年平均模拟结果如下：
                    \n（1）田块尺度\n"""
    output_text += "TN年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_f_pr.loc["mean", "TN_load(kg/ha/yr)"], result_f_pr.loc["mean", "TN_surface(rate)"] * 100,
                                                      result_f_pr.loc["mean", "TN_subsurface(rate)"] * 100)
    output_text += "TP年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_f_pr.loc["mean", "TP_load(kg/ha/yr)"], result_f_pr.loc["mean", "TP_surface(rate)"] * 100,
                                                      result_f_pr.loc["mean", "TP_subsurface(rate)"] * 100)
    output_text += "地表径流中TN年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TN_cs_f.values))
    output_text += "地表径流中TP年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TP_cs_f.values))
    output_text += "\n（2）灌排单元尺度\n"
    output_text += "TN年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_sys_pr.loc["mean", "TN_load(kg/ha/yr)"], result_sys_pr.loc["mean", "TN_surface(rate)"] * 100,
                                                        result_sys_pr.loc["mean", "TN_subsurface(rate)"] * 100)
    output_text += "TP年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_sys_pr.loc["mean", "TP_load(kg/ha/yr)"], result_sys_pr.loc["mean", "TP_surface(rate)"] * 100,
                                                        result_sys_pr.loc["mean", "TP_subsurface(rate)"] * 100)
    output_text += "地表径流中TN年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TN_cs_sys.values))
    output_text += "地表径流中TP年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TP_cs_sys.values))

    if graph == 1:
        """ Convert outputs to time series """
        yearStart = prcp_d.index[0].year
        yearEnd = prcp_d.index[-1].year + 1
        Hdp_df = Hdp_df - HDiff_FD1
        for year in range(yearStart, yearEnd):
            Hf_df1[year] = Hf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            Hdp_df1[year] = Hdp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_cf_df1[year] = TP_cf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_cd1_df1[year] = TP_cd1_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_cf_df1[year] = TN_cf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_cd1_df1[year] = TN_cd1_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            if havePond:
                TP_cp_df1[year] = TP_cp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
                TN_cp_df1[year] = TN_cp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            if orderDitch == 2:
                TP_cd2_df1[year] = TP_cd2_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
                TN_cd2_df1[year] = TN_cd2_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_fs_df1[year] = TN_fs_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_s_df1[year] = TN_s_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_fs_df1[year] = TP_fs_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_s_df1[year] = TP_s_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values

        """Write Time series outputs """
        # water level in fields
        Hf_df1.to_csv(output_dir + '/H_f_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # water level in ditches and pond
        Hdp_df1.to_csv(output_dir + '/H_dp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN concentration in fields
        TN_cf_df1.to_csv(output_dir + '/TN_cf_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP concentration in fields
        TP_cf_df1.to_csv(output_dir + '/TP_cf_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN concentration in first-order ditch
        TN_cd1_df1.to_csv(output_dir + '/TN_cd1_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP concentration in first-order ditch
        TP_cd1_df1.to_csv(output_dir + '/TP_cd1_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        if orderDitch == 2:
            # TN concentration in second-order ditch
            TN_cd2_df1.to_csv(output_dir + '/TN_cd2_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
            # TP concentration in second-order ditch
            TP_cd2_df1.to_csv(output_dir + '/TP_cd2_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        if havePond:
            # TN concentration in pond
            TN_cp_df1.to_csv(output_dir + '/TN_cp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
            # TP concentration in pond
            TP_cp_df1.to_csv(output_dir + '/TP_cp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN load from field with surface runoff
        TN_fs_df1.to_csv(output_dir + '/TN_load_fs_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN load from system with surface runoff
        TN_s_df1.to_csv(output_dir + '/TN_load_syss_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP load from field with surface runoff
        TP_fs_df1.to_csv(output_dir + '/TP_load_fs_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP load from system with surface runoff
        TP_s_df1.to_csv(output_dir + '/TP_load_syss_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')

        """ Prepare dataframe list for drawing graphs """
        if orderDitch == 1:
            TN_cd2_df1 = 0
            TP_cd2_df1 = 0
        if not havePond:
            TN_cp_df1 = 0
            TP_cp_df1 = 0
        graph_dfs = list([Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1,
                          TN_cp_df1, TP_cp_df1, TN_fs_df1, TN_s_df1, TP_fs_df1, TP_s_df1, TN_in_out, TP_in_out])

    if graph == 1:
        return (output_text, comp_data, graph_dfs)
    else:
        return (output_text, comp_data)


def both_bmps(prcp_d, startDay, endDay, wtr_mng_d, pet_d, ditch_par, pond_par, F_doy_array, fert_type_array,
             TN0_array, TP0_array, TN_I, TP_I, area_f, Ks, Fc0_f, Fc0_pd, TN_P0, TN_P1, TP_P, TN_rate_lch, TN_rate_lch_lat,
             TP_rate_lch, TP_rate_lch_lat, ENC0_dp, EPC0_dp, ENCR_f, EPCR_f, vf_N_array, vf_P, h_v_sr, vf_N_pd, vf_P_pd,
             risk_days, min_level_prerisk, min_level_recycle, Hf_Hdp_sr, run_id, output_dir=0, graph=0):
    """ The pre-risk mode, for only accurate water level management.
        Parameters:
            prcp_d, pandas DataFrame stores daily precipitation data, in mm;
        Outputs:
            Hf"""
    pre_risk_log = PrettyTable(["日期", "预估农田排水量(mm)", "排水前沟塘水位(mm)", "排水后沟塘水位(mm)", "排水量(m3)",
                                "排水平均TN浓度(mg/L)", "排水平均TP浓度(mg/L)"])
    pre_risk_log.align[""] = "l"

    Hf_df, Hdp_df, D_df, F_lat_df, Fc_f_df, Fc_pd_df, Qdp_df, I_df, TN_cf_df, TP_cf_df, TN_cd1_df, TP_cd1_df, \
    TN_cd2_df, TP_cd2_df, TN_cp_df, TP_cp_df, TN_cfb_df, TP_cfb_df, TN_cfb_lat_df, TP_cfb_lat_df, TN_b_df, TP_b_df, \
    TN_out_df, TP_out_df, Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1, \
    TN_cp_df1, TP_cp_df1, TN_fs_df1, TP_fs_df1, TN_s_df1, TP_s_df1 = create_common_df(
        prcp_d, startDay, endDay, wtr_mng_d)
    TN_b_df_4wf = TN_b_df.copy(deep=True)
    ET_df = Hf_df.copy(deep=True)
    I_pond_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                         dtype=np.float32)  # The amount of irrigated water from ponds, in mm/d
    TN_I_out_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                               dtype=np.float32)  # TN load from irrigated water from outside of the system, in mg/m2/d
    TP_I_out_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                               dtype=np.float32)  # TP load from irrigated water from outside of the system, in mg/m2/d
    TN_I_pond_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                                dtype=np.float32)  # TN load from irrigated water from pond, in mg/m2/d
    TP_I_pond_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                                dtype=np.float32)  # TN load from irrigated water from pond, in mg/m2/d

    orderDitch = ditch_par[0]
    if orderDitch == 1:
        Wf1 = ditch_par[1][0]
        HDiff_FD1 = ditch_par[1][1]
        Wd1_up = ditch_par[1][2]
        Wd1_lw = ditch_par[1][3]
        n1 = ditch_par[1][4]
        slp_d1 = ditch_par[1][5]
        n_d1 = ditch_par[1][6]
        kc_d1 = ditch_par[1][7]
        TN_cd10 = ditch_par[1][8]
        TP_cd10 = ditch_par[1][9]
        Hd10 = ditch_par[1][10]
        area_d1 = ditch_par[1][11]
        area_d2 = 0
        Wf2 = area_f / n1 / Wf1
    else:
        Wf1 = ditch_par[1][0]
        HDiff_FD1 = ditch_par[1][1]
        Wd1_up = ditch_par[1][2]
        Wd1_lw = ditch_par[1][3]
        n1 = ditch_par[1][4]
        slp_d1 = ditch_par[1][5]
        n_d1 = ditch_par[1][6]
        kc_d1 = ditch_par[1][7]
        TN_cd10 = ditch_par[1][8]
        TP_cd10 = ditch_par[1][9]
        Hd10 = ditch_par[1][10]
        area_d1 = ditch_par[1][11]
        Wf2 = ditch_par[2][0]
        HDiff_FD2 = ditch_par[2][1]
        Wd2_up = ditch_par[2][2]
        Wd2_lw = ditch_par[2][3]
        n2 = ditch_par[2][4]
        slp_d2 = ditch_par[2][5]
        n_d2 = ditch_par[2][6]
        kc_d2 = ditch_par[2][7]
        TN_cd20 = ditch_par[2][8]
        TP_cd20 = ditch_par[2][9]
        area_d2 = ditch_par[2][10]
    havePond = pond_par[0]
    if havePond:
        HDiff_FP = pond_par[1][1]
        slp_p = pond_par[1][2]
        kc_p = pond_par[1][3]
        TN_cp0 = pond_par[1][4]
        TP_cp0 = pond_par[1][5]
        area_p = pond_par[1][6]
    else:
        area_p = 0.0

    # The rate of area of ditches and pond to the total area, in %
    area_pd = area_d1 + area_d2 + area_p

    startday_doy = pd.Timestamp(2018, int(startDay.split('-')[0]), int(startDay.split('-')[1])).dayofyear
    endday_doy = pd.Timestamp(2018, int(endDay.split('-')[0]), int(endDay.split('-')[1])).dayofyear
    for column in prcp_d.columns:
        date_n = 0
        irrg_flag = 0
        while date_n < len(prcp_d.index):
            dayofYear = prcp_d.index[date_n].dayofyear
            monthDay = str(prcp_d.index[date_n].month) + "-" + str(prcp_d.index[date_n].day)
            if calendar.isleap(prcp_d.index[date_n].year):
                dayofYear = dayofYear - 1
            if dayofYear == startday_doy:
                water_adj = 0
                Hf = 0.0
                Hdp = Hd10
                Hd1 = Hd10
                TP_cd1 = TP_cd10
                TN_cd1 = TN_cd10
                if orderDitch == 2:
                    Hd2 = Hd1 + HDiff_FD2 - HDiff_FD1
                    TP_cd2 = TP_cd20
                    TN_cd2 = TN_cd20
                if havePond:
                    Hp = Hd1 + HDiff_FP - HDiff_FD1
                    TP_cp = TP_cp0
                    TN_cp = TN_cp0
                irrg_date = wtr_mng_d["H_min"].ne(0).idxmax()
                irrg_doy = pd.Timestamp(2018, int(irrg_date.split('-')[0]), int(irrg_date.split('-')[1])).dayofyear
                if F_doy_array[0] <= irrg_doy:  # 施肥日在灌溉日之前，田面水浓度初始值为施肥初始浓度
                    TP_cf = TP0_array[0]
                    TN_cf = TN0_array[0]
                else:  # 施肥日在灌溉日之后，田面水浓度初始值为灌溉水浓度
                    TP_cf = TP_I
                    TN_cf = TN_I
                fert_lag = 0

            if (dayofYear < startday_doy) | (dayofYear > endday_doy):
                prcp_d.iloc[date_n] = 0.0  # Change the precipitation in non-growing period to 0, for statistics
            else:
                """ Water balance in fields """
                # Get the precipitation and PET data of the simulation day
                prcp = prcp_d[column].iloc[date_n]
                PET = pet_d[column].iloc[date_n]
                # Calculate ET
                if Hf > 0:
                    ET = PET * wtr_mng_d.loc[monthDay, 'Kc']
                else:
                    ET = PET * 0.60
                ET_df[column].iloc[date_n] = ET
                # Do irrigation or not
                if (wtr_mng_d.loc[monthDay, 'H_min'] != 0) & (Hf < wtr_mng_d.loc[monthDay, 'H_min']) & (
                        prcp + Hf - ET < wtr_mng_d.loc[monthDay, 'H_min']):
                    irrg = wtr_mng_d.loc[monthDay, 'H_max'] - Hf  # in mm
                    irrg_v = irrg * area_f  # in mm.m2
                    # 分配来自沟塘和外部水源的灌溉量
                    if havePond is True:
                        irrg_pond_max = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - sd_functions.f_vhp(
                            min_level_recycle, HDiff_FP, area_p, slp_p)
                    elif orderDitch == 2:
                        irrg_pond_max = (sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) - sd_functions.f_vhd(
                            min_level_recycle, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) * n1 * n2
                    else:
                        irrg_pond_max = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) - sd_functions.f_vhd(
                            min_level_recycle, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) * n1
                    if irrg_pond_max > irrg_v:
                        irrg_out = 0
                        irrg_pond = irrg  # in mm
                    else:
                        irrg_pond = max(0, irrg_pond_max / area_f - 0.1)  # in mm
                        irrg_out = irrg - irrg_pond  # in mm
                    # 更新灌溉后沟塘水位和水质
                    if irrg_pond > 0:
                        Hdp1 = sd_functions.f_vhpd(Hdp, - irrg_pond * area_f, h_v_sr, area_pd, HDiff_FD1)[1]  # in mm
                        if (havePond is True) & (orderDitch == 2):
                            vd1 = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                   sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) * n1 * n2
                            vd2 = (sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) -
                                   sd_functions.f_vhd(Hdp1 + HDiff_FD2 - HDiff_FD1, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)
                                   ) * n1 * n2
                            vp = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - irrg_pond * area_f
                            TN_cp1 = (TN_cd1 * vd1 + TN_cd2 * vd2 + TN_cp * vp) / (vd1 + vd2 + vp)
                            TP_cp1 = (TP_cd1 * vd1 + TP_cd2 * vd2 + TP_cp * vp) / (vd1 + vd2 + vp)
                            Hd1 = Hdp1
                            Hd2 = Hd1 + HDiff_FD2 - HDiff_FD1
                            Hp = Hd1 + HDiff_FP - HDiff_FD1
                        elif (havePond is True) & (orderDitch == 1):
                            vd1 = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                   sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) * n1
                            vp = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - irrg_pond * area_f
                            TN_cp1 = (TN_cd1 * vd1 + TN_cp * vp) / (vd1 + vp)
                            TP_cp1 = (TP_cd1 * vd1 + TP_cp * vp) / (vd1 + vp)
                            Hd1 = Hdp1
                            Hp = Hd1 + HDiff_FP - HDiff_FD1
                        elif (havePond is False) & (orderDitch == 2):
                            vd1 = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                   sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) * n1 * n2
                            vd2 = sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) * n1 * n2 - irrg_pond * area_f
                            TN_cd21 = (TN_cd1 * vd1 + TN_cd2 * vd2) / (vd1 + vd2)
                            TP_cd21 = (TP_cd1 * vd1 + TP_cd2 * vd2) / (vd1 + vd2)
                            Hd1 = Hdp1
                            Hd2 = Hd1 + HDiff_FD2 - HDiff_FD1
                        elif (havePond is False) & (orderDitch == 1):
                            Hd1 = Hdp1
                        Hdp = Hdp1
                        del Hdp1
                else:
                    irrg = 0
                    irrg_out = 0
                    irrg_pond = 0
                if havePond is True:
                    TN_I_pond_df[column].iloc[date_n] = TN_cp * irrg_pond  # in mg/m2
                    TP_I_pond_df[column].iloc[date_n] = TP_cp * irrg_pond  # in mg/m2
                elif orderDitch == 2:
                    TN_I_pond_df[column].iloc[date_n] = TN_cd2 * irrg_pond  # in mg/m2
                    TP_I_pond_df[column].iloc[date_n] = TP_cd2 * irrg_pond  # in mg/m2
                else:
                    TN_I_pond_df[column].iloc[date_n] = TN_cd1 * irrg_pond  # in mg/m2
                    TP_I_pond_df[column].iloc[date_n] = TP_cd1 * irrg_pond  # in mg/m2
                TN_I_out_df[column].iloc[date_n] = TN_I * irrg_out  # in mg/m2
                TP_I_out_df[column].iloc[date_n] = TP_I * irrg_out  # in mg/m2
                I_df[column].iloc[date_n] = irrg  # in mm
                I_pond_df[column].iloc[date_n] = irrg_pond  # in mm
                # Calculate lateral seepage to ditches
                if orderDitch == 2:
                    F_lat_d1 = max(0, Ks * (Hf - Hd1 + HDiff_FD1) / (0.25 * Wf1 * 1000) * Wf2 * 2 * (
                            Hf + HDiff_FD1))  # in m2.mm/d
                    F_lat_d2 = max(0, Ks * (Hf - Hd2 + HDiff_FD2) / (0.25 * Wf2 * 1000) * Wf1 * 2 * (
                            Hf + HDiff_FD2))  # in m2.mm/d
                    F_lat = (F_lat_d1 + F_lat_d2) / (Wf1 * Wf2)  # in mm/d
                elif orderDitch == 1:
                    F_lat_d1 = max(0, Ks * (Hf - Hd1 + HDiff_FD1) / (0.25 * Wf1 * 1000) * Wf2 * 2 * (
                            Hf + HDiff_FD1))
                    F_lat = F_lat_d1 / (Wf1 * Wf2)  # in mm/d
                F_lat_df[column].iloc[date_n] = F_lat
                # Calculate downward percolation
                if (Hf - ET) > Fc0_f:
                    Fc_f = Fc0_f
                else:
                    Fc_f = 0
                Fc_f_df[column].iloc[date_n] = Fc_f
                # Update water level in fields, Hf, first-round
                Hf1 = Hf + prcp + irrg - ET - Fc_f - F_lat
                # Do drainage or not
                if Hf1 > wtr_mng_d.loc[monthDay, 'H_p']:
                    D = Hf1 - wtr_mng_d.loc[monthDay, 'H_p']  # in mm/d
                    Hf1 = Hf1 - D  # Update again, second-round
                else:
                    D = 0
                if D > 0:
                    water_adj = 1
                D_df[column].iloc[date_n] = D

                """ Water quality in fields"""
                # Initial TN and TP concentrations for the fertilization days, and
                # adjust TP and TN with irrigation, precipitation and ETs.
                flag_N = 0
                flag_P = 0
                TN_P = TN_P1
                for index, F_doy in enumerate(F_doy_array[TN0_array > 0]):
                    if (dayofYear >= F_doy) & (dayofYear - 15 <= F_doy):
                        TN_P = TN_P0
                for index, item in enumerate(F_doy_array):
                    if (dayofYear - fert_lag == item) & (D == 0):
                        fert_lag = 0
                        if TN0_array[index] != 0:
                            TN_cf = TN0_array[index]
                            flag_N = 1
                        if TP0_array[index] != 0:
                            TP_cf = TP0_array[index]
                            flag_P = 1
                    elif (dayofYear - fert_lag == item) & (D > 0):
                        if fert_lag < 5:
                            fert_lag += 1
                        else:
                            fert_lag = 0
                            if TN0_array[index] != 0:
                                TN_cf = TN0_array[index]
                                flag_N = 1
                            if TP0_array[index] != 0:
                                TP_cf = TP0_array[index]
                                flag_P = 1
                # Mixed concentration based on material balance
                if (Hf1 >= 10) & (Hf > 10):
                    if flag_N == 0:
                        TN_cf = (TN_cf * Hf + TN_I * irrg_out + TN_P * prcp +
                                 TN_I_pond_df[column].iloc[date_n]) / (Hf + irrg + prcp - ET)
                    if flag_P == 0:
                        TP_cf = (TP_cf * Hf + TP_I * irrg_out + TP_P * prcp +
                                 TP_I_pond_df[column].iloc[date_n]) / (Hf + irrg + prcp - ET)
                elif (Hf1 >= 10) & (Hf <= 10):
                    temp = max(min(10 + Hf, 10.0), 1)
                    if flag_N == 0:
                        TN_cf = (TN_cf * 2.0 * temp + TN_I * irrg_out + TN_P * prcp +
                                 TN_I_pond_df[column].iloc[date_n]) / (temp + irrg + prcp - ET)
                    if flag_P == 0:
                        TP_cf = (TP_cf * 4.0 * temp + TP_I * irrg_out + TP_P * prcp +
                                 TP_I_pond_df[column].iloc[date_n]) / (temp + irrg + prcp - ET)
                    del temp
                if irrg_pond > 0:
                    if havePond is True:
                        TN_cp = TN_cp1
                        TP_cp = TP_cp1
                        del TN_cp1
                        del TP_cp1
                    elif orderDitch == 2:
                        TN_cd2 = TN_cd21
                        TP_cd2 = TP_cd21
                        del TN_cd21
                        del TP_cd21
                # Adjust TN concentration for 2-7 days after tillering/filling fertilization
                for x, doy in enumerate(F_doy_array[fert_type_array == 2]):
                    if (dayofYear - 2 >= doy) & (dayofYear - 3 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.3
                    elif (dayofYear - 4 >= doy) & (dayofYear - 5 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.2
                    elif (dayofYear - 6 >= doy) & (dayofYear - 7 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.1
                    else:
                        TN_cf = TN_cf

                # Record percolated concentrations
                TN_cfb = TN_cf * TN_rate_lch
                TN_cfb_lat = max(TN_cf * TN_rate_lch_lat, ENC0_dp)
                TP_cfb = TP_cf * TP_rate_lch
                TP_cfb_lat = max(TP_cf * TP_rate_lch_lat, EPC0_dp)
                TP_cfb_df[column].iloc[date_n] = TP_cfb
                TN_cfb_df[column].iloc[date_n] = TN_cfb
                TP_cfb_lat_df[column].iloc[date_n] = TP_cfb_lat
                TN_cfb_lat_df[column].iloc[date_n] = TN_cfb_lat
                # Account for areal dynamics in the water-soil interface
                flag_N = 0
                flag_P = 0
                for index, F_doy in enumerate(F_doy_array[TN0_array > 0]):
                    if index + 1 < np.count_nonzero(TN0_array):
                        if (dayofYear > F_doy) & (dayofYear < F_doy_array[TN0_array > 0][index + 1]) & (Hf > 0) & (
                                Hf1 > 0) & (TN_cf > (ENCR_f * TN0_array[index])) & (flag_N == 0):
                            TN_cf1 = max(TN_cf * math.exp(- vf_N_array[index] * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         TN0_array[index] * ENCR_f)
                            flag_N = 1
                    else:
                        if (dayofYear > F_doy) & (Hf > 0) & (Hf1 > 0) & (TN_cf > (ENCR_f * TN0_array[index])) & (
                                flag_N == 0):
                            TN_cf1 = max(TN_cf * math.exp(- vf_N_array[index] * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         TN0_array[index] * ENCR_f)
                            flag_N = 1
                for index, F_doy in enumerate(F_doy_array[TP0_array > 0]):
                    if index + 1 < np.count_nonzero(TP0_array):
                        if (dayofYear > F_doy) & (dayofYear < F_doy_array[TP0_array > 0][index + 1]) & (Hf > 0) & (
                                Hf1 > 0) & (TP_cf > (EPCR_f * TP0_array[index])) & (flag_P == 0):
                            TP_cf1 = max(TP_cf * math.exp(- vf_P * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         EPCR_f * TP0_array[index])
                            flag_P = 1
                    else:
                        if (dayofYear > F_doy) & (Hf > 0) & (Hf1 > 0) & (TP_cf > (EPCR_f * TP0_array[index])) & (
                                flag_P == 0):
                            TP_cf1 = max(TP_cf * math.exp(- vf_P * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         EPCR_f * TP0_array[index])
                            flag_P = 1
                if flag_P == 0:
                    TP_cf1 = TP_cf
                if flag_N == 0:
                    TN_cf1 = TN_cf
                # Record water table and water quality in fields
                TP_cf_df[column].iloc[date_n] = (TP_cf + TP_cf1) / 2
                TN_cf_df[column].iloc[date_n] = (TN_cf + TN_cf1) / 2
                if Hf1 >= 0:
                    Hf_df[column].iloc[date_n] = Hf1
                else:
                    Hf_df[column].iloc[date_n] = 0.0
                Hf = Hf1

                """ Water balance in ditches and the pond """
                # Decide whether water level adjustment is needed
                if water_adj == 1:
                    for F_doy in F_doy_array:
                        if dayofYear == F_doy:
                            water_adj = 0
                if water_adj == 0:
                    Hf_adj = 0
                    for F_doy in F_doy_array:
                        if (dayofYear >= F_doy) & (dayofYear - risk_days <= F_doy):
                            # 估算农田预计排水量，mm
                            iloc_day = wtr_mng_d.index.get_loc(monthDay)
                            H_p_next = min(wtr_mng_d['H_p'].iloc[iloc_day + 1], wtr_mng_d['H_p'].iloc[iloc_day + 2],
                                           wtr_mng_d['H_p'].iloc[iloc_day + 3])
                            # Adjust water depth based on weather forecast and water requirement of fields
                            prcp_next = prcp_d[column].iloc[date_n + 1]
                            if prcp_next > 100:
                                prcp_next1 = 100
                            elif prcp_next > 50:
                                prcp_next1 = 70
                            elif prcp_next > 25:
                                prcp_next1 = 35
                            else:
                                prcp_next1 = 0
                            if (prcp_next1 > 0) & (Hf + prcp_next1 - H_p_next > 10):  # 降雨前排水
                                Hf_adj = math.ceil((Hf + prcp_next1 - H_p_next - 10) / 5) * 5
                                Hf_adj1 = min(Hf_adj, Hf_Hdp_sr.index.max())
                            elif (prcp_next1 == 0) & (Hf - wtr_mng_d['H_p'].iloc[iloc_day + 1] > 10):  # 提前一天主动排水
                                Hf_adj = math.ceil((Hf - wtr_mng_d['H_p'].iloc[iloc_day + 1] - 10) / 5) * 5
                                Hf_adj1 = min(Hf_adj, Hf_Hdp_sr.index.max())
                    # Do water level adjustment
                    if havePond is True:
                        if (Hf_adj > 0) & (Hdp + HDiff_FP - HDiff_FD1 > min_level_prerisk):
                            Hdp_adj = max(Hf_Hdp_sr[Hf_adj1], min_level_prerisk - HDiff_FP + HDiff_FD1)
                            if Hdp_adj < Hdp:
                                water_adj_potential = 1  # 是否实施提前排水，是1， 否0
                                #  计算提前排水量
                                Q_dp_adj = (h_v_sr[Hdp] - h_v_sr[Hdp_adj]) / 1000  # in m3/d
                                if orderDitch == 2:
                                    # 计算提前排水量的平均水质浓度，更新沟塘水位
                                    vd1_prerisk = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                                   sd_functions.f_vhd(Hdp_adj, HDiff_FD1, Wd1_up, Wd1_lw,
                                                                      Wf2)) * n1 * n2
                                    vd2_prerisk = (sd_functions.f_vhd(Hd2, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                                   sd_functions.f_vhd(Hdp_adj + HDiff_FD2 - HDiff_FD1,
                                                                      HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) * n1 * n2
                                    vp_prerisk = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - \
                                                 sd_functions.f_vhp(Hdp_adj + HDiff_FP - HDiff_FD1, HDiff_FP,
                                                                    area_p, slp_p)
                                    TN_prerisk = (TN_cd1 * vd1_prerisk + TN_cd2 * vd2_prerisk + TN_cp * vp_prerisk) / (
                                            vd1_prerisk + vd2_prerisk + vp_prerisk)
                                    TP_prerisk = (TP_cd1 * vd1_prerisk + TP_cd2 * vd2_prerisk + TP_cp * vp_prerisk) / (
                                            vd1_prerisk + vd2_prerisk + vp_prerisk)
                                    Hd1 = Hdp_adj
                                    Hd2 = Hdp_adj + HDiff_FD2 - HDiff_FD1
                                    Hp = Hdp_adj + HDiff_FP - HDiff_FD1
                                else:
                                    vd1_prerisk = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                                   sd_functions.f_vhd(Hdp_adj, HDiff_FD1, Wd1_up, Wd1_lw,
                                                                      Wf2)) * n1
                                    vp_prerisk = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - \
                                                 sd_functions.f_vhp(Hdp_adj + HDiff_FP - HDiff_FD1, HDiff_FP,
                                                                    area_p, slp_p)
                                    TN_prerisk = (TN_cd1 * vd1_prerisk + TN_cp * vp_prerisk) / (
                                                vd1_prerisk + vp_prerisk)
                                    TP_prerisk = (TP_cd1 * vd1_prerisk + TP_cp * vp_prerisk) / (
                                                vd1_prerisk + vp_prerisk)
                                    Hd1 = Hdp_adj
                                    Hp = Hdp_adj + HDiff_FP - HDiff_FD1
                            else:
                                water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                        else:
                            water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                    elif orderDitch == 2:
                        if (Hf_adj > 0) & (Hdp + HDiff_FD2 - HDiff_FD1 > min_level_prerisk):
                            Hdp_adj = max(Hf_Hdp_sr[Hf_adj1], min_level_prerisk - HDiff_FD2 + HDiff_FD1)
                            if Hdp_adj < Hdp:
                                water_adj_potential = 1  # 是否实施提前排水，是1， 否0
                                #  计算提前排水量
                                Q_dp_adj = (h_v_sr[Hdp] - h_v_sr[Hdp_adj]) / 1000  # in m3/d
                                # 计算提前排水量的平均水质浓度，更新沟塘水位
                                vd1_prerisk = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                               sd_functions.f_vhd(Hdp_adj, HDiff_FD1, Wd1_up, Wd1_lw,
                                                                  Wf2)) * n1 * n2
                                vd2_prerisk = (sd_functions.f_vhd(Hd2, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                               sd_functions.f_vhd(Hdp_adj + HDiff_FD2 - HDiff_FD1,
                                                                  HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) * n1 * n2
                                TN_prerisk = (TN_cd1 * vd1_prerisk + TN_cd2 * vd2_prerisk) / (
                                        vd1_prerisk + vd2_prerisk)
                                TP_prerisk = (TP_cd1 * vd1_prerisk + TP_cd2 * vd2_prerisk) / (
                                        vd1_prerisk + vd2_prerisk)
                                Hd1 = Hdp_adj
                                Hd2 = Hdp_adj + HDiff_FD2 - HDiff_FD1
                            else:
                                water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                        else:
                            water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                    elif orderDitch == 1:
                        if (Hf_adj > 0) & (Hdp > min_level_prerisk):
                            Hdp_adj = max(Hf_Hdp_sr[Hf_adj1], min_level_prerisk)
                            if Hdp_adj < Hdp:
                                water_adj_potential = 1  # 是否实施提前排水，是1， 否0
                                # 计算提前排水量
                                Q_dp_adj = (h_v_sr[Hdp] - h_v_sr[Hdp_adj]) / 1000  # in m3/d
                                # 计算提前排水量的平均水质浓度，更新沟塘水位
                                TN_prerisk = TN_cd1
                                TP_prerisk = TP_cd1
                                Hd1 = Hdp_adj
                            else:
                                water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                        else:
                            water_adj_potential = 0  # 是否实施提前排水，是1， 否0
                    if water_adj_potential == 1:
                        pre_risk_log.add_row(
                            [prcp_d.index[date_n], Hf_adj, round(Hdp), round(Hdp_adj), round(Q_dp_adj),
                             round(TN_prerisk, 2), round(TP_prerisk, 2)])
                        Hdp = Hdp_adj
                        del Hf_adj
                        del Hf_adj1
                        del Hdp_adj
                else:
                    water_adj_potential = 0

                # Calculate ET and Fc from ditches and the pond
                if Hd1 > 0:
                    Wd1_d = (Wd1_up - Wd1_lw) * Hd1 / HDiff_FD1 + Wd1_lw
                    if orderDitch == 2:
                        area_d1_d = Wd1_d * Wf2 * n1 * n2
                    elif orderDitch == 1:
                        area_d1_d = Wd1_d * Wf2 * n1
                    if Hd1 - Fc0_pd > (kc_d1 * PET):
                        Fc_d1_v = Fc0_pd * area_d1_d
                    else:
                        Fc_d1_v = 0.0
                else:
                    area_d1_d = 0
                    Fc_d1_v = 0.0

                if orderDitch == 2:
                    if Hd2 > 0:
                        Wd2_d = (Wd2_up - Wd2_lw) * Hd2 / HDiff_FD2 + Wd2_lw
                        area_d2_d = Wd2_d * Wf1 * n1 * n2
                        if Hd2 - Fc0_pd > (kc_d2 * PET):
                            Fc_d2_v = Fc0_pd * area_d2_d
                        else:
                            Fc_d2_v = 0.0
                    else:
                        area_d2_d = 0
                        Fc_d2_v = 0.0

                if havePond == True:
                    if Hp > 0:
                        area_p_lw = (math.sqrt(area_p) - HDiff_FP / 1000 * slp_p * 2) ** 2  # in m2
                        hp1 = math.sqrt(area_p_lw) / 2 / slp_p  # in m
                        area_p_d = area_p_lw * ((hp1 + Hp / 1000) ** 2) / (hp1 ** 2)  # in m2
                        if Hp - Fc0_pd > (kc_p * PET):
                            Fc_p_v = Fc0_pd * area_p_d
                        else:
                            Fc_p_v = 0.0
                    else:
                        area_p_d = 0
                        Fc_p_v = 0.0
                # ET, ET_pd;, downward percoloation, Fc_pd
                if (havePond is True) & (orderDitch == 2):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_d2 * PET * area_d2_d + kc_p * PET * area_p_d + \
                              0.60 * PET * (area_pd - area_d1_d - area_d2_d - area_p_d)
                    Fc_pd_v = Fc_d1_v + Fc_d2_v + Fc_p_v
                elif (havePond is True) & (orderDitch == 1):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_p * PET * area_p_d + 0.60 * PET * (
                            area_pd - area_d1_d - area_p_d)
                    Fc_pd_v = Fc_d1_v + Fc_p_v
                elif (havePond is False) & (orderDitch == 2):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_d2 * PET * area_d2_d + 0.60 * PET * (
                            area_pd - area_d1_d - area_d2_d)
                    Fc_pd_v = Fc_d1_v + Fc_d2_v
                elif (havePond is False) & (orderDitch == 1):
                    ET_pd_v = kc_d1 * PET * area_d1_d + 0.60 * PET * (
                            area_pd - area_d1_d)
                    Fc_pd_v = Fc_d1_v

                # Update water level in ditches and ponds, Hdp; and calculate outflow Q_dp of the system from the pond
                volume_add = prcp * area_pd - Fc_pd_v - ET_pd_v + (F_lat + D) * area_f
                Q_dp = sd_functions.f_vhpd(Hdp, volume_add, h_v_sr, area_pd, HDiff_FD1)[0]  # in m3/d
                Hdp1 = sd_functions.f_vhpd(Hdp, volume_add, h_v_sr, area_pd, HDiff_FD1)[1]  # in mm
                if water_adj_potential == 1:
                    Qdp_df[column].iloc[date_n] = (Q_dp + Q_dp_adj) / (area_f + area_pd) * 1000  # in mm/d
                else:
                    Qdp_df[column].iloc[date_n] = Q_dp / (area_f + area_pd) * 1000  # in mm/d
                Hdp_df[column].iloc[date_n] = Hdp1

                """ Water quality in ditches """
                # 1st order ditch
                # Record percolation concentration
                TN_cd1b = TN_cd1 * TN_rate_lch
                TP_cd1b = TP_cd1 * TP_rate_lch
                # Calculate mixed concentration
                if orderDitch == 2:
                    TN_cd1_out = (TN_cf * D * Wf1 * Wf2 + TN_cfb_lat * F_lat_d1 + TN_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TN_P * prcp * area_d1 / n1 / n2
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1 / n2)
                    TP_cd1_out = (TP_cf * D * Wf1 * Wf2 + TP_cfb_lat * F_lat_d1 + TP_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TP_P * prcp * area_d1 / n1 / n2
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1 / n2)
                elif orderDitch == 1:
                    TN_cd1_out = (TN_cf * D * Wf1 * Wf2 + TN_cfb_lat * F_lat_d1 + TN_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TN_P * prcp * area_d1 / n1
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1)
                    TP_cd1_out = (TP_cf * D * Wf1 * Wf2 + TP_cfb_lat * F_lat_d1 + TP_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TP_P * prcp * area_d1 / n1
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1)
                if TN_cd1_out <= 0:
                    TN_cd1_out = TN_cd1
                if TP_cd1_out <= 0:
                    TP_cd1_out = TP_cd1
                # Calculate retained concentration
                if Hdp1 > 0:
                    if TN_cd1 > ENC0_dp:
                        TN_cd1 = max(TN_cd1_out * math.exp(- vf_N_pd * 1 / max(0.05, Hdp1 / 1000)), ENC0_dp)
                    else:
                        TN_cd1 = TN_cd1_out
                    if TP_cd1_out > EPC0_dp:
                        TP_cd1 = max(TP_cd1_out * math.exp(- vf_P_pd * 1 / max(0.05, Hdp1 / 1000)), EPC0_dp)
                    else:
                        TP_cd1 = TP_cd1_out
                else:
                    TN_cd1 = TN_cd1_out
                    TP_cd1 = TP_cd1_out
                # Record concentrations
                TN_cd1_df[column].iloc[date_n] = (TN_cd1_out + TN_cd1) / 2
                TP_cd1_df[column].iloc[date_n] = (TP_cd1_out + TP_cd1) / 2
                TN_cf = TN_cf1
                TP_cf = TP_cf1
                # Prepare flow rate (m3/s) leaving the 1st order ditch for pond retention calculation
                if (Q_dp > 0) & (orderDitch == 1):
                    A_d1 = (Wd1_up + Wd1_lw) * HDiff_FD1 / 1000 / 2  # in m2，截面积
                    if D > 0:
                        Q_f0 = D * area_f / n1 / 24 / 3600 / 1000  # in m3/s,入流量
                        v_d1_out = (Q_f0 ** 0.4) * (slp_d1 ** 0.3) / (n_d1 ** 0.6) / (Wd1_up ** 0.4)  # in m/s，流速
                        Q_d1_out = v_d1_out * A_d1 * n1  # in m3/s,出流量
                        del v_d1_out
                    else:
                        Pw_d1 = Wd1_lw + (((Wd1_up - Wd1_lw) / 2) ** 2 + (HDiff_FD1 / 1000) ** 2) ** 0.5 * 2  # Wet perimeter, in m
                        Q_d1_out = (A_d1 ** 1.67) * (slp_d1 ** 0.5) / n_d1 / (Pw_d1 ** 0.67) * n1  # in m3/s
                        del Pw_d1
                    del A_d1

                # 2nd order ditch
                if orderDitch == 2:
                    # Record percolation concentration
                    TN_cd2b = TN_cd2 * TN_rate_lch
                    TP_cd2b = TP_cd2 * TP_rate_lch
                    # Prepare flow rate (m3/s) of the 2nd order ditch
                    if Hd2 > 0:
                        Pw_d2 = Wd2_lw + (((Wd2_up - Wd2_lw) / 2) ** 2 + (HDiff_FD2 / 1000) ** 2) ** 0.5 * \
                                Hd2 / HDiff_FD2 * 2  # Wet perimeter, in m
                        Wd2 = Wd2_lw + (Wd2_up - Wd2_lw) * Hd2 / HDiff_FD2  # water surface width, in m
                        A_d2 = (Wd2 + Wd2_lw) * Hd2 / 2000  # in m2
                        Q_d2 = (A_d2 ** 1.67) * (slp_d2 ** 0.5) / n_d2 / (Pw_d2 ** 0.67)  # in m3/s
                        del Pw_d2
                        del Wd2
                        del A_d2
                    else:
                        Q_d2 = 0.01
                    if D > 0:
                        # Calculate concentrations at the outlet of the 2nd order ditch by one-dimentional water quality model
                        # Prepare flow rate (m3/s) of 1st and 2nd order ditches for mixed concentration estimation #
                        Q_d1_0 = (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)
                                  + (prcp * area_d1 - kc_d1 * PET * area_d1_d - Fc_d1_v) / n1 / n2 - sd_functions.f_vhd(
                                    Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) / 1000  # in m3
                        TN_cd2_mix = (TN_cfb_lat * F_lat_d2 + TN_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TN_P * prcp * area_d2 / n1 / n2 + TN_cd1_out *
                                      Q_d1_0 * 1000) / (F_lat_d2 + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (prcp * area_d2 - kc_d2 * PET * area_d2_d
                                                                    ) / n1 / n2 + Q_d1_0 * 1000)
                        TP_cd2_mix = (TP_cfb_lat * F_lat_d2 + TP_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TP_P * prcp * area_d2 / n1 / n2 + TP_cd1_out *
                                      Q_d1_0 * 1000) / (F_lat_d2 + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (prcp * area_d2 - kc_d2 * PET * area_d2_d
                                                                    ) / n1 / n2 + Q_d1_0 * 1000)
                        # water surface width, in m
                        if Q_dp > 0:
                            Wd2 = Wd2_up
                            Hd2_d = HDiff_FD2 / 1000  # in m
                        else:
                            Hd2_d = max(0.01, (Hd2 + Hdp1 + HDiff_FD2 - HDiff_FD1) / 2000)  # in m
                            Wd2 = Wd2_lw + (Wd2_up - Wd2_lw) * Hd2_d * 1000 / HDiff_FD2
                        v_d2 = ((Q_d1_0 / 24 / 3600) ** 0.4) * (slp_d2 ** 0.3) / (n_d2 ** 0.6) / (
                                Wd2 ** 0.4)  # in m/s
                        # Calculate concentrations at the outlet
                        TN_rate_array = np.zeros(n1)
                        TP_rate_array = np.zeros(n1)
                        for i in range(0, n1):
                            TN_rate_array[i] = math.exp(- vf_N_pd / Hd2_d * (Wf1 * (i + 0.5) / v_d2 / 3600 / 24))
                            TP_rate_array[i] = math.exp(- vf_P_pd / Hd2_d * Wf1 * (i + 0.5) / v_d2 / 3600 / 24)
                        if TN_cd2_mix > ENC0_dp:
                            TN_cd2_out = TN_cd2_mix * TN_rate_array.mean()
                        else:
                            TN_cd2_out = TN_cd2_mix
                        if TP_cd2_mix > EPC0_dp:
                            TP_cd2_out = TP_cd2_mix * TP_rate_array.mean()
                        else:
                            TP_cd2_out = TP_cd2_mix
                        # Calculate the flow discharge at the outlet of the second-order ditch
                        v_d2_out = ((Q_d1_0 * n1 / 24 / 3600) ** 0.4) * (slp_d2 ** 0.3) / (n_d2 ** 0.6) / \
                                   (Wd2 ** 0.4)  # in m/s
                        A_d2 = (Wd2 + Wd2_lw) * Hd2_d / 2  # in m2
                        Q_d2_out = v_d2_out * A_d2 * n2  # in m3/s
                        del Wd2
                        del Hd2_d
                        del v_d2_out
                        del v_d2
                        del A_d2
                        del TN_rate_array
                        del TP_rate_array
                    else:
                        # Calculate mixed concentration with lateral flow, precipitation and ET
                        TN_cd2_mix = (TN_cfb_lat * F_lat_d2 + TN_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TN_P * prcp * area_d2 / n1 / n2) / (
                                             F_lat_d2 + sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (
                                             prcp * area_d2 - kc_d2 * PET * area_d2_d) / n1 / n2)
                        TP_cd2_mix = (TP_cfb_lat * F_lat_d2 + TP_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TP_P * prcp * area_d2 / n1 / n2) / (
                                             F_lat_d2 + sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (
                                             prcp * area_d2 - kc_d2 * PET * area_d2_d) / n1 / n2)
                        if TN_cd2_mix <= 0:
                            TN_cd2_mix = TN_cd2
                        if TP_cd2_mix <= 0:
                            TP_cd2_mix = TP_cd2
                        TN_cd2_out = TN_cd2_mix
                        TP_cd2_out = TP_cd2_mix
                        # Calculate the flow discharge at the outlet of the second-order ditch
                        Q_d2_out = Q_d2 * n2
                    # Calculate the retained concentration within the second-order ditch
                    if (Hdp1 + HDiff_FD2 - HDiff_FD1) > 0:
                        if TN_cd2_mix > ENC0_dp:
                            TN_cd2 = max(TN_cd2_mix * math.exp(
                                - vf_N_pd * 1 / max(0.05, (Hdp1 + HDiff_FD2 - HDiff_FD1) / 1000)), ENC0_dp)
                        else:
                            TN_cd2 = TN_cd2_mix
                        if TP_cd2_mix > EPC0_dp:
                            TP_cd2 = max(TP_cd2_mix * math.exp(
                                - vf_P_pd * 1 / max(0.05, (Hdp1 + HDiff_FD2 - HDiff_FD1) / 1000)), EPC0_dp)
                        else:
                            TP_cd2 = TP_cd2_mix
                    else:
                        TN_cd2 = TN_cd2_mix
                        TP_cd2 = TP_cd2_mix
                    # Record concentrations
                    TN_cd2_df[column].iloc[date_n] = (TN_cd2_out + TN_cd2) / 2
                    TP_cd2_df[column].iloc[date_n] = (TP_cd2_out + TP_cd2) / 2

                """ Record percolation TN and TP losses from the IDU """
                if havePond is True:
                    # Record percolation concentrations
                    TN_cpb = TN_cp * TN_rate_lch
                    TP_cpb = TP_cp * TP_rate_lch
                if (havePond is True) & (orderDitch == 2):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v + TN_cd2b * Fc_d2_v +
                                                    TN_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v + TP_cd2b * Fc_d2_v +
                                                    TP_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cd2b <= 10.0:
                        TN_cd2b_4wf = 0
                    else:
                        TN_cd2b_4wf = TN_cd2b
                    if TN_cpb <= 10.0:
                        TN_cpb_4wf = 0
                    else:
                        TN_cpb_4wf = TN_cpb
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cd2b_4wf * Fc_d2_v + TN_cpb_4wf * Fc_p_v) / (
                            area_f + area_pd) / 100
                elif (havePond is True) & (orderDitch == 1):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v +
                                                    TN_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v +
                                                    TP_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cpb <= 10.0:
                        TN_cpb_4wf = 0
                    else:
                        TN_cpb_4wf = TN_cpb
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cpb_4wf * Fc_p_v) / (
                            area_f + area_pd) / 100
                elif (havePond is False) & (orderDitch == 2):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v + TN_cd2b * Fc_d2_v
                                                    ) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v + TP_cd2b * Fc_d2_v
                                                    ) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cd2b <= 10.0:
                        TN_cd2b_4wf = 0
                    else:
                        TN_cd2b_4wf = TN_cd2b
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cd2b_4wf * Fc_d2_v) / (
                            area_f + area_pd) / 100
                elif (havePond is False) & (orderDitch == 1):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v) / (
                            area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v) / (
                            area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v) / (
                            area_f + area_pd) / 100

                """ Water quality in the pond """
                if havePond == True:
                    # Calculate mixed concentration first, and then calculate first-order retention/release
                    # Mixed concentration
                    if orderDitch == 2:
                        # total volume in ditches before balance, total_volume_d; and after balance, total_volume_d_bln; in mm.m2
                        total_volume_d = (D + F_lat) * area_f + n1 * n2 * (sd_functions.f_vhd(
                            Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) + prcp * (area_d1 + area_d2) \
                                         - kc_d1 * PET * area_d1_d - kc_d2 * PET * area_d2_d - Fc_d1_v - Fc_d2_v
                        total_volume_d_bln = n1 * n2 * (sd_functions.f_vhd(
                            Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + sd_functions.f_vhd(
                            Hdp1 + HDiff_FD2 - HDiff_FD1, HDiff_FD2, Wd2_up, Wd2_lw, Wf1))
                        # mix flow from ditches with the water in ponds and precipitation water, in mg/L
                        if total_volume_d > total_volume_d_bln:
                            TN_cp_mix = (TN_cd2_out * (total_volume_d - total_volume_d_bln) + TN_cp *
                                         sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cd2_out * (total_volume_d - total_volume_d_bln) + TP_cp *
                                         sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                        else:
                            TN_cp_mix = (TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (sd_functions.f_vhp(
                                Hp, HDiff_FP, area_p, slp_p) + prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (sd_functions.f_vhp(
                                Hp, HDiff_FP, area_p, slp_p) + prcp * area_p - kc_p * PET * area_p_d)
                    elif orderDitch == 1:
                        # total volume in ditches before balance, total_volume_d; and after balance, total_volume_d_bln; in mm.m2
                        total_volume_d = (D + F_lat) * area_f + n1 * sd_functions.f_vhd(
                            Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + prcp * area_d1 - kc_d1 * PET * area_d1_d - Fc_d1_v
                        total_volume_d_bln = n1 * sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)
                        # mix flow from ditches with the water in ponds and precipitation water, in mg/L
                        if total_volume_d > total_volume_d_bln:
                            TN_cp_mix = (TN_cd1_out * (total_volume_d - total_volume_d_bln) +
                                         TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cd1_out * (total_volume_d - total_volume_d_bln) +
                                         TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                        else:
                            TN_cp_mix = (TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                    if TN_cp_mix <= 0:
                        TN_cp_mix = TN_cp
                    if TP_cp_mix <= 0:
                        TP_cp_mix = TP_cp

                    # Calculate the retained concentration at the end of a day in the pond
                    if (Hdp1 + HDiff_FP - HDiff_FD1) > 0:
                        Hp_ave = (Hp + Hdp1 + HDiff_FP - HDiff_FD1) / 2000  # in m
                    if ((Hdp1 + HDiff_FP - HDiff_FD1) > 0) & (TN_cp_mix > ENC0_dp):
                        TN_cp = TN_cp_mix * math.exp(- vf_N_pd * 1 / max(0.05, Hp_ave))
                    else:
                        TN_cp = TN_cp_mix
                    if ((Hdp1 + HDiff_FP - HDiff_FD1) > 0) & (TP_cp_mix > EPC0_dp):
                        TP_cp = TP_cp_mix * math.exp(- vf_P_pd * 1 / max(0.05, Hp_ave))
                    else:
                        TP_cp = TP_cp_mix

                    # Calculate the concentrations at the outflow of the pond
                    if Q_dp > 0:
                        # Prepare the average depth in pond during the retained time
                        Hp_rt = (Hp + HDiff_FP) / 2000  # in m
                        if orderDitch == 2:
                            # Prepare the retained time in pond before outflow
                            TT = (sd_functions.f_vhp(HDiff_FP, HDiff_FP, area_p, slp_p) -
                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p)) / 1000 / Q_d2_out / 3600 / 24  # in d
                            # Calculate with first-order dynamic equation
                            if TN_cp_mix > ENC0_dp:
                                TN_cp_out = TN_cp_mix * math.exp(- vf_N_pd * TT / Hp_rt) * math.exp(
                                    - vf_N_pd * area_p / (Q_d2_out * 3600 * 24))
                            else:
                                TN_cp_out = TN_cp_mix
                            if TP_cp_mix > EPC0_dp:
                                TP_cp_out = TP_cp_mix * math.exp(- vf_P_pd * TT / Hp_rt) * math.exp(
                                    - vf_P_pd * area_p / (Q_d2_out * 3600 * 24))
                            else:
                                TP_cp_out = TP_cp_mix
                        elif orderDitch == 1:
                            # Prepare the retained time in pond before outflow
                            TT = (sd_functions.f_vhp(HDiff_FP, HDiff_FP, area_p, slp_p) -
                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p)) / 1000 / Q_d1_out / 3600 / 24  # in d
                            # Calculate with first-order dynamic equation
                            if TN_cp_mix > ENC0_dp:
                                TN_cp_out = TN_cp_mix * math.exp(- vf_N_pd * TT / Hp_rt) * math.exp(
                                    - vf_N_pd * area_p / (Q_d1_out * 3600 * 24))
                            else:
                                TN_cp_out = TN_cp_mix
                            if TP_cp_mix > EPC0_dp:
                                TP_cp_out = TP_cp_mix * math.exp(- vf_P_pd * TT / Hp_rt) * math.exp(
                                    - vf_P_pd * area_p / (Q_d1_out * 3600 * 24))
                            else:
                                TP_cp_out = TP_cp_mix
                        del Hp_rt
                        del TT
                        if water_adj_potential == 1:
                            TN_out_df[column].iloc[date_n] = (TN_cp_out * Q_dp + TN_prerisk * Q_dp_adj) / (
                                    Q_dp + Q_dp_adj)
                            TP_out_df[column].iloc[date_n] = (TP_cp_out * Q_dp + TP_prerisk * Q_dp_adj) / (
                                    Q_dp + Q_dp_adj)
                            del Q_dp_adj
                            del TN_prerisk
                            del TP_prerisk
                        else:
                            TN_out_df[column].iloc[date_n] = TN_cp_out
                            TP_out_df[column].iloc[date_n] = TP_cp_out
                    elif water_adj_potential == 1:
                        TN_out_df[column].iloc[date_n] = TN_prerisk
                        TP_out_df[column].iloc[date_n] = TP_prerisk
                        del Q_dp_adj
                        del TN_prerisk
                        del TP_prerisk

                     # Record TN and TP concentration
                    TN_cp_df[column].iloc[date_n] = (TN_cp_mix + TN_cp) / 2
                    TP_cp_df[column].iloc[date_n] = (TP_cp_mix + TP_cp) / 2

                elif orderDitch == 2:
                    if water_adj_potential == 1:
                        TN_out_df[column].iloc[date_n] = (TN_cd2_out * Q_dp + TN_prerisk * Q_dp_adj) / (Q_dp + Q_dp_adj)
                        TP_out_df[column].iloc[date_n] = (TP_cd2_out * Q_dp + TP_prerisk * Q_dp_adj) / (Q_dp + Q_dp_adj)
                        del Q_dp_adj
                        del TN_prerisk
                        del TP_prerisk
                    else:
                        TN_out_df[column].iloc[date_n] = TN_cd2_out
                        TP_out_df[column].iloc[date_n] = TP_cd2_out
                else:
                    if water_adj_potential == 1:
                        TN_out_df[column].iloc[date_n] = (TN_cd1_out * Q_dp + TN_prerisk * Q_dp_adj) / (Q_dp + Q_dp_adj)
                        TP_out_df[column].iloc[date_n] = (TP_cd1_out * Q_dp + TP_prerisk * Q_dp_adj) / (Q_dp + Q_dp_adj)
                        del Q_dp_adj
                        del TN_prerisk
                        del TP_prerisk
                    else:
                        TN_out_df[column].iloc[date_n] = TN_cd1_out
                        TP_out_df[column].iloc[date_n] = TP_cd1_out

                # Update water level in 1st, 2nd order ditches and in pond, in mm
                Hdp = Hdp1
                Hd1 = Hdp1
                if orderDitch == 2:
                    Hd2 = Hdp1 + HDiff_FD2 - HDiff_FD1
                if havePond is True:
                    Hp = Hdp1 + HDiff_FP - HDiff_FD1

            date_n += 1

    """ Calculate daily concentrations in surface flow from fields and from the IDU outlet """
    TN_cf_out = TN_cf_df.copy()
    TN_cf_out[D_df == 0] = np.NAN
    TP_cf_out = TP_cf_df.copy()
    TP_cf_out[D_df == 0] = np.NAN
    TN_out_df[Qdp_df == 0] = np.NAN
    TP_out_df[Qdp_df == 0] = np.NAN
    """ Calculate the total TN and TP loads, in kg/ha/d """
    # from fields
    TN_fs_df = TN_cf_df * D_df / 100
    TP_fs_df = TP_cf_df * D_df / 100
    TN_fb_lat_df = TN_cfb_lat_df * F_lat_df / 100
    TP_fb_lat_df = TP_cfb_lat_df * F_lat_df / 100
    TN_fb_df = TN_cfb_df * Fc_f_df / 100 + TN_fb_lat_df
    TP_fb_df = TP_cfb_df * Fc_f_df / 100 + TP_fb_lat_df
    TN_f_df = TN_fs_df + TN_fb_df
    TP_f_df = TP_fs_df + TP_fb_df
    # from system unit
    TN_s_df = TN_out_df.multiply(Qdp_df, fill_value=0.0) / 100
    TP_s_df = TP_out_df.multiply(Qdp_df, fill_value=0.0) / 100
    TN_sys_df = TN_s_df + TN_b_df
    TP_sys_df = TP_s_df + TP_b_df
    del TN_cfb_df
    del TP_cfb_df
    # for water footprint calculation
    TN_s_df_4wf = TN_s_df.where(TN_out_df > 2.0, 0.0)
    TP_s_df_4wf = TP_s_df.where(TP_out_df > 0.4, 0.0)
    TN_fs_df_4wf = TN_fs_df.where(TN_cf_df > 2.0, 0.0)
    TP_fs_df_4wf = TP_fs_df.where(TP_cf_df > 0.4, 0.0)
    TN_fb_lat_df_4wf = TN_fb_lat_df.where(TN_cfb_lat_df > 2.0, 0.0)
    TP_fb_lat_df_4wf = TP_fb_lat_df.where(TP_cfb_lat_df > 0.4, 0.0)
    TN_fs_A_4wf = (TN_fs_df_4wf + TN_fb_lat_df_4wf).resample("A", kind="period").sum()
    TP_fs_A_4wf = (TP_fs_df_4wf + TP_fb_lat_df_4wf).resample("A", kind="period").sum()
    TN_s_A_4wf = TN_s_df_4wf.resample("A", kind="period").sum()
    TP_s_A_4wf = TP_s_df_4wf.resample("A", kind="period").sum()
    TN_b_A_4wf = TN_b_df_4wf.resample("A", kind="period").sum()
    loads_4gwf = pd.DataFrame({"TN_runoff": TN_s_A_4wf.values.reshape(-1),
                               "TP_runoff": TP_s_A_4wf.values.reshape(-1),
                               "TN_leaching": TN_b_A_4wf.values.reshape(-1),
                               "TN_fs": TN_fs_A_4wf.values.reshape(-1),
                               "TP_fs": TP_fs_A_4wf.values.reshape(-1)})
    del TN_cfb_lat_df
    del TP_cfb_lat_df
    """ Calculate the proportions of surface and subsurface loads"""
    # from system unit
    TN_s_A = TN_s_df.resample('A', kind='period').sum()
    TN_b_A = TN_b_df.resample('A', kind='period').sum()
    TN_sys_A = TN_sys_df.resample('A', kind='period').sum()
    TP_s_A = TP_s_df.resample('A', kind='period').sum()
    TP_b_A = TP_b_df.resample('A', kind='period').sum()
    TP_sys_A = TP_sys_df.resample('A', kind='period').sum()
    TN_s_pr = TN_s_A / TN_sys_A
    TN_b_pr = TN_b_A / TN_sys_A
    TP_s_pr = TP_s_A / TP_sys_A
    TP_b_pr = TP_b_A / TP_sys_A
    result_sys_pr_df = pd.DataFrame({'TN_load(kg/ha/yr)': TN_sys_A.values.reshape(-1),
                                     'TN_surface(kg/ha/yr)': TN_s_A.values.reshape(-1),
                                     'TN_surface(rate)': TN_s_pr.values.reshape(-1),
                                     'TN_subsurface(kg/ha/yr)': TN_b_A.values.reshape(-1),
                                     'TN_subsurface(rate)': TN_b_pr.values.reshape(-1),
                                     'TP_load(kg/ha/yr)': TP_sys_A.values.reshape(-1),
                                     'TP_surface(kg/ha/yr)': TP_s_A.values.reshape(-1),
                                     'TP_surface(rate)': TP_s_pr.values.reshape(-1),
                                     'TP_subsurface(kg/ha/yr)': TP_b_A.values.reshape(-1),
                                     'TP_subsurface(rate)': TP_b_pr.values.reshape(-1)})
    result_sys_pr = result_sys_pr_df.describe()
    # from fields
    TN_fs_A = TN_fs_df.resample('A', kind='period').sum()
    TN_fb_A = TN_fb_df.resample('A', kind='period').sum()
    TN_fb_lat_A = TN_fb_lat_df.resample('A', kind='period').sum()
    TN_f_A = TN_f_df.resample('A', kind='period').sum()
    TP_fs_A = TP_fs_df.resample('A', kind='period').sum()
    TP_fb_A = TP_fb_df.resample('A', kind='period').sum()
    TP_fb_lat_A = TP_fb_lat_df.resample('A', kind='period').sum()
    TP_f_A = TP_f_df.resample('A', kind='period').sum()
    TN_fs_pr = TN_fs_A / TN_f_A
    TN_fb_pr = TN_fb_A / TN_f_A
    TP_fs_pr = TP_fs_A / TP_f_A
    TP_fb_pr = TP_fb_A / TP_f_A
    result_f_pr_df = pd.DataFrame({'TN_load(kg/ha/yr)': TN_f_A.values.reshape(-1),
                                   'TN_surface(kg/ha/yr)': TN_fs_A.values.reshape(-1),
                                   'TN_surface(rate)': TN_fs_pr.values.reshape(-1),
                                   'TN_subsurface(kg/ha/yr)': TN_fb_A.values.reshape(-1),
                                   'TN_subsurface(rate)': TN_fb_pr.values.reshape(-1),
                                   'TP_load(kg/ha/yr)': TP_f_A.values.reshape(-1),
                                   'TP_surface(kg/ha/yr)': TP_fs_A.values.reshape(-1),
                                   'TP_surface(rate)': TP_fs_pr.values.reshape(-1),
                                   'TP_subsurface(kg/ha/yr)': TP_fb_A.values.reshape(-1),
                                   'TP_subsurface(rate)': TP_fb_pr.values.reshape(-1)})
    result_f_pr = result_f_pr_df.describe()
    """ Calculate the annual average flow-weighted concentrations, in mg/L """
    # Concentrations in surface flow from fields
    TN_cs_f = TN_fs_A * 100 / (D_df.resample('A', kind='period').sum())
    TP_cs_f = TP_fs_A * 100 / (D_df.resample('A', kind='period').sum())
    # Concentrations in surface flow from system unit outlet
    TN_cs_sys = TN_s_A * 100 / (Qdp_df.resample('A', kind='period').sum())
    TP_cs_sys = TP_s_A * 100 / (Qdp_df.resample('A', kind='period').sum())

    """ Calculate the total nutrient inputs to and outputs from the field-ditch-pond system unit.
        The inputs are from irrigation and precipitation, the outputs are as surface runoff and leaching."""
    irrg_A = I_df.resample('A', kind='period').sum()
    prcp_A = prcp_d.resample('A', kind='period').sum()
    TN_I_out_A = TN_I_out_df.resample('A', kind='period').sum()
    TP_I_out_A = TP_I_out_df.resample('A', kind='period').sum()
    # field scale
    TN_I_pond_A = TN_I_pond_df.resample("A", kind='period').sum() / 100  # in kg/ha/yr
    TP_I_pond_A = TP_I_pond_df.resample('A', kind='period').sum() / 100  # in kg/ha/yr
    irrg_TN_input_f = TN_I_out_A / 100 + TN_I_pond_A  # in kg/ha/yr
    irrg_TP_input_f = TP_I_out_A / 100 + TP_I_pond_A  # in kg/ha/yr
    prcp_TN_input_f = prcp_A * TN_P / 100  # in kg/ha/yr
    prcp_TP_input_f = prcp_A * TP_P / 100  # in kg/ha/yr
    # IDU scale
    irrg_TN_input_sys = TN_I_out_A / 100 * area_f / (area_f + area_pd)  # in kg/ha/yr
    irrg_TP_input_sys = TP_I_out_A / 100 * area_f / (area_f + area_pd)  # in kg/ha/yr
    prcp_TN_input_sys = prcp_A * TN_P / 100  # in kg/ha/yr
    prcp_TP_input_sys = prcp_A * TP_P / 100  # in kg/ha/yr
    # Calculate the net nutrient export from fields and from the system unit, respectively.
    TN_export_f = TN_f_A - irrg_TN_input_f - prcp_TN_input_f
    TP_export_f = TP_f_A - irrg_TP_input_f - prcp_TP_input_f
    TN_export_sys = TN_sys_A - irrg_TN_input_sys - prcp_TN_input_sys
    TP_export_sys = TP_sys_A - irrg_TP_input_sys - prcp_TP_input_sys
    # The input-output dataframes
    TN_in_out = pd.concat([pd.DataFrame({"irrigation input": irrg_TN_input_f.values.reshape(-1),
                                         "precipitation input": prcp_TN_input_f.values.reshape(-1),
                                         "runoff loss": TN_fs_A.values.reshape(-1),
                                         "leaching loss": TN_fb_A.values.reshape(-1),
                                         "net export": TN_export_f.values.reshape(-1),
                                         "scale": "field"}),
                           pd.DataFrame({"irrigation input": irrg_TN_input_sys.values.reshape(-1),
                                         "precipitation input": prcp_TN_input_sys.values.reshape(-1),
                                         "runoff loss": TN_s_A.values.reshape(-1),
                                         "leaching loss": TN_b_A.values.reshape(-1),
                                         "net export": TN_export_sys.values.reshape(-1),
                                         "scale": "IDU"})],
                          ignore_index=True)
    TP_in_out = pd.concat([pd.DataFrame({"irrigation input": irrg_TP_input_f.values.reshape(-1),
                                         "precipitation input": prcp_TP_input_f.values.reshape(-1),
                                         "runoff loss": TP_fs_A.values.reshape(-1),
                                         "leaching loss": TP_fb_A.values.reshape(-1),
                                         "net export": TP_export_f.values.reshape(-1),
                                         "scale": "field"}),
                           pd.DataFrame({"irrigation input": irrg_TP_input_sys.values.reshape(-1),
                                         "precipitation input": prcp_TP_input_sys.values.reshape(-1),
                                         "runoff loss": TP_s_A.values.reshape(-1),
                                         "leaching loss": TP_b_A.values.reshape(-1),
                                         "net export": TP_export_sys.values.reshape(-1),
                                         "scale": "IDU"})],
                          ignore_index=True)
    # Calculate the export reduction by ditches and ponds
    TN_reduction = (TN_fs_A * area_f - TN_s_A * (area_f + area_pd)) / (TN_fs_A * area_f) * 100  # in %
    TP_reduction = (TP_fs_A * area_f - TP_s_A * (area_f + area_pd)) / (TP_fs_A * area_f) * 100  # in %
    TN_reduction = TN_reduction.where(TN_reduction > 0, 0)
    TN_reduction = TN_reduction.where(TN_reduction < 100, 100)
    TP_reduction = TP_reduction.where(TP_reduction > 0, 0)
    TP_reduction = TP_reduction.where(TP_reduction < 100, 100)
    result_export_df = pd.DataFrame({'TN_reduction(%)': TN_reduction.values.reshape(-1),
                                     'TP_reduction(%)': TP_reduction.values.reshape(-1)})
    result_export = result_export_df.describe()
    """ Calculate the percentage of irrigated water from pond to the total irrigated water content"""
    I_pond_A = I_pond_df.resample('A', kind='period').sum()
    I_pond_rate = (I_pond_A / irrg_A).values.reshape(-1)
    I_pond_rate_mean = I_pond_rate.mean() * 100
    I_pond_rate_min = I_pond_rate.min() * 100
    I_pond_rate_max = I_pond_rate.max() * 100
    I_out_A = irrg_A - I_pond_A
    # Calculating the rate of nutrient recycling (RNRc) and rate of water resource recycling (RWRc)
    RNRc_TN = TN_I_pond_A / (TN_fs_A + TN_fb_lat_A)
    RNRc_TP = TP_I_pond_A / (TP_fs_A + TP_fb_lat_A)
    RNRc_TN["nutrient"] = "TN"
    RNRc_TP["nutrient"] = "TP"
    RNRc = pd.concat([RNRc_TN, RNRc_TP], axis=0)
    water_f_A = (D_df + F_lat_df).resample("A", kind='period').sum()
    RWRc = I_pond_A / water_f_A

    # Prepare data for bmp comparison
    comp_data = {"TN_load": result_sys_pr.loc["mean", "TN_load(kg/ha/yr)"],
                 "TP_load": result_sys_pr.loc["mean", "TP_load(kg/ha/yr)"],
                 "TN_conc": round(np.nanmean(TN_cs_sys.values), 3),
                 "TP_conc": round(np.nanmean(TP_cs_sys.values), 3)}

    """ Write the results into text files """
    # Write daily irrigation into files
    with open(output_dir + '/I_d.txt', mode='w') as output_f:
        output_f.write(
            'The simulated irrigation (mm/d) to paddy fields are below:\n')
        output_f.close()
    I_df.to_csv(output_dir + '/I_d.txt', mode='a', header=True, index=True, sep='\t',
                float_format='%.4f')
    with open(output_dir + '/I_pond_d.txt', mode='w') as output_f:
        output_f.write(
            'The simulated irrigation (mm/d) from local ponds to paddy fields are below:\n')
        output_f.close()
    I_pond_df.to_csv(output_dir + '/I_pond_d.txt', mode='a', header=True, index=True, sep='\t',
                     float_format='%.4f')
    # Write runoff and leaching loads for water footprint calculation
    with open(output_dir + '/loads_4gwf{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write('The simulated nutrient loads (kg/ha) from IDUs for water footprint calculation are below:\n')
        output_f.close()
    loads_4gwf.to_csv(output_dir + '/loads_4gwf{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                       float_format='%.4f')
    # Write ET_d into files
    with open(output_dir + '/ET_d.txt', mode='w') as output_f:
        output_f.write(
            'The simulated ET (mm/d) from paddy fields are below:\n')
        output_f.close()
    ET_df.to_csv(output_dir + '/ET_d.txt', mode='a', header=True, index=True, sep='\t',
                 float_format='%.4f')
    # Write output summary file
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write('The simulated nutrient loads from the fields are summarized as below:\n')
        output_f.close()
    result_f_pr.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                       float_format='%.2f')
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
        output_f.write('\nThe simulated nutrient loads from the system unit are summarized as below:\n')
        output_f.close()
    result_sys_pr.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                         float_format='%.2f')
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
        output_f.write('\nThe reduction rate by ditches and ponds are summarized as below:\n')
        output_f.close()
    result_export.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                         float_format='%.2f')
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
        output_f.write('\nThe simulated percentage of irrigated water from pond is:\n')
        output_f.write('Mean (Range)\n')
        output_f.write('{0:.1f}% ({1:.1f}% - {2:.1f}%)'.format(I_pond_rate_mean, I_pond_rate_min, I_pond_rate_max))
        output_f.close()

    # Write average concentrations
    with open(output_dir + '/TN_cs_sys{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write(
            'The simulated flow-weighted annual average TN concentrations (mg/L) from system unit outlet are:\n')
        output_f.close()
    TN_cs_sys.to_csv(output_dir + '/TN_cs_sys{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                     float_format='%.3f')
    with open(output_dir + '/TP_cs_sys{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write(
            'The simulated flow-weighted annual average TP concentrations (mg/L) from system unit outlet are:\n')
        output_f.close()
    TP_cs_sys.to_csv(output_dir + '/TP_cs_sys{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                     float_format='%.3f')

    # Irrigated water content
    with open(output_dir + '/I_all{}.txt'.format(run_id), 'w') as f:
        f.write("The irrigated water content is simulated as below, in mm.\n")
        f.close()
    irrg_A.to_csv(output_dir + '/I_all{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    with open(output_dir + '/I_out{}.txt'.format(run_id), 'w') as f:
        f.write("The irrigated water content from out of the system is simulated as below, in mm.\n")
        f.close()
    I_out_A.to_csv(output_dir + '/I_out{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    # precipitation water content
    with open(output_dir + '/prcp_A.txt', 'w') as f:
        f.write("The precipitation water content during the growing season is simulated as below, in mm.\n")
        f.close()
    prcp_A.to_csv(output_dir + '/prcp_A.txt', mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    # RNRc and RWRc
    with open(output_dir + '/RNRc{}.txt'.format(run_id), 'w') as f:
        f.write("The rate of nutrient recycling is simulated as below, unitless.\n")
        f.close()
    RNRc.to_csv(output_dir + '/RNRc{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    with open(output_dir + '/RWRc{}.txt'.format(run_id), 'w') as f:
        f.write("The rate of water resource recycling is simulated as below, unitless.\n")
        f.close()
    RWRc.to_csv(output_dir + '/RWRc{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    # Annual load from system unit
    with open(output_dir + '/TN_load_sysA{}.txt'.format(run_id), 'w') as output_f:
        output_f.write("The annual TN loads (kg/ha) from the system unit are simulated as:\n")
        output_f.close()
    TN_sys_A.to_csv(output_dir + '/TN_load_sysA{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                    float_format='%.3f')
    with open(output_dir + '/TP_load_sysA{}.txt'.format(run_id), 'w') as output_f:
        output_f.write("The annual TP loads (kg/ha) from the system unit are simulated as:\n")
        output_f.close()
    TP_sys_A.to_csv(output_dir + '/TP_load_sysA{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                    float_format='%.3f')
    # TN reduction percentages
    TN_reduction.to_csv(output_dir + '/TN_reduction_rate{}.txt'.format(run_id), header=True, index=True, sep='\t',
                        float_format='%.2f')
    # TP reduction percentages
    TP_reduction.to_csv(output_dir + '/TP_reduction_rate{}.txt'.format(run_id), header=True, index=True, sep='\t',
                        float_format='%.2f')
    # Nutrient input output results.
    with open(output_dir + '/TN_in_out{}.txt'.format(run_id), mode='w') as f:
        f.write("The TN flux to and from the field V.S. IDU scale is as below, in kg/ha. \n")
        f.close()
    TN_in_out.to_csv(output_dir + '/TN_in_out{}.txt'.format(run_id), mode='a', header=True, index=False, sep='\t',
                     float_format='%.4f')
    with open(output_dir + '/TP_in_out{}.txt'.format(run_id), mode='w') as f:
        f.write("The TP flux to and from the field V.S. IDU scale is as below, in kg/ha. \n")
        f.close()
    TP_in_out.to_csv(output_dir + '/TP_in_out{}.txt'.format(run_id), mode='a', header=True, index=False, sep='\t',
                     float_format='%.4f')
    with open(output_dir + "/pre-risk_log{}.txt".format(run_id), mode='w') as f:
        f.write("The pre-risk drainage management was simulated to be conducted in dates below:\n")
        f.write(pre_risk_log.get_string())
        f.close()

    """ Prepare results summary for GUI """
    output_text = f"""||模拟结果小结||
                    \n计算时期内，水稻季的年平均模拟结果如下：
                    \n（1）田块尺度\n"""
    output_text += "TN年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_f_pr.loc["mean", "TN_load(kg/ha/yr)"], result_f_pr.loc["mean", "TN_surface(rate)"] * 100,
                                                      result_f_pr.loc["mean", "TN_subsurface(rate)"] * 100)
    output_text += "TP年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_f_pr.loc["mean", "TP_load(kg/ha/yr)"], result_f_pr.loc["mean", "TP_surface(rate)"] * 100,
                                                      result_f_pr.loc["mean", "TP_subsurface(rate)"] * 100)
    output_text += "地表径流中TN年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TN_cs_f.values))
    output_text += "地表径流中TP年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TP_cs_f.values))
    output_text += "\n（2）灌排单元尺度\n"
    output_text += "TN年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_sys_pr.loc["mean", "TN_load(kg/ha/yr)"], result_sys_pr.loc["mean", "TN_surface(rate)"] * 100,
                                                        result_sys_pr.loc["mean", "TN_subsurface(rate)"] * 100)
    output_text += "TP年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_sys_pr.loc["mean", "TP_load(kg/ha/yr)"], result_sys_pr.loc["mean", "TP_surface(rate)"] * 100,
                                                        result_sys_pr.loc["mean", "TP_subsurface(rate)"] * 100)
    output_text += "地表径流中TN年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TN_cs_sys.values))
    output_text += "地表径流中TP年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TP_cs_sys.values))

    if graph == 1:
        """ Convert outputs to time series """
        yearStart = prcp_d.index[0].year
        yearEnd = prcp_d.index[-1].year + 1
        Hdp_df = Hdp_df - HDiff_FD1
        for year in range(yearStart, yearEnd):
            Hf_df1[year] = Hf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            Hdp_df1[year] = Hdp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_cf_df1[year] = TP_cf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_cd1_df1[year] = TP_cd1_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_cf_df1[year] = TN_cf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_cd1_df1[year] = TN_cd1_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            if havePond:
                TP_cp_df1[year] = TP_cp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
                TN_cp_df1[year] = TN_cp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            if orderDitch == 2:
                TP_cd2_df1[year] = TP_cd2_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
                TN_cd2_df1[year] = TN_cd2_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_fs_df1[year] = TN_fs_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_s_df1[year] = TN_s_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_fs_df1[year] = TP_fs_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_s_df1[year] = TP_s_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values

        """Write Time series outputs """
        # water level in fields
        Hf_df1.to_csv(output_dir + '/H_f_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # water level in ditches and pond
        Hdp_df1.to_csv(output_dir + '/H_dp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN concentration in fields
        TN_cf_df1.to_csv(output_dir + '/TN_cf_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP concentration in fields
        TP_cf_df1.to_csv(output_dir + '/TP_cf_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN concentration in first-order ditch
        TN_cd1_df1.to_csv(output_dir + '/TN_cd1_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP concentration in first-order ditch
        TP_cd1_df1.to_csv(output_dir + '/TP_cd1_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        if orderDitch == 2:
            # TN concentration in second-order ditch
            TN_cd2_df1.to_csv(output_dir + '/TN_cd2_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
            # TP concentration in second-order ditch
            TP_cd2_df1.to_csv(output_dir + '/TP_cd2_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        if havePond:
            # TN concentration in pond
            TN_cp_df1.to_csv(output_dir + '/TN_cp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
            # TP concentration in pond
            TP_cp_df1.to_csv(output_dir + '/TP_cp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN load from field with surface runoff
        TN_fs_df1.to_csv(output_dir + '/TN_load_fs_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN load from system with surface runoff
        TN_s_df1.to_csv(output_dir + '/TN_load_syss_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP load from field with surface runoff
        TP_fs_df1.to_csv(output_dir + '/TP_load_fs_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP load from system with surface runoff
        TP_s_df1.to_csv(output_dir + '/TP_load_syss_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')

        """ Prepare dataframe list for drawing graphs """
        if orderDitch == 1:
            TN_cd2_df1 = 0
            TP_cd2_df1 = 0
        if not havePond:
            TN_cp_df1 = 0
            TP_cp_df1 = 0
        graph_dfs = list([Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1,
                          TN_cp_df1, TP_cp_df1, TN_fs_df1, TN_s_df1, TP_fs_df1, TP_s_df1, TN_in_out, TP_in_out])

    if graph == 1:
        return output_text, comp_data, graph_dfs
    else:
        return output_text, comp_data


def recycle_irrg(prcp_d, startDay, endDay, wtr_mng_d, pet_d, ditch_par, pond_par, F_doy_array, fert_type_array,
             TN0_array, TP0_array, TN_I, TP_I, area_f, Ks, Fc0_f, Fc0_pd, TN_P0, TN_P1, TP_P, TN_rate_lch, TN_rate_lch_lat,
             TP_rate_lch, TP_rate_lch_lat, ENC0_dp, EPC0_dp, ENCR_f, EPCR_f, vf_N_array, vf_P, h_v_sr, vf_N_pd, vf_P_pd,
             min_level, run_id, output_dir=0, graph=0):
    """ The pre-risk mode, for only accurate water level management.
        Parameters:
            prcp_d, pandas DataFrame stores daily precipitation data, in mm;
        Outputs:
            Hf"""
    Hf_df, Hdp_df, D_df, F_lat_df, Fc_f_df, Fc_pd_df, Qdp_df, I_df, TN_cf_df, TP_cf_df, TN_cd1_df, TP_cd1_df, \
    TN_cd2_df, TP_cd2_df, TN_cp_df, TP_cp_df, TN_cfb_df, TP_cfb_df, TN_cfb_lat_df, TP_cfb_lat_df, TN_b_df, TP_b_df, \
    TN_out_df, TP_out_df, Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1, \
    TN_cp_df1, TP_cp_df1, TN_fs_df1, TP_fs_df1, TN_s_df1, TP_s_df1 = create_common_df(
        prcp_d, startDay, endDay, wtr_mng_d)
    TN_b_df_4wf = TN_b_df.copy(deep=True)
    ET_df = Hf_df.copy(deep=True)
    I_pond_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                         dtype=np.float32)  # The amount of irrigated water from ponds, in mm/d
    TN_I_out_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                               dtype=np.float32)  # TN load from irrigated water from outside of the system, in mg/m2/d
    TP_I_out_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                               dtype=np.float32)  # TP load from irrigated water from outside of the system, in mg/m2/d
    TN_I_pond_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                                dtype=np.float32)  # TN load from irrigated water from pond, in mg/m2/d
    TP_I_pond_df = pd.DataFrame(index=prcp_d.index, columns=prcp_d.columns,
                                dtype=np.float32)  # TN load from irrigated water from pond, in mg/m2/d

    orderDitch = ditch_par[0]
    if orderDitch == 1:
        Wf1 = ditch_par[1][0]
        HDiff_FD1 = ditch_par[1][1]
        Wd1_up = ditch_par[1][2]
        Wd1_lw = ditch_par[1][3]
        n1 = ditch_par[1][4]
        slp_d1 = ditch_par[1][5]
        n_d1 = ditch_par[1][6]
        kc_d1 = ditch_par[1][7]
        TN_cd10 = ditch_par[1][8]
        TP_cd10 = ditch_par[1][9]
        Hd10 = ditch_par[1][10]
        area_d1 = ditch_par[1][11]
        area_d2 = 0
        Wf2 = area_f / n1 / Wf1
    else:
        Wf1 = ditch_par[1][0]
        HDiff_FD1 = ditch_par[1][1]
        Wd1_up = ditch_par[1][2]
        Wd1_lw = ditch_par[1][3]
        n1 = ditch_par[1][4]
        slp_d1 = ditch_par[1][5]
        n_d1 = ditch_par[1][6]
        kc_d1 = ditch_par[1][7]
        TN_cd10 = ditch_par[1][8]
        TP_cd10 = ditch_par[1][9]
        Hd10 = ditch_par[1][10]
        area_d1 = ditch_par[1][11]
        Wf2 = ditch_par[2][0]
        HDiff_FD2 = ditch_par[2][1]
        Wd2_up = ditch_par[2][2]
        Wd2_lw = ditch_par[2][3]
        n2 = ditch_par[2][4]
        slp_d2 = ditch_par[2][5]
        n_d2 = ditch_par[2][6]
        kc_d2 = ditch_par[2][7]
        TN_cd20 = ditch_par[2][8]
        TP_cd20 = ditch_par[2][9]
        area_d2 = ditch_par[2][10]
    havePond = pond_par[0]
    if havePond:
        HDiff_FP = pond_par[1][1]
        slp_p = pond_par[1][2]
        kc_p = pond_par[1][3]
        TN_cp0 = pond_par[1][4]
        TP_cp0 = pond_par[1][5]
        area_p = pond_par[1][6]
    else:
        area_p = 0.0

    # The rate of area of ditches and pond to the total area, in %
    area_pd = area_d1 + area_d2 + area_p

    startday_doy = pd.Timestamp(2018, int(startDay.split('-')[0]), int(startDay.split('-')[1])).dayofyear
    endday_doy = pd.Timestamp(2018, int(endDay.split('-')[0]), int(endDay.split('-')[1])).dayofyear
    for column in prcp_d.columns:
        date_n = 0
        while date_n < len(prcp_d.index):
            dayofYear = prcp_d.index[date_n].dayofyear
            monthDay = str(prcp_d.index[date_n].month) + "-" + str(prcp_d.index[date_n].day)
            if calendar.isleap(prcp_d.index[date_n].year):
                dayofYear = dayofYear - 1
            if dayofYear == startday_doy:
                Hf = 0.0
                Hdp = Hd10
                Hd1 = Hd10
                TP_cd1 = TP_cd10
                TN_cd1 = TN_cd10
                if orderDitch == 2:
                    Hd2 = Hd1 + HDiff_FD2 - HDiff_FD1
                    TP_cd2 = TP_cd20
                    TN_cd2 = TN_cd20
                if havePond:
                    Hp = Hd1 + HDiff_FP - HDiff_FD1
                    TP_cp = TP_cp0
                    TN_cp = TN_cp0
                irrg_date = wtr_mng_d["H_min"].ne(0).idxmax()
                irrg_doy = pd.Timestamp(2018, int(irrg_date.split('-')[0]), int(irrg_date.split('-')[1])).dayofyear
                if F_doy_array[0] <= irrg_doy:  # 施肥日在灌溉日之前，田面水浓度初始值为施肥初始浓度
                    TP_cf = TP0_array[0]
                    TN_cf = TN0_array[0]
                else:  # 施肥日在灌溉日之后，田面水浓度初始值为灌溉水浓度
                    TP_cf = TP_I
                    TN_cf = TN_I
                fert_lag = 0

            if (dayofYear < startday_doy) | (dayofYear > endday_doy):
                prcp_d.iloc[date_n] = 0.0  # Change the precipitation in non-growing period to 0, for statistics
            else:
                """ Water balance in fields """
                # Get the precipitation and PET data of the simulation day
                prcp = prcp_d[column].iloc[date_n]
                PET = pet_d[column].iloc[date_n]
                # Calculate ET
                if Hf > 0:
                    ET = PET * wtr_mng_d.loc[monthDay, 'Kc']
                else:
                    ET = PET * 0.60
                ET_df[column].iloc[date_n] = ET
                # Do irrigation or not
                if (wtr_mng_d.loc[monthDay, 'H_min'] != 0) & (Hf < wtr_mng_d.loc[monthDay, 'H_min']) & (
                        prcp + Hf - ET < wtr_mng_d.loc[monthDay, 'H_min']):
                    irrg = wtr_mng_d.loc[monthDay, 'H_max'] - Hf  # in mm
                    irrg_v = irrg * area_f  # in mm.m2
                    # 分配来自沟塘和外部水源的灌溉量
                    if (havePond is True) & (orderDitch == 2):
                        irrg_pond_max = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - sd_functions.f_vhp(
                            min_level, HDiff_FP, area_p, slp_p) + (sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) - sd_functions.f_vhd(
                            min_level, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) * n1 * n2
                    elif (havePond is True) & (orderDitch == 1):
                        irrg_pond_max = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - sd_functions.f_vhp(
                            min_level, HDiff_FP, area_p, slp_p)
                    elif orderDitch == 2:
                        irrg_pond_max = (sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) - sd_functions.f_vhd(
                            min_level, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) * n1 * n2
                    else:
                        irrg_pond_max = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) - sd_functions.f_vhd(
                            min_level, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) * n1
                    if irrg_pond_max > irrg_v:
                        irrg_out = 0
                        irrg_pond = irrg  # in mm
                    else:
                        irrg_pond = max(0, irrg_pond_max / area_f - 0.1)  # in mm
                        irrg_out = irrg - irrg_pond  # in mm
                    # 分配来自沟和塘的循环灌溉
                    if (havePond is True) & (orderDitch == 2) & (irrg_pond > 0):
                        irrg_ditch_max = (sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) - sd_functions.f_vhd(
                            min_level, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) * n1 * n2 / area_f
                        irrg_ditch_max = max(irrg_ditch_max, 0)
                        if irrg_ditch_max < irrg_pond:
                            irrg_pond_d = irrg_ditch_max
                            irrg_pond_p = irrg_pond - irrg_pond_d
                        else:
                            irrg_pond_d = irrg_pond
                            irrg_pond_p = 0
                    # 更新灌溉后沟塘水位和水质
                    if irrg_pond > 0:
                        Hdp1 = sd_functions.f_vhpd(Hdp, - irrg_pond * area_f, h_v_sr, area_pd, HDiff_FD1)[1]  # in mm
                        if (havePond is True) & (orderDitch == 2):
                            vd1 = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                   sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) * n1 * n2
                            vd2 = sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) * n1 * n2 - \
                                  irrg_pond_d * area_f
                            vp_ir = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - irrg_pond_p * area_f
                            vp = vp_ir - sd_functions.f_vhp(Hdp1, HDiff_FP, area_p, slp_p)
                            if vp > 0:
                                TN_cp1 = TN_cp
                                TP_cp1 = TP_cp
                                TN_cd21 = (TN_cd1 * vd1 + TN_cd2 * vd2 + TN_cp * vp) / (vd1 + vd2 + vp)
                                TP_cd21 = (TP_cd1 * vd1 + TP_cd2 * vd2 + TP_cp * vp) / (vd1 + vd2 + vp)
                            else:
                                TN_cd21 = (TN_cd1 * vd1 + TN_cd2 * vd2) / (vd1 + vd2)
                                TP_cd21 = (TP_cd1 * vd1 + TP_cd2 * vd2) / (vd1 + vd2)
                                TN_cp1 = (- TN_cd21 * vp + TN_cp * vp_ir) / (- vp + vp_ir)
                                TP_cp1 = (- TP_cd21 * vp + TP_cp * vp_ir) / (- vp + vp_ir)
                            Hd1 = Hdp1
                            Hd2 = Hd1 + HDiff_FD2 - HDiff_FD1
                            Hp = Hd1 + HDiff_FP - HDiff_FD1
                        elif (havePond is True) & (orderDitch == 1):
                            vd1 = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                   sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) * n1
                            vp = sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) - irrg_pond * area_f
                            TN_cp1 = (TN_cd1 * vd1 + TN_cp * vp) / (vd1 + vp)
                            TP_cp1 = (TP_cd1 * vd1 + TP_cp * vp) / (vd1 + vp)
                            Hd1 = Hdp1
                            Hp = Hd1 + HDiff_FP - HDiff_FD1
                        elif (havePond is False) & (orderDitch == 2):
                            vd1 = (sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) -
                                   sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) * n1 * n2
                            vd2 = sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) * n1 * n2 - irrg_pond * area_f
                            TN_cd21 = (TN_cd1 * vd1 + TN_cd2 * vd2) / (vd1 + vd2)
                            TP_cd21 = (TP_cd1 * vd1 + TP_cd2 * vd2) / (vd1 + vd2)
                            Hd1 = Hdp1
                            Hd2 = Hd1 + HDiff_FD2 - HDiff_FD1
                        elif (havePond is False) & (orderDitch == 1):
                            Hd1 = Hdp1
                        Hdp = Hdp1
                        del Hdp1
                else:
                    irrg = 0
                    irrg_out = 0
                    irrg_pond = 0
                    irrg_pond_d = 0
                    irrg_pond_p = 0
                if havePond is True:
                    if orderDitch == 2:
                        TN_I_pond_df[column].iloc[date_n] = TN_cp * irrg_pond_p + TN_cd2 * irrg_pond_d  # in mg/m2
                        TP_I_pond_df[column].iloc[date_n] = TP_cp * irrg_pond_p + TP_cd2 * irrg_pond_d  # in mg/m2
                    else:
                        TN_I_pond_df[column].iloc[date_n] = TN_cp * irrg_pond  # in mg/m2
                        TP_I_pond_df[column].iloc[date_n] = TP_cp * irrg_pond  # in mg/m2
                elif orderDitch == 2:
                    TN_I_pond_df[column].iloc[date_n] = TN_cd2 * irrg_pond  # in mg/m2
                    TP_I_pond_df[column].iloc[date_n] = TP_cd2 * irrg_pond  # in mg/m2
                else:
                    TN_I_pond_df[column].iloc[date_n] = TN_cd1 * irrg_pond  # in mg/m2
                    TP_I_pond_df[column].iloc[date_n] = TP_cd1 * irrg_pond  # in mg/m2
                TN_I_out_df[column].iloc[date_n] = TN_I * irrg_out  # in mg/m2
                TP_I_out_df[column].iloc[date_n] = TP_I * irrg_out  # in mg/m2
                I_df[column].iloc[date_n] = irrg  # in mm
                I_pond_df[column].iloc[date_n] = irrg_pond  # in mm
                # Calculate lateral seepage to ditches
                if orderDitch == 2:
                    F_lat_d1 = max(0, Ks * (Hf - Hd1 + HDiff_FD1) / (0.25 * Wf1 * 1000) * Wf2 * 2 * (
                            Hf + HDiff_FD1))  # in m2.mm/d
                    F_lat_d2 = max(0, Ks * (Hf - Hd2 + HDiff_FD2) / (0.25 * Wf2 * 1000) * Wf1 * 2 * (
                            Hf + HDiff_FD2))  # in m2.mm/d
                    F_lat = (F_lat_d1 + F_lat_d2) / (Wf1 * Wf2)  # in mm/d
                elif orderDitch == 1:
                    F_lat_d1 = max(0, Ks * (Hf - Hd1 + HDiff_FD1) / (0.25 * Wf1 * 1000) * Wf2 * 2 * (
                            Hf + HDiff_FD1))
                    F_lat = F_lat_d1 / (Wf1 * Wf2)  # in mm/d
                F_lat_df[column].iloc[date_n] = F_lat
                # Calculate downward percolation
                if (Hf - ET) > Fc0_f:
                    Fc_f = Fc0_f
                else:
                    Fc_f = 0
                Fc_f_df[column].iloc[date_n] = Fc_f
                # Update water level in fields, Hf, first-round
                Hf1 = Hf + prcp + irrg - ET - Fc_f - F_lat
                # Do drainage or not
                if Hf1 > wtr_mng_d.loc[monthDay, 'H_p']:
                    D = Hf1 - wtr_mng_d.loc[monthDay, 'H_p']  # in mm/d
                    Hf1 = Hf1 - D  # Update again, second-round
                else:
                    D = 0
                D_df[column].iloc[date_n] = D

                """ Water quality in fields"""
                # Initial TN and TP concentrations for the fertilization days, and
                # adjust TP and TN with irrigation, precipitation and ETs.
                flag_N = 0
                flag_P = 0
                TN_P = TN_P1
                for index, F_doy in enumerate(F_doy_array[TN0_array > 0]):
                    if (dayofYear >= F_doy) & (dayofYear - 15 <= F_doy):
                        TN_P = TN_P0
                for index, item in enumerate(F_doy_array):
                    if (dayofYear - fert_lag == item) & (D == 0):
                        fert_lag = 0
                        if TN0_array[index] != 0:
                            TN_cf = TN0_array[index]
                            flag_N = 1
                        if TP0_array[index] != 0:
                            TP_cf = TP0_array[index]
                            flag_P = 1
                    elif (dayofYear - fert_lag == item) & (D > 0):
                        if fert_lag < 5:
                            fert_lag += 1
                        else:
                            fert_lag = 0
                            if TN0_array[index] != 0:
                                TN_cf = TN0_array[index]
                                flag_N = 1
                            if TP0_array[index] != 0:
                                TP_cf = TP0_array[index]
                                flag_P = 1
                # Mixed concentration based on material balance
                if (Hf1 >= 10) & (Hf > 10):
                    if flag_N == 0:
                        TN_cf = (TN_cf * Hf + TN_I * irrg_out + TN_P * prcp +
                                 TN_I_pond_df[column].iloc[date_n]) / (Hf + irrg + prcp - ET)
                    if flag_P == 0:
                        TP_cf = (TP_cf * Hf + TP_I * irrg_out + TP_P * prcp +
                                 TP_I_pond_df[column].iloc[date_n]) / (Hf + irrg + prcp - ET)
                elif (Hf1 >= 10) & (Hf <= 10):
                    temp = max(min(10 + Hf, 10.0), 1)
                    if flag_N == 0:
                        TN_cf = (TN_cf * 2.0 * temp + TN_I * irrg_out + TN_P * prcp +
                                 TN_I_pond_df[column].iloc[date_n]) / (temp + irrg + prcp - ET)
                    if flag_P == 0:
                        TP_cf = (TP_cf * 4.0 * temp + TP_I * irrg_out + TP_P * prcp +
                                 TP_I_pond_df[column].iloc[date_n]) / (temp + irrg + prcp - ET)
                    del temp
                if irrg_pond > 0:
                    if havePond is True:
                        if orderDitch == 2:
                            TN_cp = TN_cp1
                            TP_cp = TP_cp1
                            TN_cd2 = TN_cd21
                            TP_cd2 = TP_cd21
                            del TN_cd21
                            del TP_cd21
                        else:
                            TN_cp = TN_cp1
                            TP_cp = TP_cp1
                        del TN_cp1
                        del TP_cp1
                    elif orderDitch == 2:
                        TN_cd2 = TN_cd21
                        TP_cd2 = TP_cd21
                        del TN_cd21
                        del TP_cd21
                # Adjust TN concentration for 2-7 days after tillering/filling fertilization
                for x, doy in enumerate(F_doy_array[fert_type_array == 2]):
                    if (dayofYear - 2 >= doy) & (dayofYear - 3 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.3
                    elif (dayofYear - 4 >= doy) & (dayofYear - 5 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.2
                    elif (dayofYear - 6 >= doy) & (dayofYear - 7 <= doy):
                        TN_cf = TN_cf + TN0_array[fert_type_array == 2][x] * 0.1
                    else:
                        TN_cf = TN_cf

                # Record percolated concentrations
                TN_cfb = TN_cf * TN_rate_lch
                TN_cfb_lat = max(TN_cf * TN_rate_lch_lat, ENC0_dp)
                TP_cfb = TP_cf * TP_rate_lch
                TP_cfb_lat = max(TP_cf * TP_rate_lch_lat, EPC0_dp)
                TP_cfb_df[column].iloc[date_n] = TP_cfb
                TN_cfb_df[column].iloc[date_n] = TN_cfb
                TP_cfb_lat_df[column].iloc[date_n] = TP_cfb_lat
                TN_cfb_lat_df[column].iloc[date_n] = TN_cfb_lat
                # Account for areal dynamics in the water-soil interface
                flag_N = 0
                flag_P = 0
                for index, F_doy in enumerate(F_doy_array[TN0_array > 0]):
                    if index + 1 < np.count_nonzero(TN0_array):
                        if (dayofYear > F_doy) & (dayofYear < F_doy_array[TN0_array > 0][index + 1]) & (Hf > 0) & (
                                Hf1 > 0) & (TN_cf > (ENCR_f * TN0_array[index])) & (flag_N == 0):
                            TN_cf1 = max(TN_cf * math.exp(- vf_N_array[index] * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         TN0_array[index] * ENCR_f)
                            flag_N = 1
                    else:
                        if (dayofYear > F_doy) & (Hf > 0) & (Hf1 > 0) & (TN_cf > (ENCR_f * TN0_array[index])) & (
                                flag_N == 0):
                            TN_cf1 = max(TN_cf * math.exp(- vf_N_array[index] * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         TN0_array[index] * ENCR_f)
                            flag_N = 1
                for index, F_doy in enumerate(F_doy_array[TP0_array > 0]):
                    if index + 1 < np.count_nonzero(TP0_array):
                        if (dayofYear > F_doy) & (dayofYear < F_doy_array[TP0_array > 0][index + 1]) & (Hf > 0) & (
                                Hf1 > 0) & (TP_cf > (EPCR_f * TP0_array[index])) & (flag_P == 0):
                            TP_cf1 = max(TP_cf * math.exp(- vf_P * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         EPCR_f * TP0_array[index])
                            flag_P = 1
                    else:
                        if (dayofYear > F_doy) & (Hf > 0) & (Hf1 > 0) & (TP_cf > (EPCR_f * TP0_array[index])) & (
                                flag_P == 0):
                            TP_cf1 = max(TP_cf * math.exp(- vf_P * 1 / max(0.05, (Hf + Hf1) / 2000)),
                                         EPCR_f * TP0_array[index])
                            flag_P = 1
                if flag_P == 0:
                    TP_cf1 = TP_cf
                if flag_N == 0:
                    TN_cf1 = TN_cf
                # Record water table and water quality in fields
                TP_cf_df[column].iloc[date_n] = (TP_cf + TP_cf1) / 2
                TN_cf_df[column].iloc[date_n] = (TN_cf + TN_cf1) / 2
                if Hf1 >= 0:
                    Hf_df[column].iloc[date_n] = Hf1
                else:
                    Hf_df[column].iloc[date_n] = 0.0
                Hf = Hf1

                """ Water balance in ditches and the pond """
                # Calculate ET and Fc from ditches and the pond
                if Hd1 > 0:
                    Wd1_d = (Wd1_up - Wd1_lw) * Hd1 / HDiff_FD1 + Wd1_lw
                    if orderDitch == 2:
                        area_d1_d = Wd1_d * Wf2 * n1 * n2
                    elif orderDitch == 1:
                        area_d1_d = Wd1_d * Wf2 * n1
                    if Hd1 - Fc0_pd > (kc_d1 * PET):
                        Fc_d1_v = Fc0_pd * area_d1_d
                    else:
                        Fc_d1_v = 0.0
                else:
                    area_d1_d = 0
                    Fc_d1_v = 0.0

                if orderDitch == 2:
                    if Hd2 > 0:
                        Wd2_d = (Wd2_up - Wd2_lw) * Hd2 / HDiff_FD2 + Wd2_lw
                        area_d2_d = Wd2_d * Wf1 * n1 * n2
                        if Hd2 - Fc0_pd > (kc_d2 * PET):
                            Fc_d2_v = Fc0_pd * area_d2_d
                        else:
                            Fc_d2_v = 0.0
                    else:
                        area_d2_d = 0
                        Fc_d2_v = 0.0

                if havePond == True:
                    if Hp > 0:
                        area_p_lw = (math.sqrt(area_p) - HDiff_FP / 1000 * slp_p * 2) ** 2  # in m2
                        hp1 = math.sqrt(area_p_lw) / 2 / slp_p  # in m
                        area_p_d = area_p_lw * ((hp1 + Hp / 1000) ** 2) / (hp1 ** 2)  # in m2
                        if Hp - Fc0_pd > (kc_p * PET):
                            Fc_p_v = Fc0_pd * area_p_d
                        else:
                            Fc_p_v = 0.0
                    else:
                        area_p_d = 0
                        Fc_p_v = 0.0
                # ET, ET_pd;, downward percoloation, Fc_pd
                if (havePond is True) & (orderDitch == 2):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_d2 * PET * area_d2_d + kc_p * PET * area_p_d + \
                              0.60 * PET * (area_pd - area_d1_d - area_d2_d - area_p_d)
                    Fc_pd_v = Fc_d1_v + Fc_d2_v + Fc_p_v
                elif (havePond is True) & (orderDitch == 1):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_p * PET * area_p_d + 0.60 * PET * (
                            area_pd - area_d1_d - area_p_d)
                    Fc_pd_v = Fc_d1_v + Fc_p_v
                elif (havePond is False) & (orderDitch == 2):
                    ET_pd_v = kc_d1 * PET * area_d1_d + kc_d2 * PET * area_d2_d + 0.60 * PET * (
                            area_pd - area_d1_d - area_d2_d)
                    Fc_pd_v = Fc_d1_v + Fc_d2_v
                elif (havePond is False) & (orderDitch == 1):
                    ET_pd_v = kc_d1 * PET * area_d1_d + 0.60 * PET * (
                            area_pd - area_d1_d)
                    Fc_pd_v = Fc_d1_v

                # Update water level in ditches and ponds, Hdp; and calculate outflow Q_dp of the system from the pond
                volume_add = prcp * area_pd - Fc_pd_v - ET_pd_v + (F_lat + D) * area_f
                Q_dp = sd_functions.f_vhpd(Hdp, volume_add, h_v_sr, area_pd, HDiff_FD1)[0]  # in m3/d
                Hdp1 = sd_functions.f_vhpd(Hdp, volume_add, h_v_sr, area_pd, HDiff_FD1)[1]  # in mm
                Qdp_df[column].iloc[date_n] = Q_dp / (area_f + area_pd) * 1000  # in mm/d
                Hdp_df[column].iloc[date_n] = Hdp1

                """ Water quality in ditches """
                # 1st order ditch
                # Record percolation concentration
                TN_cd1b = TN_cd1 * TN_rate_lch
                TP_cd1b = TP_cd1 * TP_rate_lch
                # Calculate mixed concentration
                if orderDitch == 2:
                    TN_cd1_out = (TN_cf * D * Wf1 * Wf2 + TN_cfb_lat * F_lat_d1 + TN_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TN_P * prcp * area_d1 / n1 / n2
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1 / n2)
                    TP_cd1_out = (TP_cf * D * Wf1 * Wf2 + TP_cfb_lat * F_lat_d1 + TP_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TP_P * prcp * area_d1 / n1 / n2
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1 / n2)
                elif orderDitch == 1:
                    TN_cd1_out = (TN_cf * D * Wf1 * Wf2 + TN_cfb_lat * F_lat_d1 + TN_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TN_P * prcp * area_d1 / n1
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1)
                    TP_cd1_out = (TP_cf * D * Wf1 * Wf2 + TP_cfb_lat * F_lat_d1 + TP_cd1 * sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + TP_P * prcp * area_d1 / n1
                                  ) / (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(
                        Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + (prcp * area_d1 - kc_d1 * PET * area_d1_d) / n1)
                if TN_cd1_out <= 0:
                    TN_cd1_out = TN_cd1
                if TP_cd1_out <= 0:
                    TP_cd1_out = TP_cd1
                # Calculate retained concentration
                if Hdp1 > 0:
                    if TN_cd1 > ENC0_dp:
                        TN_cd1 = max(TN_cd1_out * math.exp(- vf_N_pd * 1 / max(0.05, Hdp1 / 1000)), ENC0_dp)
                    else:
                        TN_cd1 = TN_cd1_out
                    if TP_cd1_out > EPC0_dp:
                        TP_cd1 = max(TP_cd1_out * math.exp(- vf_P_pd * 1 / max(0.05, Hdp1 / 1000)), EPC0_dp)
                    else:
                        TP_cd1 = TP_cd1_out
                else:
                    TN_cd1 = TN_cd1_out
                    TP_cd1 = TP_cd1_out
                # Record concentrations
                TN_cd1_df[column].iloc[date_n] = (TN_cd1_out + TN_cd1) / 2
                TP_cd1_df[column].iloc[date_n] = (TP_cd1_out + TP_cd1) / 2
                TN_cf = TN_cf1
                TP_cf = TP_cf1
                # Prepare flow rate (m3/s) leaving the 1st order ditch for pond retention calculation
                if (Q_dp > 0) & (orderDitch == 1):
                    A_d1 = (Wd1_up + Wd1_lw) * HDiff_FD1 / 1000 / 2  # in m2，截面积
                    if D > 0:
                        Q_f0 = D * area_f / n1 / 24 / 3600 / 1000  # in m3/s,入流量
                        v_d1_out = (Q_f0 ** 0.4) * (slp_d1 ** 0.3) / (n_d1 ** 0.6) / (Wd1_up ** 0.4)  # in m/s，流速
                        Q_d1_out = v_d1_out * A_d1 * n1  # in m3/s,出流量
                        del v_d1_out
                    else:
                        Pw_d1 = Wd1_lw + (((Wd1_up - Wd1_lw) / 2) ** 2 + (HDiff_FD1 / 1000) ** 2) ** 0.5 * 2  # Wet perimeter, in m
                        Q_d1_out = (A_d1 ** 1.67) * (slp_d1 ** 0.5) / n_d1 / (Pw_d1 ** 0.67) * n1  # in m3/s
                        del Pw_d1
                    del A_d1

                # 2nd order ditch
                if orderDitch == 2:
                    # Record percolation concentration
                    TN_cd2b = TN_cd2 * TN_rate_lch
                    TP_cd2b = TP_cd2 * TP_rate_lch
                    # Prepare flow rate (m3/s) of the 2nd order ditch
                    if Hd2 > 0:
                        Pw_d2 = Wd2_lw + (((Wd2_up - Wd2_lw) / 2) ** 2 + (HDiff_FD2 / 1000) ** 2) ** 0.5 * \
                                Hd2 / HDiff_FD2 * 2  # Wet perimeter, in m
                        Wd2 = Wd2_lw + (Wd2_up - Wd2_lw) * Hd2 / HDiff_FD2  # water surface width, in m
                        A_d2 = (Wd2 + Wd2_lw) * Hd2 / 2000  # in m2
                        Q_d2 = (A_d2 ** 1.67) * (slp_d2 ** 0.5) / n_d2 / (Pw_d2 ** 0.67)  # in m3/s
                        del Pw_d2
                        del Wd2
                        del A_d2
                    else:
                        Q_d2 = 0.01
                    if D > 0:
                        # Calculate concentrations at the outlet of the 2nd order ditch by one-dimentional water quality model
                        # Prepare flow rate (m3/s) of 1st and 2nd order ditches for mixed concentration estimation #
                        Q_d1_0 = (D * Wf1 * Wf2 + F_lat_d1 + sd_functions.f_vhd(Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)
                                  + (prcp * area_d1 - kc_d1 * PET * area_d1_d - Fc_d1_v) / n1 / n2 - sd_functions.f_vhd(
                                    Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)) / 1000  # in m3
                        TN_cd2_mix = (TN_cfb_lat * F_lat_d2 + TN_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TN_P * prcp * area_d2 / n1 / n2 + TN_cd1_out *
                                      Q_d1_0 * 1000) / (F_lat_d2 + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (prcp * area_d2 - kc_d2 * PET * area_d2_d
                                                                    ) / n1 / n2 + Q_d1_0 * 1000)
                        TP_cd2_mix = (TP_cfb_lat * F_lat_d2 + TP_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TP_P * prcp * area_d2 / n1 / n2 + TP_cd1_out *
                                      Q_d1_0 * 1000) / (F_lat_d2 + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (prcp * area_d2 - kc_d2 * PET * area_d2_d
                                                                    ) / n1 / n2 + Q_d1_0 * 1000)
                        # water surface width, in m
                        if Q_dp > 0:
                            Wd2 = Wd2_up
                            Hd2_d = HDiff_FD2 / 1000  # in m
                        else:
                            Hd2_d = max(0.01, (Hd2 + Hdp1 + HDiff_FD2 - HDiff_FD1) / 2000)  # in m
                            Wd2 = Wd2_lw + (Wd2_up - Wd2_lw) * Hd2_d * 1000 / HDiff_FD2
                        v_d2 = ((Q_d1_0 / 24 / 3600) ** 0.4) * (slp_d2 ** 0.3) / (n_d2 ** 0.6) / (
                                Wd2 ** 0.4)  # in m/s
                        # Calculate concentrations at the outlet
                        TN_rate_array = np.zeros(n1)
                        TP_rate_array = np.zeros(n1)
                        for i in range(0, n1):
                            TN_rate_array[i] = math.exp(- vf_N_pd / Hd2_d * (Wf1 * (i + 0.5) / v_d2 / 3600 / 24))
                            TP_rate_array[i] = math.exp(- vf_P_pd / Hd2_d * Wf1 * (i + 0.5) / v_d2 / 3600 / 24)
                        if TN_cd2_mix > ENC0_dp:
                            TN_cd2_out = TN_cd2_mix * TN_rate_array.mean()
                        else:
                            TN_cd2_out = TN_cd2_mix
                        if TP_cd2_mix > EPC0_dp:
                            TP_cd2_out = TP_cd2_mix * TP_rate_array.mean()
                        else:
                            TP_cd2_out = TP_cd2_mix
                        # Calculate the flow discharge at the outlet of the second-order ditch
                        v_d2_out = ((Q_d1_0 * n1 / 24 / 3600) ** 0.4) * (slp_d2 ** 0.3) / (n_d2 ** 0.6) / \
                                   (Wd2 ** 0.4)  # in m/s
                        A_d2 = (Wd2 + Wd2_lw) * Hd2_d / 2  # in m2
                        Q_d2_out = v_d2_out * A_d2 * n2  # in m3/s
                        del Wd2
                        del Hd2_d
                        del v_d2_out
                        del v_d2
                        del A_d2
                        del TN_rate_array
                        del TP_rate_array
                    else:
                        # Calculate mixed concentration with lateral flow, precipitation and ET
                        TN_cd2_mix = (TN_cfb_lat * F_lat_d2 + TN_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TN_P * prcp * area_d2 / n1 / n2) / (
                                             F_lat_d2 + sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (
                                             prcp * area_d2 - kc_d2 * PET * area_d2_d) / n1 / n2)
                        TP_cd2_mix = (TP_cfb_lat * F_lat_d2 + TP_cd2 * sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + TP_P * prcp * area_d2 / n1 / n2) / (
                                             F_lat_d2 + sd_functions.f_vhd(Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1) + (
                                             prcp * area_d2 - kc_d2 * PET * area_d2_d) / n1 / n2)
                        if TN_cd2_mix <= 0:
                            TN_cd2_mix = TN_cd2
                        if TP_cd2_mix <= 0:
                            TP_cd2_mix = TP_cd2
                        TN_cd2_out = TN_cd2_mix
                        TP_cd2_out = TP_cd2_mix
                        # Calculate the flow discharge at the outlet of the second-order ditch
                        Q_d2_out = Q_d2 * n2
                    # Calculate the retained concentration within the second-order ditch
                    if (Hdp1 + HDiff_FD2 - HDiff_FD1) > 0:
                        if TN_cd2_mix > ENC0_dp:
                            TN_cd2 = max(TN_cd2_mix * math.exp(
                                - vf_N_pd * 1 / max(0.05, (Hdp1 + HDiff_FD2 - HDiff_FD1) / 1000)), ENC0_dp)
                        else:
                            TN_cd2 = TN_cd2_mix
                        if TP_cd2_mix > EPC0_dp:
                            TP_cd2 = max(TP_cd2_mix * math.exp(
                                - vf_P_pd * 1 / max(0.05, (Hdp1 + HDiff_FD2 - HDiff_FD1) / 1000)), EPC0_dp)
                        else:
                            TP_cd2 = TP_cd2_mix
                    else:
                        TN_cd2 = TN_cd2_mix
                        TP_cd2 = TP_cd2_mix
                    # Record concentrations
                    TN_cd2_df[column].iloc[date_n] = (TN_cd2_out + TN_cd2) / 2
                    TP_cd2_df[column].iloc[date_n] = (TP_cd2_out + TP_cd2) / 2

                """ Record percolation TN and TP losses from the IDU """
                if havePond is True:
                    # Record percolation concentrations
                    TN_cpb = TN_cp * TN_rate_lch
                    TP_cpb = TP_cp * TP_rate_lch
                if (havePond is True) & (orderDitch == 2):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v + TN_cd2b * Fc_d2_v +
                                                    TN_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v + TP_cd2b * Fc_d2_v +
                                                    TP_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cd2b <= 10.0:
                        TN_cd2b_4wf = 0
                    else:
                        TN_cd2b_4wf = TN_cd2b
                    if TN_cpb <= 10.0:
                        TN_cpb_4wf = 0
                    else:
                        TN_cpb_4wf = TN_cpb
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cd2b_4wf * Fc_d2_v + TN_cpb_4wf * Fc_p_v) / (
                                                               area_f + area_pd) / 100
                elif (havePond is True) & (orderDitch == 1):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v +
                                                    TN_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v +
                                                    TP_cpb * Fc_p_v) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cpb <= 10.0:
                        TN_cpb_4wf = 0
                    else:
                        TN_cpb_4wf = TN_cpb
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cpb_4wf * Fc_p_v) / (
                                                               area_f + area_pd) / 100
                elif (havePond is False) & (orderDitch == 2):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v + TN_cd2b * Fc_d2_v
                                                    ) / (area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v + TP_cd2b * Fc_d2_v
                                                    ) / (area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    if TN_cd2b <= 10.0:
                        TN_cd2b_4wf = 0
                    else:
                        TN_cd2b_4wf = TN_cd2b
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v +
                                                        TN_cd2b_4wf * Fc_d2_v) / (
                                                               area_f + area_pd) / 100
                elif (havePond is False) & (orderDitch == 1):
                    TN_b_df[column].iloc[date_n] = (TN_cfb * Fc_f * area_f + TN_cd1b * Fc_d1_v) / (
                            area_f + area_pd) / 100
                    TP_b_df[column].iloc[date_n] = (TP_cfb * Fc_f * area_f + TP_cd1b * Fc_d1_v) / (
                            area_f + area_pd) / 100
                    if TN_cfb <= 10.0:
                        TN_cfb_4wf = 0
                    else:
                        TN_cfb_4wf = TN_cfb
                    if TN_cd1b <= 10.0:
                        TN_cd1b_4wf = 0
                    else:
                        TN_cd1b_4wf = TN_cd1b
                    TN_b_df_4wf[column].iloc[date_n] = (TN_cfb_4wf * Fc_f * area_f + TN_cd1b_4wf * Fc_d1_v) / (
                            area_f + area_pd) / 100

                """ Water quality in the pond """
                if havePond == True:
                    # Calculate mixed concentration first, and then calculate first-order retention/release
                    # Mixed concentration
                    if orderDitch == 2:
                        # total volume in ditches before balance, total_volume_d; and after balance, total_volume_d_bln; in mm.m2
                        total_volume_d = (D + F_lat) * area_f + n1 * n2 * (sd_functions.f_vhd(
                            Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + sd_functions.f_vhd(
                            Hd2, HDiff_FD2, Wd2_up, Wd2_lw, Wf1)) + prcp * (area_d1 + area_d2) \
                                         - kc_d1 * PET * area_d1_d - kc_d2 * PET * area_d2_d - Fc_d1_v - Fc_d2_v
                        total_volume_d_bln = n1 * n2 * (sd_functions.f_vhd(
                            Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + sd_functions.f_vhd(
                            Hdp1 + HDiff_FD2 - HDiff_FD1, HDiff_FD2, Wd2_up, Wd2_lw, Wf1))
                        # mix flow from ditches with the water in ponds and precipitation water, in mg/L
                        if total_volume_d > total_volume_d_bln:
                            TN_cp_mix = (TN_cd2_out * (total_volume_d - total_volume_d_bln) + TN_cp *
                                         sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cd2_out * (total_volume_d - total_volume_d_bln) + TP_cp *
                                         sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                        else:
                            TN_cp_mix = (TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (sd_functions.f_vhp(
                                Hp, HDiff_FP, area_p, slp_p) + prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (sd_functions.f_vhp(
                                Hp, HDiff_FP, area_p, slp_p) + prcp * area_p - kc_p * PET * area_p_d)
                    elif orderDitch == 1:
                        # total volume in ditches before balance, total_volume_d; and after balance, total_volume_d_bln; in mm.m2
                        total_volume_d = (D + F_lat) * area_f + n1 * sd_functions.f_vhd(
                            Hd1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2) + prcp * area_d1 - kc_d1 * PET * area_d1_d - Fc_d1_v
                        total_volume_d_bln = n1 * sd_functions.f_vhd(Hdp1, HDiff_FD1, Wd1_up, Wd1_lw, Wf2)
                        # mix flow from ditches with the water in ponds and precipitation water, in mg/L
                        if total_volume_d > total_volume_d_bln:
                            TN_cp_mix = (TN_cd1_out * (total_volume_d - total_volume_d_bln) +
                                         TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cd1_out * (total_volume_d - total_volume_d_bln) +
                                         TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (total_volume_d - total_volume_d_bln +
                                                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                        else:
                            TN_cp_mix = (TN_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TN_P * prcp * area_p) / (sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                            TP_cp_mix = (TP_cp * sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                         TP_P * prcp * area_p) / (sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p) +
                                                                  prcp * area_p - kc_p * PET * area_p_d)
                    if TN_cp_mix <= 0:
                        TN_cp_mix = TN_cp
                    if TP_cp_mix <= 0:
                        TP_cp_mix = TP_cp

                    # Calculate the retained concentration at the end of a day in the pond
                    if (Hdp1 + HDiff_FP - HDiff_FD1) > 0:
                        Hp_ave = (Hp + Hdp1 + HDiff_FP - HDiff_FD1) / 2000  # in m
                    if ((Hdp1 + HDiff_FP - HDiff_FD1) > 0) & (TN_cp_mix > ENC0_dp):
                        TN_cp = TN_cp_mix * math.exp(- vf_N_pd * 1 / max(0.05, Hp_ave))
                    else:
                        TN_cp = TN_cp_mix
                    if ((Hdp1 + HDiff_FP - HDiff_FD1) > 0) & (TP_cp_mix > EPC0_dp):
                        TP_cp = TP_cp_mix * math.exp(- vf_P_pd * 1 / max(0.05, Hp_ave))
                    else:
                        TP_cp = TP_cp_mix

                    # Calculate the concentrations at the outflow of the pond
                    if Q_dp > 0:
                        # Prepare the average depth in pond during the retained time
                        Hp_rt = (Hp + HDiff_FP) / 2000  # in m
                        if orderDitch == 2:
                            # Prepare the retained time in pond before outflow
                            TT = (sd_functions.f_vhp(HDiff_FP, HDiff_FP, area_p, slp_p) -
                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p)) / 1000 / Q_d2_out / 3600 / 24  # in d
                            # Calculate with first-order dynamic equation
                            if TN_cp_mix > ENC0_dp:
                                TN_cp_out = TN_cp_mix * math.exp(- vf_N_pd * TT / Hp_rt) * math.exp(
                                    - vf_N_pd * area_p / (Q_d2_out * 3600 * 24))
                            else:
                                TN_cp_out = TN_cp_mix
                            if TP_cp_mix > EPC0_dp:
                                TP_cp_out = TP_cp_mix * math.exp(- vf_P_pd * TT / Hp_rt) * math.exp(
                                    - vf_P_pd * area_p / (Q_d2_out * 3600 * 24))
                            else:
                                TP_cp_out = TP_cp_mix
                        elif orderDitch == 1:
                            # Prepare the retained time in pond before outflow
                            TT = (sd_functions.f_vhp(HDiff_FP, HDiff_FP, area_p, slp_p) -
                                  sd_functions.f_vhp(Hp, HDiff_FP, area_p, slp_p)) / 1000 / Q_d1_out / 3600 / 24  # in d
                            # Calculate with first-order dynamic equation
                            if TN_cp_mix > ENC0_dp:
                                TN_cp_out = TN_cp_mix * math.exp(- vf_N_pd * TT / Hp_rt) * math.exp(
                                    - vf_N_pd * area_p / (Q_d1_out * 3600 * 24))
                            else:
                                TN_cp_out = TN_cp_mix
                            if TP_cp_mix > EPC0_dp:
                                TP_cp_out = TP_cp_mix * math.exp(- vf_P_pd * TT / Hp_rt) * math.exp(
                                    - vf_P_pd * area_p / (Q_d1_out * 3600 * 24))
                            else:
                                TP_cp_out = TP_cp_mix
                        del Hp_rt
                        del TT
                        TN_out_df[column].iloc[date_n] = TN_cp_out
                        TP_out_df[column].iloc[date_n] = TP_cp_out

                     # Record TN and TP concentration
                    TN_cp_df[column].iloc[date_n] = (TN_cp_mix + TN_cp) / 2
                    TP_cp_df[column].iloc[date_n] = (TP_cp_mix + TP_cp) / 2

                elif orderDitch == 2:
                    TN_out_df[column].iloc[date_n] = TN_cd2_out
                    TP_out_df[column].iloc[date_n] = TP_cd2_out
                else:
                    TN_out_df[column].iloc[date_n] = TN_cd1_out
                    TP_out_df[column].iloc[date_n] = TP_cd1_out

                # Update water level in 1st, 2nd order ditches and in pond, in mm
                Hdp = Hdp1
                Hd1 = Hdp1
                if orderDitch == 2:
                    Hd2 = Hdp1 + HDiff_FD2 - HDiff_FD1
                if havePond is True:
                    Hp = Hdp1 + HDiff_FP - HDiff_FD1

            date_n += 1

    """ Calculate daily concentrations in surface flow from fields and from the IDU outlet """
    TN_cf_out = TN_cf_df.copy()
    TN_cf_out[D_df == 0] = np.NAN
    TP_cf_out = TP_cf_df.copy()
    TP_cf_out[D_df == 0] = np.NAN
    TN_out_df[Qdp_df == 0] = np.NAN
    TP_out_df[Qdp_df == 0] = np.NAN
    """ Calculate the total TN and TP loads, in kg/ha/d """
    # from fields
    TN_fs_df = TN_cf_df * D_df / 100
    TP_fs_df = TP_cf_df * D_df / 100
    TN_fb_lat_df = TN_cfb_lat_df * F_lat_df / 100
    TP_fb_lat_df = TP_cfb_lat_df * F_lat_df / 100
    TN_fb_df = TN_cfb_df * Fc_f_df / 100 + TN_fb_lat_df
    TP_fb_df = TP_cfb_df * Fc_f_df / 100 + TP_fb_lat_df
    TN_f_df = TN_fs_df + TN_fb_df
    TP_f_df = TP_fs_df + TP_fb_df
    # from system unit
    TN_s_df = TN_out_df.multiply(Qdp_df, fill_value=0.0) / 100
    TP_s_df = TP_out_df.multiply(Qdp_df, fill_value=0.0) / 100
    TN_sys_df = TN_s_df + TN_b_df
    TP_sys_df = TP_s_df + TP_b_df
    del TN_cfb_df
    del TP_cfb_df
    # for water footprint calculation
    TN_s_df_4wf = TN_s_df.where(TN_out_df > 2.0, 0.0)
    TP_s_df_4wf = TP_s_df.where(TP_out_df > 0.4, 0.0)
    TN_fs_df_4wf = TN_fs_df.where(TN_cf_df > 2.0, 0.0)
    TP_fs_df_4wf = TP_fs_df.where(TP_cf_df > 0.4, 0.0)
    TN_fb_lat_df_4wf = TN_fb_lat_df.where(TN_cfb_lat_df > 2.0, 0.0)
    TP_fb_lat_df_4wf = TP_fb_lat_df.where(TP_cfb_lat_df > 0.4, 0.0)
    TN_fs_A_4wf = (TN_fs_df_4wf + TN_fb_lat_df_4wf).resample("A", kind="period").sum()
    TP_fs_A_4wf = (TP_fs_df_4wf + TP_fb_lat_df_4wf).resample("A", kind="period").sum()
    TN_s_A_4wf = TN_s_df_4wf.resample("A", kind="period").sum()
    TP_s_A_4wf = TP_s_df_4wf.resample("A", kind="period").sum()
    TN_b_A_4wf = TN_b_df_4wf.resample("A", kind="period").sum()
    loads_4gwf = pd.DataFrame({"TN_runoff": TN_s_A_4wf.values.reshape(-1),
                               "TP_runoff": TP_s_A_4wf.values.reshape(-1),
                               "TN_leaching": TN_b_A_4wf.values.reshape(-1),
                               "TN_fs": TN_fs_A_4wf.values.reshape(-1),
                               "TP_fs": TP_fs_A_4wf.values.reshape(-1)})
    del TN_cfb_lat_df
    del TP_cfb_lat_df
    """ Calculate the proportions of surface and subsurface loads"""
    # from system unit
    TN_s_A = TN_s_df.resample('A', kind='period').sum()
    TN_b_A = TN_b_df.resample('A', kind='period').sum()
    TN_sys_A = TN_sys_df.resample('A', kind='period').sum()
    TP_s_A = TP_s_df.resample('A', kind='period').sum()
    TP_b_A = TP_b_df.resample('A', kind='period').sum()
    TP_sys_A = TP_sys_df.resample('A', kind='period').sum()
    TN_s_pr = TN_s_A / TN_sys_A
    TN_b_pr = TN_b_A / TN_sys_A
    TP_s_pr = TP_s_A / TP_sys_A
    TP_b_pr = TP_b_A / TP_sys_A
    result_sys_pr_df = pd.DataFrame({'TN_load(kg/ha/yr)': TN_sys_A.values.reshape(-1),
                                     'TN_surface(kg/ha/yr)': TN_s_A.values.reshape(-1),
                                     'TN_surface(rate)': TN_s_pr.values.reshape(-1),
                                     'TN_subsurface(kg/ha/yr)': TN_b_A.values.reshape(-1),
                                     'TN_subsurface(rate)': TN_b_pr.values.reshape(-1),
                                     'TP_load(kg/ha/yr)': TP_sys_A.values.reshape(-1),
                                     'TP_surface(kg/ha/yr)': TP_s_A.values.reshape(-1),
                                     'TP_surface(rate)': TP_s_pr.values.reshape(-1),
                                     'TP_subsurface(kg/ha/yr)': TP_b_A.values.reshape(-1),
                                     'TP_subsurface(rate)': TP_b_pr.values.reshape(-1)})
    result_sys_pr = result_sys_pr_df.describe()
    # from fields
    TN_fs_A = TN_fs_df.resample('A', kind='period').sum()
    TN_fb_A = TN_fb_df.resample('A', kind='period').sum()
    TN_fb_lat_A = TN_fb_lat_df.resample('A', kind='period').sum()
    TN_f_A = TN_f_df.resample('A', kind='period').sum()
    TP_fs_A = TP_fs_df.resample('A', kind='period').sum()
    TP_fb_A = TP_fb_df.resample('A', kind='period').sum()
    TP_fb_lat_A = TP_fb_lat_df.resample('A', kind='period').sum()
    TP_f_A = TP_f_df.resample('A', kind='period').sum()
    TN_fs_pr = TN_fs_A / TN_f_A
    TN_fb_pr = TN_fb_A / TN_f_A
    TP_fs_pr = TP_fs_A / TP_f_A
    TP_fb_pr = TP_fb_A / TP_f_A
    result_f_pr_df = pd.DataFrame({'TN_load(kg/ha/yr)': TN_f_A.values.reshape(-1),
                                   'TN_surface(kg/ha/yr)': TN_fs_A.values.reshape(-1),
                                   'TN_surface(rate)': TN_fs_pr.values.reshape(-1),
                                   'TN_subsurface(kg/ha/yr)': TN_fb_A.values.reshape(-1),
                                   'TN_subsurface(rate)': TN_fb_pr.values.reshape(-1),
                                   'TP_load(kg/ha/yr)': TP_f_A.values.reshape(-1),
                                   'TP_surface(kg/ha/yr)': TP_fs_A.values.reshape(-1),
                                   'TP_surface(rate)': TP_fs_pr.values.reshape(-1),
                                   'TP_subsurface(kg/ha/yr)': TP_fb_A.values.reshape(-1),
                                   'TP_subsurface(rate)': TP_fb_pr.values.reshape(-1)})
    result_f_pr = result_f_pr_df.describe()
    """ Calculate the annual average flow-weighted concentrations, in mg/L """
    # Concentrations in surface flow from fields
    TN_cs_f = TN_fs_A * 100 / (D_df.resample('A', kind='period').sum())
    TP_cs_f = TP_fs_A * 100 / (D_df.resample('A', kind='period').sum())
    # Concentrations in surface flow from system unit outlet
    TN_cs_sys = TN_s_A * 100 / (Qdp_df.resample('A', kind='period').sum())
    TP_cs_sys = TP_s_A * 100 / (Qdp_df.resample('A', kind='period').sum())

    """ Calculate the total nutrient inputs to and outputs from the field-ditch-pond system unit.
        The inputs are from irrigation and precipitation, the outputs are as surface runoff and leaching."""
    irrg_A = I_df.resample('A', kind='period').sum()
    prcp_A = prcp_d.resample('A', kind='period').sum()
    TN_I_out_A = TN_I_out_df.resample('A', kind='period').sum()
    TP_I_out_A = TP_I_out_df.resample('A', kind='period').sum()
    # field scale
    TN_I_pond_A = TN_I_pond_df.resample("A", kind='period').sum() / 100  # in kg/ha/yr
    TP_I_pond_A = TP_I_pond_df.resample('A', kind='period').sum() / 100  # in kg/ha/yr
    irrg_TN_input_f = TN_I_out_A / 100 + TN_I_pond_A  # in kg/ha/yr
    irrg_TP_input_f = TP_I_out_A / 100 + TP_I_pond_A  # in kg/ha/yr
    prcp_TN_input_f = prcp_A * TN_P / 100  # in kg/ha/yr
    prcp_TP_input_f = prcp_A * TP_P / 100  # in kg/ha/yr
    # IDU scale
    irrg_TN_input_sys = TN_I_out_A / 100 * area_f / (area_f + area_pd)  # in kg/ha/yr
    irrg_TP_input_sys = TP_I_out_A / 100 * area_f / (area_f + area_pd)  # in kg/ha/yr
    prcp_TN_input_sys = prcp_A * TN_P / 100  # in kg/ha/yr
    prcp_TP_input_sys = prcp_A * TP_P / 100  # in kg/ha/yr
    # Calculate the net nutrient export from fields and from the system unit, respectively.
    TN_export_f = TN_f_A - irrg_TN_input_f - prcp_TN_input_f
    TP_export_f = TP_f_A - irrg_TP_input_f - prcp_TP_input_f
    TN_export_sys = TN_sys_A - irrg_TN_input_sys - prcp_TN_input_sys
    TP_export_sys = TP_sys_A - irrg_TP_input_sys - prcp_TP_input_sys
    # The input-output dataframes
    TN_in_out = pd.concat([pd.DataFrame({"irrigation input": irrg_TN_input_f.values.reshape(-1),
                                         "precipitation input": prcp_TN_input_f.values.reshape(-1),
                                         "runoff loss": TN_fs_A.values.reshape(-1),
                                         "leaching loss": TN_fb_A.values.reshape(-1),
                                         "net export": TN_export_f.values.reshape(-1),
                                         "scale": "field"}),
                           pd.DataFrame({"irrigation input": irrg_TN_input_sys.values.reshape(-1),
                                         "precipitation input": prcp_TN_input_sys.values.reshape(-1),
                                         "runoff loss": TN_s_A.values.reshape(-1),
                                         "leaching loss": TN_b_A.values.reshape(-1),
                                         "net export": TN_export_sys.values.reshape(-1),
                                         "scale": "IDU"})],
                          ignore_index=True)
    TP_in_out = pd.concat([pd.DataFrame({"irrigation input": irrg_TP_input_f.values.reshape(-1),
                                         "precipitation input": prcp_TP_input_f.values.reshape(-1),
                                         "runoff loss": TP_fs_A.values.reshape(-1),
                                         "leaching loss": TP_fb_A.values.reshape(-1),
                                         "net export": TP_export_f.values.reshape(-1),
                                         "scale": "field"}),
                           pd.DataFrame({"irrigation input": irrg_TP_input_sys.values.reshape(-1),
                                         "precipitation input": prcp_TP_input_sys.values.reshape(-1),
                                         "runoff loss": TP_s_A.values.reshape(-1),
                                         "leaching loss": TP_b_A.values.reshape(-1),
                                         "net export": TP_export_sys.values.reshape(-1),
                                         "scale": "IDU"})],
                          ignore_index=True)
    # Calculate the export reduction by ditches and ponds
    TN_reduction = (TN_fs_A * area_f - TN_s_A * (area_f + area_pd)) / (TN_fs_A * area_f) * 100  # in %
    TP_reduction = (TP_fs_A * area_f - TP_s_A * (area_f + area_pd)) / (TP_fs_A * area_f) * 100  # in %
    TN_reduction = TN_reduction.where(TN_reduction > 0, 0)
    TN_reduction = TN_reduction.where(TN_reduction < 100, 100)
    TP_reduction = TP_reduction.where(TP_reduction > 0, 0)
    TP_reduction = TP_reduction.where(TP_reduction < 100, 100)
    result_export_df = pd.DataFrame({'TN_reduction(%)': TN_reduction.values.reshape(-1),
                                     'TP_reduction(%)': TP_reduction.values.reshape(-1)})
    result_export = result_export_df.describe()
    """ Calculate the percentage of irrigated water from pond to the total irrigated water content"""
    I_pond_A = I_pond_df.resample('A', kind='period').sum()
    I_pond_rate = (I_pond_A / irrg_A).values.reshape(-1)
    I_pond_rate_mean = I_pond_rate.mean() * 100
    I_pond_rate_min = I_pond_rate.min() * 100
    I_pond_rate_max = I_pond_rate.max() * 100
    I_out_A = irrg_A - I_pond_A
    # Calculating the rate of nutrient recycling (RNRc) and rate of water resource recycling (RWRc)
    RNRc_TN = TN_I_pond_A / (TN_fs_A + TN_fb_lat_A)
    RNRc_TP = TP_I_pond_A / (TP_fs_A + TP_fb_lat_A)
    RNRc_TN["nutrient"] = "TN"
    RNRc_TP["nutrient"] = "TP"
    RNRc = pd.concat([RNRc_TN, RNRc_TP], axis=0)
    water_f_A = (D_df + F_lat_df).resample("A", kind='period').sum()
    RWRc = I_pond_A / water_f_A

    # Prepare data for bmp comparison
    comp_data = {"TN_load": result_sys_pr.loc["mean", "TN_load(kg/ha/yr)"],
                 "TP_load": result_sys_pr.loc["mean", "TP_load(kg/ha/yr)"],
                 "TN_conc": round(np.nanmean(TN_cs_sys.values), 3),
                 "TP_conc": round(np.nanmean(TP_cs_sys.values), 3)}

    """ Write the results into text files """
    # Write daily irrigation into files
    with open(output_dir + '/I_d.txt', mode='w') as output_f:
        output_f.write(
            'The simulated irrigation (mm/d) to paddy fields are below:\n')
        output_f.close()
    I_df.to_csv(output_dir + '/I_d.txt', mode='a', header=True, index=True, sep='\t',
                float_format='%.4f')
    with open(output_dir + '/I_pond_d.txt', mode='w') as output_f:
        output_f.write(
            'The simulated irrigation (mm/d) from local ponds to paddy fields are below:\n')
        output_f.close()
    I_pond_df.to_csv(output_dir + '/I_pond_d.txt', mode='a', header=True, index=True, sep='\t',
                float_format='%.4f')
    # Write runoff and leaching loads for water footprint calculation
    with open(output_dir + '/loads_4gwf{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write('The simulated nutrient loads (kg/ha) from IDUs for water footprint calculation are below:\n')
        output_f.close()
    loads_4gwf.to_csv(output_dir + '/loads_4gwf{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                      float_format='%.4f')
    # Write ET_d into files
    with open(output_dir + '/ET_d.txt', mode='w') as output_f:
        output_f.write(
            'The simulated ET (mm/d) from paddy fields are below:\n')
        output_f.close()
    ET_df.to_csv(output_dir + '/ET_d.txt', mode='a', header=True, index=True, sep='\t',
                 float_format='%.4f')
    # Write output summary file
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write('The simulated nutrient loads from the fields are summarized as below:\n')
        output_f.close()
    result_f_pr.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                       float_format='%.2f')
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
        output_f.write('\nThe simulated nutrient loads from the system unit are summarized as below:\n')
        output_f.close()
    result_sys_pr.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                         float_format='%.2f')
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
        output_f.write('\nThe reduction rate by ditches and ponds are summarized as below:\n')
        output_f.close()
    result_export.to_csv(output_dir + '/output_summary{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                         float_format='%.2f')
    with open(output_dir + '/output_summary{}.txt'.format(run_id), mode='a') as output_f:
        output_f.write('\nThe simulated percentage of irrigated water from pond is:\n')
        output_f.write('Mean (Range)\n')
        output_f.write('{0:.1f}% ({1:.1f}% - {2:.1f}%)'.format(I_pond_rate_mean, I_pond_rate_min, I_pond_rate_max))
        output_f.close()

    # Write average concentrations
    with open(output_dir + '/TN_cs_sys{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write(
            'The simulated flow-weighted annual average TN concentrations (mg/L) from system unit outlet are:\n')
        output_f.close()
    TN_cs_sys.to_csv(output_dir + '/TN_cs_sys{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                     float_format='%.3f')
    with open(output_dir + '/TP_cs_sys{}.txt'.format(run_id), mode='w') as output_f:
        output_f.write(
            'The simulated flow-weighted annual average TP concentrations (mg/L) from system unit outlet are:\n')
        output_f.close()
    TP_cs_sys.to_csv(output_dir + '/TP_cs_sys{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                     float_format='%.3f')

    # Irrigated water content
    with open(output_dir + '/I_all{}.txt'.format(run_id), 'w') as f:
        f.write("The irrigated water content is simulated as below, in mm.\n")
        f.close()
    irrg_A.to_csv(output_dir + '/I_all{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    with open(output_dir + '/I_out{}.txt'.format(run_id), 'w') as f:
        f.write("The irrigated water content from out of the system is simulated as below, in mm.\n")
        f.close()
    I_out_A.to_csv(output_dir + '/I_out{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    # RNRc and RWRc
    with open(output_dir + '/RNRc{}.txt'.format(run_id), 'w') as f:
        f.write("The rate of nutrient recycling is simulated as below, unitless.\n")
        f.close()
    RNRc.to_csv(output_dir + '/RNRc{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    with open(output_dir + '/RWRc{}.txt'.format(run_id), 'w') as f:
        f.write("The rate of water resource recycling is simulated as below, unitless.\n")
        f.close()
    RWRc.to_csv(output_dir + '/RWRc{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    # precipitation water content
    with open(output_dir + '/prcp_A.txt', 'w') as f:
        f.write("The precipitation water content during the growing season is simulated as below, in mm.\n")
        f.close()
    prcp_A.to_csv(output_dir + '/prcp_A.txt', mode='a', header=True, index=True, sep='\t', float_format='%.2f')
    # Annual load from system unit
    with open(output_dir + '/TN_load_sysA{}.txt'.format(run_id), 'w') as output_f:
        output_f.write("The annual TN loads (kg/ha) from the system unit are simulated as:\n")
        output_f.close()
    TN_sys_A.to_csv(output_dir + '/TN_load_sysA{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                    float_format='%.3f')
    with open(output_dir + '/TP_load_sysA{}.txt'.format(run_id), 'w') as output_f:
        output_f.write("The annual TP loads (kg/ha) from the system unit are simulated as:\n")
        output_f.close()
    TP_sys_A.to_csv(output_dir + '/TP_load_sysA{}.txt'.format(run_id), mode='a', header=True, index=True, sep='\t',
                    float_format='%.3f')
    # TN reduction percentages
    TN_reduction.to_csv(output_dir + '/TN_reduction_rate{}.txt'.format(run_id), header=True, index=True, sep='\t',
                        float_format='%.2f')
    # TP reduction percentages
    TP_reduction.to_csv(output_dir + '/TP_reduction_rate{}.txt'.format(run_id), header=True, index=True, sep='\t',
                        float_format='%.2f')
    # Nutrient input output results.
    with open(output_dir + '/TN_in_out{}.txt'.format(run_id), mode='w') as f:
        f.write("The TN flux to and from the field V.S. IDU scale is as below, in kg/ha. \n")
        f.close()
    TN_in_out.to_csv(output_dir + '/TN_in_out{}.txt'.format(run_id), mode='a', header=True, index=False, sep='\t',
                     float_format='%.4f')
    with open(output_dir + '/TP_in_out{}.txt'.format(run_id), mode='w') as f:
        f.write("The TP flux to and from the field V.S. IDU scale is as below, in kg/ha. \n")
        f.close()
    TP_in_out.to_csv(output_dir + '/TP_in_out{}.txt'.format(run_id), mode='a', header=True, index=False, sep='\t',
                     float_format='%.4f')

    """ Prepare results summary for GUI """
    output_text = f"""||模拟结果小结||
                    \n计算时期内，水稻季的年平均模拟结果如下：
                    \n（1）田块尺度\n"""
    output_text += "TN年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_f_pr.loc["mean", "TN_load(kg/ha/yr)"], result_f_pr.loc["mean", "TN_surface(rate)"] * 100,
                                                      result_f_pr.loc["mean", "TN_subsurface(rate)"] * 100)
    output_text += "TP年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_f_pr.loc["mean", "TP_load(kg/ha/yr)"], result_f_pr.loc["mean", "TP_surface(rate)"] * 100,
                                                      result_f_pr.loc["mean", "TP_subsurface(rate)"] * 100)
    output_text += "地表径流中TN年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TN_cs_f.values))
    output_text += "地表径流中TP年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TP_cs_f.values))
    output_text += "\n（2）灌排单元尺度\n"
    output_text += "TN年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_sys_pr.loc["mean", "TN_load(kg/ha/yr)"], result_sys_pr.loc["mean", "TN_surface(rate)"] * 100,
                                                        result_sys_pr.loc["mean", "TN_subsurface(rate)"] * 100)
    output_text += "TP年均流失负荷：{:.2f} kg/ha，其中随地表径流流失的占{:.1f}%，随地下淋溶流失的占{:.1f}%。\n".format(
        result_sys_pr.loc["mean", "TP_load(kg/ha/yr)"], result_sys_pr.loc["mean", "TP_surface(rate)"] * 100,
                                                        result_sys_pr.loc["mean", "TP_subsurface(rate)"] * 100)
    output_text += "地表径流中TN年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TN_cs_sys.values))
    output_text += "地表径流中TP年均浓度：{:.2f} mg/L。\n".format(np.nanmean(TP_cs_sys.values))

    if graph == 1:
        """ Convert outputs to time series """
        yearStart = prcp_d.index[0].year
        yearEnd = prcp_d.index[-1].year + 1
        Hdp_df = Hdp_df - HDiff_FD1
        for year in range(yearStart, yearEnd):
            Hf_df1[year] = Hf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            Hdp_df1[year] = Hdp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_cf_df1[year] = TP_cf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_cd1_df1[year] = TP_cd1_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_cf_df1[year] = TN_cf_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_cd1_df1[year] = TN_cd1_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            if havePond:
                TP_cp_df1[year] = TP_cp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
                TN_cp_df1[year] = TN_cp_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            if orderDitch == 2:
                TP_cd2_df1[year] = TP_cd2_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
                TN_cd2_df1[year] = TN_cd2_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_fs_df1[year] = TN_fs_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TN_s_df1[year] = TN_s_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_fs_df1[year] = TP_fs_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values
            TP_s_df1[year] = TP_s_df.loc[str(year) + '-' + startDay:str(year) + '-' + endDay].values

        """Write Time series outputs """
        # water level in fields
        Hf_df1.to_csv(output_dir + '/H_f_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # water level in ditches and pond
        Hdp_df1.to_csv(output_dir + '/H_dp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN concentration in fields
        TN_cf_df1.to_csv(output_dir + '/TN_cf_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP concentration in fields
        TP_cf_df1.to_csv(output_dir + '/TP_cf_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN concentration in first-order ditch
        TN_cd1_df1.to_csv(output_dir + '/TN_cd1_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP concentration in first-order ditch
        TP_cd1_df1.to_csv(output_dir + '/TP_cd1_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        if orderDitch == 2:
            # TN concentration in second-order ditch
            TN_cd2_df1.to_csv(output_dir + '/TN_cd2_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
            # TP concentration in second-order ditch
            TP_cd2_df1.to_csv(output_dir + '/TP_cd2_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        if havePond:
            # TN concentration in pond
            TN_cp_df1.to_csv(output_dir + '/TN_cp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
            # TP concentration in pond
            TP_cp_df1.to_csv(output_dir + '/TP_cp_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN load from field with surface runoff
        TN_fs_df1.to_csv(output_dir + '/TN_load_fs_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TN load from system with surface runoff
        TN_s_df1.to_csv(output_dir + '/TN_load_syss_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP load from field with surface runoff
        TP_fs_df1.to_csv(output_dir + '/TP_load_fs_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')
        # TP load from system with surface runoff
        TP_s_df1.to_csv(output_dir + '/TP_load_syss_ts.txt', header=True, index=True, sep='\t', float_format='%.2f')

        """ Prepare dataframe list for drawing graphs """
        if orderDitch == 1:
            TN_cd2_df1 = 0
            TP_cd2_df1 = 0
        if not havePond:
            TN_cp_df1 = 0
            TP_cp_df1 = 0
        graph_dfs = list([Hf_df1, Hdp_df1, TN_cf_df1, TP_cf_df1, TN_cd1_df1, TP_cd1_df1, TN_cd2_df1, TP_cd2_df1,
                          TN_cp_df1, TP_cp_df1, TN_fs_df1, TN_s_df1, TP_fs_df1, TP_s_df1, TN_in_out, TP_in_out])

    if graph == 1:
        return output_text, comp_data, graph_dfs
    else:
        return output_text, comp_data

