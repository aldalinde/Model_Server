import pandas as pd
import numpy as np
import xgboost as xgb


def gender_num(gender):
    dict_gender = {'Male': 0, 'Female': 1}
    return dict_gender.get(gender)


def matrim_num(status):
    dict_status = {'Alone': 0, 'Other': 1}
    return dict_status.get(status)


def num_usage_prof(veh_usage):
    if veh_usage == 'Professional':
        v_usage_prof = 1
    else:
        v_usage_prof = 0
    return v_usage_prof

def num_usage_priv_trip(veh_usage):
    if veh_usage == 'Private+trip to office':
        v_usage_prof_trip = 1
    else:
        v_usage_prof_trip = 0
    return v_usage_prof_trip


def num_usage_private(veh_usage):
    if veh_usage == 'Private':
        v_usage_private = 1
    else:
        v_usage_private = 0
    return v_usage_private


def num_usage_prof_run(veh_usage):
    if veh_usage == 'Professional run':
        v_usage_prof_run = 1
    else:
        v_usage_prof_run = 0
    return v_usage_prof_run


def socio_categ(socio_cat):
    #soc_cat_1= soc_cat_2= soc_cat_3=soc_cat_4= soc_cat_5= soc_cat_6= soc_cat_7 = [[0 for _ in range(len(socio_cat))]]
    soc_cat_1 = soc_cat_2 = soc_cat_3 = soc_cat_4 = soc_cat_5 = soc_cat_6 = soc_cat_7 = 0
    #i = 0

    if socio_cat['SocioCateg'][3] == '1':
        soc_cat_1 = 1
    elif socio_cat['SocioCateg'][3] == '2':
        soc_cat_2 = 1
    elif socio_cat['SocioCateg'][3] == '3':
        soc_cat_3 = 1
    elif socio_cat['SocioCateg'][3] == '4':
        soc_cat_4 = 1
    elif socio_cat['SocioCateg'][3] == '5':
        soc_cat_5 = 1
    elif socio_cat['SocioCateg'][3] == '6':
        soc_cat_6 = 1
    elif socio_cat['SocioCateg'][3] == '7':
        soc_cat_7 = 1

        #i += 1
    return soc_cat_1, soc_cat_2, soc_cat_3, soc_cat_4, soc_cat_5, soc_cat_6, soc_cat_7

def driv_age_sq(driv_age):
    return driv_age**2


def process_input(json_input):
    lic_age = int(json_input['LicAge'])
    gender = gender_num(json_input['Gender'])
    matrim_stat = matrim_num(json_input['MariStat'])
    driv_age = int(json_input['DrivAge'])
    km_limit = int(json_input['HasKmLimit'])
    bonus_mal = int(json_input['BonusMalus'])
    out_use_nb = int(json_input['OutUseNb'])
    risk_area = int(json_input['RiskArea'])
    veh_use_private = num_usage_private(json_input['VehUsage'])
    veh_use_priv_trip = num_usage_priv_trip(json_input['VehUsage'])
    veh_use_prof = num_usage_prof(json_input['VehUsage'])
    veh_use_prof_run = num_usage_prof_run(json_input['VehUsage'])
    soc_cat_1, soc_cat_2, soc_cat_3, soc_cat_4, soc_cat_5, soc_cat_6, soc_cat_7 = socio_categ(json_input)
    driv_age_square = driv_age_sq(json_input['DrivAge'])



    arr = np.array([[lic_age,  gender,  matrim_stat,driv_age, km_limit, bonus_mal, out_use_nb, risk_area,
                   veh_use_private, veh_use_priv_trip, veh_use_prof, veh_use_prof_run, soc_cat_1,
                   soc_cat_2, soc_cat_3, soc_cat_4, soc_cat_5, soc_cat_6, soc_cat_7, driv_age_square]])

    return arr


def get_dmatrix(json):

    df = process_input(json)
    dmatrix_df = xgb.DMatrix(df)

    return dmatrix_df






