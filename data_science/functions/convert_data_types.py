import numpy as np
import pandas as pd

def convert_data_types(df):
  data_types = {
        'boolean': {
            'sex_female',
            'skin_transplanted',
            'flap_loss',
            'wound_infection',
            'nonunion',
            'tmj_luxation'
        },
        'category': {
            'indication',
            'prior_flap',
            'flap_donor_site',
            'long_plate_thickness',
            'tmj_replacement_type',
            'flap_segment_count'
        },
        'string': {
            'which_autoimmune_disease',
            'which_bleeding_disorder',
        },
        'UInt8': {
            'age_surgery_years',
            'height_cm',
            'weight_kg'
        },
        'UInt16': {
            'surgery_duration_min'
        },
        'Float32': {
            'bmi'
        }
    }

    for column in df.columns:
        if column.startswith('days_to_'):
            df[column] = df[column].astype('UInt16')
        elif "___" in column:
                df[column] = df[column].astype('boolean')
        elif column in data_types['boolean']:         
            df[column] = np.where(df[column] == 'True', True, np.where(df[column] == 'False', False, df[column]))
            df[column] = df[column].astype('boolean')
        else:
            for data_type in ['category', 'string', 'UInt8', 'UInt16', 'Float32']:
                if column in data_types[data_type]:
                    df[column] = df[column].astype(data_type)

    return df
  