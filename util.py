"""
Utility functions and definitions.
"""

import numpy as np
import pandas as pd

# Definitions.
pos_group_colors = {
    'Position Group A': 'crimson',
    'Position Group B': 'slategray',
    'Position Group C': 'yellow',
    'Position Group D': 'rebeccapurple',
}

var_groups = {
    'load': [
        'Load: Practice',
        'Load: S&C',
        'Load: Competition',
    ],
    'hr': [
        'Practice HR Avg',
        'Practice HR Max',
    ],
    'accel_decel_sprint': [
        'Accelerations',
        'Decelerations',
        'Sprints'
    ],
    'distance': [
        'Distance',
    ],
    'distance_high_speed':
    [
        'High Speed Distance',
    ],
    'survey': [
        'Fatigue',
        'Mood',
        'Motivation',
        'Soreness',
        'Stress',
        'Sleep',
    ]
}


def preprocess_data(df):
    """
    Preprocess the load and wellness data.
    """
    print((df['Practice HR Avg'] == 0).sum())
    df_out = df.replace({
        var: {0: np.nan}
        for var in var_groups['hr']
    })

    # Certain variables that are left-skewed
    # would be more easily modeled if they were
    # right-skewed.
    df_out['100-Mood'] = 100.0 - df_out['Mood']
    df_out['100-Motivation'] = 100.0 - df_out['Motivation']
    df_out['100-Sleep'] = 100.0 - df_out['Sleep']

    print((df_out['Practice HR Avg'] == 0).sum())
    return df_out

