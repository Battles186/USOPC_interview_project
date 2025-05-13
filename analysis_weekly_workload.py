"""
This module contains analyses pertaining to weekly workload.
"""

# Data management.
import numpy as np
import pandas as pd

# Data preparation.
from sklearn.preprocessing import RobustScaler

# Models.
from sklearn.linear_model import LinearRegression, GammaRegressor, BayesianRidge, RANSACRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

# Model selection.
from sklearn.model_selection import train_test_split, GridSearchCV

# Transformers.
from sklearn.preprocessing import FunctionTransformer

# Pipeline.
from sklearn.pipeline import Pipeline

# Scoring.
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import boxcox

# Visualization.
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities.
from util import preprocess_data
from copy import deepcopy
from pprint import pprint
import os
import json

cols_mean = [
    'Fatigue',
    'Mood',
    'Motivation',
    'Soreness',
    'Stress',
    'Sleep',
    '100-Sleep',
    '100-Mood',
    '100-Motivation',
]

cols_sum = [
    'Load: Practice',
    'Load: S&C',
    'Load: Competition',
    'Practice HR Avg',
    'Practice HR Max',
    'Distance',
    'High Speed Distance',
    'Accelerations',
    'Decelerations',
    'Sprints',
]


def bin_data_athlete_week(
    df, athlete,
    cols_mean: list = None, cols_sum: list = None
    ):
    """
    Finds all data for an athlete and bins it such that values are aggregated across
    each week. Hence, each output row is an athlete-week.

    df: the DataFrame containing athlete load and wellness data.
    """
    # Get all the data pertaining to the athlete.
    df_athlete = df[df['Athlete'] == athlete]

    # Aggregate data on a weekly basis, computing means for one subset of
    # variables and sums for another subset.
    df_athlete_mean = df_athlete.groupby('week')[cols_mean].mean()
    df_athlete_sum = df_athlete.groupby('week')[cols_sum].sum()

    # Join the two DataFrames, make sure that we keep track of who
    # the athlete is in the output, and then reorganize the columns.
    df_out = pd.concat([df_athlete_mean, df_athlete_sum], axis=1)
    df_out['Athlete'] = athlete
    df_out = df_out[['Athlete'] + cols_mean + cols_sum]

    # Compute change in each variable.
    df_out_diff: pd.DataFrame = df_out[cols_mean + cols_sum].diff()
    df_out_diff.rename(mapper={
        col: col + '_delta'
        for col in df_out_diff.columns
    }, axis=1, inplace=True)

    df_out = pd.concat([
        df_out,
        df_out_diff
    ], axis=1)

    return df_out


def fit_grid_search(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    predictors: list[str],
    outcome: str,
    gs: GridSearchCV,
    path_out: str,
    ):
    """
    Fits a grid search to the training data and reports performance on
    both training and testing sets.

    df_train: DataFrame containing training examples.

    df_test: DataFrame containing testing examples.

    predictors: list of predictor variables.

    outcome: outcome variable.

    model: model to fit to the data.
    """
    print()
    print(f"Fit going to {path_out}")

    # Ensure we have no nan input.
    df_train_clean = df_train[predictors + [outcome]].dropna()
    df_test_clean = df_test[predictors + [outcome]].dropna()

    print(f"{df_train_clean.shape[0]} training examples")
    print(f"{df_test_clean.shape[0]} testing examples")

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    df_train_clean.to_csv(os.path.join(path_out, 'data_train.csv'))
    df_test_clean.to_csv(os.path.join(path_out, 'data_test.csv'))
    return

    # Separate data.
    X_train = df_train_clean[predictors]
    y_train = df_train_clean[outcome]

    X_test = df_test_clean[predictors]
    y_test = df_test_clean[outcome]

    

    # print("y Values:")
    # print(y_train)
    # print(y_test)

    # Fit model to training data.
    gs.fit(X_train, y_train)

    # Extract crossfold validation results.
    df_cv_results = pd.DataFrame(gs.cv_results_).sort_values(by='rank_test_score')

    # Determine performance on training and testing sets.
    y_pred_train = gs.predict(X_train)
    y_pred_test = gs.predict(X_test)

    # Error and fit quality.
    rmse_train = root_mean_squared_error(y_pred=y_pred_train, y_true=y_train)
    rmse_test = root_mean_squared_error(y_pred=y_pred_test, y_true=y_test)
    r_test = pearsonr(y_pred_test, y_test).statistic

    # Figure and other output.
    if path_out is not None:
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        plt.scatter(y_test, y_pred_test)
        plt.xlabel(f'{outcome.replace("_", " ")}, True')
        plt.ylabel(f'{outcome.replace("_", " ")}, Predicted')
        plt.savefig(os.path.join(path_out, 'true_pred_test.png'))
        plt.close('all')

        coef_ = gs.best_estimator_['estimator'].coef_
        sns.barplot(x='', )

        # Write CV results to disk.
        df_cv_results.to_csv(os.path.join(path_out, 'cv_results.csv'))

        # Write fit information to disk.
        with open(os.path.join(path_out, 'fit_quality.json'), 'w') as f:
            f.write(json.dumps(
                dict(
                    experiment=path_out.split(os.sep)[-1],
                    rmse_train=rmse_train,
                    rmse_test=rmse_test,
                    r_test=r_test,
                    best_model=str(df_cv_results['param_estimator'].values[0]),
                    best_model_parameters=str(df_cv_results['params'].values[0]),
                )
            ))

        # Write predictor

    out = dict(
        gs_fit=gs,
        df_cv_results=df_cv_results,
        error_train=rmse_train,
        error_test=rmse_test,
        r_test=r_test,
    )

    return out


# Load and preprocess data.
df = pd.read_excel('LoadandWellnessData.xlsx')
df = preprocess_data(df)

print(df.columns)

# Calculate the week in which each data point was observed.
df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].apply(
    func=lambda date: int((date - df['Date'].min()).days / 7)
)

# Get numeric data.
df_num = df.select_dtypes(include=np.number)
athlete_week_columns = ['Athlete', 'Position Group', 'week'] + list(df_num.columns)[:-1]

# Bin the data on a weekly basis so that we have profiles for each
# athlete in each week.
df_athlete_week = pd.concat([
    bin_data_athlete_week(
        df=df,
        athlete=athlete,
        cols_mean=cols_mean,
        cols_sum=cols_sum,
    )
    for athlete in df['Athlete'].unique()
])

print(df_athlete_week)
print(df_athlete_week.columns)

# Split data.
df_train, df_test = train_test_split(
    df_athlete_week,
    test_size=0.30,
    random_state=42,
)

# We want to model each of our outcomes as a function of
# our predictors. Specifically, we're interested in how
# a given workload alters our outcome variables of interest.

# We need our DataFrame to not contain any NaN values in
# columns we are interested in.

# Define pipeline and parameters.
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('estimator', LinearRegression()),
])

gs_params = {
    'estimator': [
        LinearRegression(),
        BayesianRidge(),
        SVR(kernel='linear'),
    ]
}

# Define grid search.
gs = GridSearchCV(
    estimator=pipeline,
    param_grid=gs_params,
    n_jobs=-1,
    error_score='raise',
)

model_fit_params = {

    # Workload and fatigue.
    'workload_sleep_fatigue': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Sleep',
        ],
        'outcome': 'Fatigue',
    },
    'workload_sleep_fatigue_delta': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Sleep',
        ],
        'outcome': 'Fatigue_delta',
    },

    # Workload and motivation.
    'workload_sleep_motivation': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Sleep',
        ],
        'outcome': 'Motivation',
    },
    'workload_sleep_motivation_delta': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Sleep',
        ],
        'outcome': 'Motivation_delta',
    },

    # Workload and mood.
    'workload_sleep_mood': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Sleep',
        ],
        'outcome': 'Mood',
    },
    'workload_sleep_mood_delta': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Sleep',
        ],
        'outcome': 'Mood_delta',
    },

    # Workload and stress.
    'workload_sleep_stress': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Sleep',
        ],
        'outcome': 'Stress',
    },
    'workload_sleep_stress_delta': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Sleep',
        ],
        'outcome': 'Stress_delta',
    },

    # Let's look at how our workloads affect sleep.
    'workload_sleep': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
        ],
        'outcome': '100-Sleep',
    },

    # Let's see how our distances affect our soreness.
    'workload_distance_sleep_soreness': {
        'predictors': [
            'Load: Practice',
            'Load: S&C',
            'Load: Competition',
            'Distance',
            'High Speed Distance',
            'Sprints',
            'Sleep'
        ],
        'outcome': 'Soreness'
    },
}

# rbscaler = RobustScaler()
# df_train_nonan = df_train.dropna()
# 
# gamma_reg = Pipeline([
#     ('rb', RobustScaler()),
#     ('gamma', GammaRegressor())
# ])
# gamma_reg.fit(df_train_nonan[['Load: Competition', 'Sleep']], df_train_nonan['Fatigue'] + 0.5)
# print("Fit succeeded")
# print(gamma_reg)
# quit()

fit_results = {
    fit_name: fit_grid_search(
        df_train=df_train,
        df_test=df_test,
        gs=deepcopy(gs),
        path_out=os.path.join('out', 'analysis_weekly_workload', fit_name),
        **fit_params
    )
    for fit_name, fit_params in model_fit_params.items()
}

# pprint(fit_results)

