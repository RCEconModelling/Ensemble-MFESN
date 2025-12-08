"""
Ensemble MFESN Medium MD

Authors:    andrew-li-yc@github
            giob1994@github
"""

# %load_ext autoreload
# %autoreload 2

import os
from pathlib import Path
import pickle
import warnings

# Reduce depracation warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from statsmodels.tsa.ar_model import AutoReg, OLS

from rich import box
from rich.console import Console
from rich.table import Table

# Load custom toolboxes
from newToolbox_ESN import ESN, stateMatrixGenerator
from newToolbox_ESN_Multi import ESNMultiFrequency

# Load custom functions
from data_funcs import *
from comb_multistep_funcs import *
from esn_multistep_ensemble_funcs import *

#region // GLOBAL SETTINGS ------------------------------------

FORECAST_STEPS = 8
OUTPUT_STEPS = [1, 2, 4, 8]
ESN_MATRIX_RESAMPLES = 1000 # set this to 1000 for paper

# Only generate plots (need to have results saved)?
DO_FORECASTS = False
DO_PLOTS = False
OPTIONAL_PLOTS = False

# Save plots and/or results?
SAVE_PLOTS = False
SAVE_RESULTS = False
SAVE_CSVS = False

# Paths
PATH_WD = Path(os.path.dirname(os.path.abspath(os.getcwd())))
PATH_PREFIX = None
# PATH_PREFIX = "Ensemble-MFESN-Economic-Forecasting"
if not PATH_PREFIX is None:
    PATH_WD = os.path.join(PATH_WD, PATH_PREFIX)

PATH_DATA = os.path.join(PATH_WD, "data", "preprocessed")
PATH_DFM = os.path.join(PATH_WD, "data", "dfm_medium")
PATH_FIGURES = os.path.join(PATH_WD, "figures_multistep")
PATH_RESULTS = os.path.join(PATH_WD, "results")
PATH_EXPORTS = os.path.join(PATH_WD, "exports")

# Utility functions
def save_to_pickle(object, filename):
    try:
        _file = open(os.path.join(PATH_RESULTS, filename), 'wb')
        pickle.dump(object, _file)
        _file.close()
    except:
        print(f"[!] Could not save to pickle '{filename}'")
    pass

def load_from_pickle(filename):
    try:
        _file = open(os.path.join(PATH_RESULTS, filename), 'rb')
        object = pickle.load(_file)
        _file.close()
    except:
        print(f"[!] Could not load from pickle '{filename}'")
        object = None
    return object

# Set up console
console = Console()

# Set up plot axes
plt.rc('axes', axisbelow=True)

def plt_xaxis_marker(x, y=None, **kwargs):
    if y is None:
        y = plt.gca().get_ylim()[0]
    plt.plot(x, y, zorder=10, clip_on=False, **kwargs)

#endregion

#region // DATA WRANGLING -----------------------------------

print("\n\n// PREPARING DATA -----------------------------------\n")

# Sample dates
startSampleDate = pd.to_datetime('1990-01-01')
endSampleDate   = pd.to_datetime('2019-12-31')

# Check data availability
preproc_data_filled = pd.read_csv(
    os.path.join(PATH_DATA, 'fullsample_output_filled_data.csv'),
    parse_dates = ["Date"]
).set_index("Date")

preproc_data_filled[~pd.isna(preproc_data_filled)] = 1

preproc_data = pd.read_csv(
    os.path.join(PATH_DATA, 'fullsample_output_data.csv'), 
    parse_dates = ["Date"]
).set_index("Date")
data = preproc_data.loc[startSampleDate:endSampleDate]

# Non-filled data
d_data = data[['D'+str(i) for i in range(1, 14)]].dropna()
m_data = data[['M'+str(i) for i in range(1, 21)]].dropna()
GDP_data = data[['Q1']].dropna()

# Interpolate non-filled daily data
d_data_interp = pd.DataFrame()
for c in d_data.columns:
    d_data_interp = d_data_interp.join(
        daily_begginingInterp(d_data[c], length=24, endOfMonth=False).rename(columns={0: c}),
        how='right',
    )

# Filled data
m_data_prefill = preproc_data[['M'+str(i) for i in range(1, 21)]].loc[
    (startSampleDate - pd.offsets.MonthBegin()):endSampleDate
].dropna()
GDP_data_prefill = preproc_data[['Q1']].loc[
    (startSampleDate - pd.offsets.QuarterBegin()):endSampleDate
].dropna()

# Interpolate non-filled daily data to month end
d_data_fill_interp = pd.DataFrame()
for c in d_data.columns:
    d_data_fill_interp = d_data_fill_interp.join(
        daily_begginingInterp(d_data[c], length=24, endOfMonth=True).rename(columns={0: c}),
        how='right',
    )

# Aggregate interpolated daily to mean over 6 days windows
d_data_fill_interp_agg = daily_aggregateByIndex(d_data_fill_interp, window_size=6, endOfWindow=True)

# Shift monthly, quarterly data to solar end-of-month date
m_data_prefill.index = m_data_prefill.index + pd.offsets.MonthEnd(0)
GDP_data_prefill.index = GDP_data_prefill.index + pd.offsets.QuarterEnd(0)

# Re-align filled data to daily indexes, join dataframe
pre_index = pd.date_range(
    start=startSampleDate - pd.offsets.YearBegin(), end=endSampleDate, freq='D'
)
pre_data_interp = pd.DataFrame(
    data=np.full((len(pre_index), 1), np.nan), index=pre_index
)

m_data_interp = pre_data_interp.join(m_data_prefill).ffill()
#GDP_data_interp = pre_data_interp.join(GDP_data_prefill).ffill()

# Make a new dataframe for filled explanatory variables
md_fill_data = d_data_fill_interp.join(m_data_interp.iloc[:,1:])#.join(GDP_data_interp.iloc[:,1:])

md_fill_agg_data = d_data_fill_interp_agg.join(m_data_interp.iloc[:,1:])

# Shift target data
GDP_fill_data  = GDP_data
GDP_fill_data.index = GDP_fill_data.index + pd.offsets.QuarterEnd(0)

# Define quick dataset creation functions
def getMediumDatasets(startDate, endDate, trainTestSplitDate=None):
    return get_Medium_Datasets(
        data={
            "GDP_data": GDP_data,
            "m_data": m_data,
            "d_data_interp": d_data_interp,
            "GDP_fill_data": GDP_fill_data,
            "md_fill_data": md_fill_data,
            "md_fill_agg_data": md_fill_agg_data,
        },
        startSampleDate=startDate,
        endSampleDate=endDate,
        trainTestSplitDate=trainTestSplitDate
    )

def getMediumDatasets_extended(startDate, endDate, trainTestSplitDate=None):
    return get_Medium_Datasets_extended(
        preproc_data=preproc_data,
        startSampleDate=startDate,
        endSampleDate=endDate,
        trainTestSplitDate=trainTestSplitDate
    )

#endregion

#region // DATASETS -----------------------------------------

if DO_FORECASTS:
    print("[!] Preparing datasets...")

    dataset2007 = getMediumDatasets(
        startDate=startSampleDate,
        endDate=endSampleDate,
        trainTestSplitDate='2007-12-31'
    )
    dataset2007_ext = getMediumDatasets_extended(
        startDate='1975-01-01', 
        endDate='2019-12-31', 
        trainTestSplitDate='2007-12-31'
    )

    # Pickle results
    if SAVE_RESULTS:
        dataset2007_save = {
            "GDP_data_train": dataset2007["GDP_data_train"],
            "GDP_data_test": dataset2007["GDP_data_test"],
        }
        save_to_pickle(
            dataset2007_save, 
            "dataset2007_GDP_data.pkl"
        )
        print("[!] GDP dataset (train / test) saved to results/ folder as pickle.")
    else:
        print("[!] GDP dataset not saved, set SAVE_RESULTS = True to save them.")

else:
    # Load datasets
    dataset2007_loaded = load_from_pickle(
        "dataset2007_GDP_data.pkl"
    )
    assert not dataset2007_loaded is None

    dataset2007 = {
        "GDP_data_train": dataset2007_loaded["GDP_data_train"],
        "GDP_data_test": dataset2007_loaded["GDP_data_test"],
    }

    print("[!] GDP dataset (train / test) loaded from results/ folder as pickle.")

#endregion

#region // FIXED PARAMETER PRE-CRISIS EVALUATION ------------

#region --- Benchmark Models

print("\n\n// PREPARING MODELS ---------------------------------\n")

def UnMean_fix_fit_multistep(steps, data):
    GDP_data_train = data['GDP_data_train']
    GDP_data_test = data['GDP_data_test']

    GDP_test_dates = GDP_data_test.index

    forecast_fix_multistep = np.zeros((len(GDP_test_dates), steps))
    for s in range(steps):
        forecast_fix_multistep[:,s] = np.ones(len(GDP_test_dates)) * GDP_data_train.mean().to_numpy()

    return pd.DataFrame(data=forecast_fix_multistep, index=GDP_test_dates, columns=range(steps))

def lowfreqAR_fix_fit_multistep(steps, data, direct=False):
    GDP_data_train = data['GDP_data_train'].asfreq('Q-DEC')
    GDP_data_test = data['GDP_data_test'].asfreq('Q-DEC')

    GDP_test_dates = GDP_data_test.index

    GDP_data_train, GDP_data_test, GDP_mu_train, GDP_sig_train = (
        normalize_train_test(GDP_data_train, GDP_data_test, 
            return_mu_sig=True)
    )

    forecast_fix_multistep = np.zeros((len(GDP_test_dates), steps))

    if direct:
         # Direct AR(1) multi-step forecasts
        for h in range(1, steps+1):
            # Fit AR(1) for step h
            ar1_fit_h = OLS(
                GDP_data_train.iloc[h:, :].to_numpy(),
                np.column_stack((
                    np.ones(len(GDP_data_train) - h),
                    GDP_data_train.iloc[:-h, :].to_numpy()
                ))
            ).fit()
            pars_h = ar1_fit_h.params
            
            # Predict step h
            GDP_data_pred = np.concatenate([
                    GDP_data_train.iloc[(-h):, :].to_numpy(),
                    GDP_data_test.iloc[:(-h), :].to_numpy()
                ]).squeeze()
            forecast_fix_multistep[:, h-1] = pars_h[0] + pars_h[1] * GDP_data_pred

    else:
        # Iterative AR(1) multi-step forecasts
        ar1_fit = OLS(
            GDP_data_train.iloc[1:, :].to_numpy(),
            np.column_stack((
                np.ones(len(GDP_data_train) - 1),
                GDP_data_train.iloc[:-1, :].to_numpy()
            ))
        ).fit()
        pars = ar1_fit.params

        for h in range(1, steps+1):
            GDP_data_pred = np.concatenate([
                    GDP_data_train.iloc[(-h):, :].to_numpy(),
                    GDP_data_test.iloc[:(-h), :].to_numpy()
                ]).squeeze()

            # Iteratively forecast step h
            for _ in range(h):
                GDP_data_pred = pars[0] + pars[1] * GDP_data_pred

            forecast_fix_multistep[:, h-1] = GDP_data_pred

    # De-normalize forecasts
    forecast_fix_multistep = forecast_fix_multistep * GDP_sig_train.to_numpy() + GDP_mu_train.to_numpy()

    return pd.DataFrame(data=forecast_fix_multistep, index=GDP_test_dates, columns=range(1, steps+1))

#endregion

#region --- DFM Models

def getDFMMultiStepAheadForecast(model_name, steps, date_slice=None):
    DFM_multistep_forecast = pd.DataFrame(
        data=None, columns=range(steps), index=date_slice,
    )
    for h in range(steps):
        # Read from files
        DFM_forecast_train_h = pd.read_csv(
            os.path.join(PATH_DFM, model_name, 'train_predictions_'+str(12*(h+1))+'.csv'),
        parse_dates = [0, 1]).set_index('Target.Date')
        DFM_forecast_test_h = pd.read_csv(
            os.path.join(PATH_DFM, model_name, 'test_predictions_'+str(12*(h+1))+'.csv'),
        parse_dates = [0, 1]).set_index('Target.Date')
        DFM_forecast_h = pd.concat([DFM_forecast_train_h, DFM_forecast_test_h])
        # Insert
        DFM_multistep_forecast.iloc[:,h] = DFM_forecast_h.loc[date_slice,'X0'].to_numpy()
    return DFM_multistep_forecast

#endregion

#region --- MIDAS Models

#! NOT CONSIDERED WITH MEDIUM MD DATASET!

#endregion

#region --- Single ESN 

esn_S_A = ESN(
    N=30,
    A=stateMatrixGenerator(
        (30, 30), 
        dist='sparse_normal', sparsity=10/30, normalize='eig',
        seed=20220623
    ),
    C=stateMatrixGenerator(
        (30, int(m_data.shape[1] + d_data.shape[1])), 
        dist='sparse_uniform', sparsity=10/30, normalize='norm2',
        seed=20220623
    ),
    rho=0.5,
    gamma=1,
    leak_rate=0.1,
    activation=np.tanh,
)

singleESN_A = ESNMultiFrequency((esn_S_A,), ar=False) 

esn_S_B = ESN(
    N=120,
    A=stateMatrixGenerator(
        (120, 120), 
        dist='sparse_normal', sparsity=10/120, normalize='eig',
        seed=19120623
    ),
    C=stateMatrixGenerator(
        (120, int(m_data.shape[1] + d_data.shape[1])), 
        dist='sparse_uniform', sparsity=10/120, normalize='norm2',
        seed=19120623
    ),
    rho=0.5,
    gamma=1,
    leak_rate=0.1,
    activation=np.tanh,
)

singleESN_B = ESNMultiFrequency((esn_S_B,), ar=False) 

#endregion

#region --- Multi ESN

esn_M_A = ESN(
    N=100,
    A=stateMatrixGenerator(
        (100, 100), 
        dist='sparse_normal', sparsity=10/100, normalize='eig',
        seed=20220623
    ),
    C=stateMatrixGenerator(
        (100, m_data.shape[1]), 
        dist='sparse_uniform', sparsity=10/100, normalize='norm2',
        seed=20220623
    ),
    rho=0.5,
    gamma=1.5,
    leak_rate=0,
    activation=np.tanh,
)

esn_D_A = ESN(
    N=20,
    A=stateMatrixGenerator(
        (20, 20), 
        dist='sparse_normal', sparsity=10/20, normalize='eig',
        seed=20220623
    ),
    C=stateMatrixGenerator(
        (20, d_data.shape[1]), 
        dist='sparse_uniform', sparsity=10/20, normalize='norm2',
        seed=20220623
    ),
    rho=0.5,
    gamma=0.5,
    leak_rate=0.1,
    activation=np.tanh,
)

multiESN_A = ESNMultiFrequency((esn_M_A, esn_D_A), ar=False) 

esn_M_B = ESN(
    N=100,
    A=stateMatrixGenerator(
        (100, 100), 
        dist='sparse_normal', sparsity=10/100, normalize='eig',
        seed=19120623
    ),
    C=stateMatrixGenerator(
        (100, m_data.shape[1]), 
        dist='sparse_uniform', sparsity=10/100, normalize='norm2',
        seed=19120623
    ),
    rho=0.08,
    gamma=0.25,
    leak_rate=0.3,
    activation=np.tanh,
)

esn_D_B = ESN(
    N=20,
    A=stateMatrixGenerator(
        (20, 20), 
        dist='sparse_normal', sparsity=10/20, normalize='eig',
        seed=19120623
    ),
    C=stateMatrixGenerator(
        (20, d_data.shape[1]), 
        dist='sparse_uniform', sparsity=10/20, normalize='norm2',
        seed=19120623
    ),
    rho=0.01,
    gamma=0.01,
    leak_rate=0.99,
    activation=np.tanh,
)

multiESN_B = ESNMultiFrequency((esn_M_B, esn_D_B), ar=False) 

#endregion

#region --- Estimation and forecasting

print("\n\n// FIXED PARAMETER PRE-CRISIS EVALUATION ------------\n")

if DO_FORECASTS:
    print("[!] Computing forecasts...")

    # DFM Forecasts
    DFM_slice_2007 = (
        pd.date_range('2008-01-01', '2019-12-31', freq="BQ") 
            - pd.tseries.offsets.MonthBegin(1) + pd.tseries.offsets.Day(23)
    )

    DFM_A_fix_multistep_for_2007 = getDFMMultiStepAheadForecast(
        'stock__10__6__[1]_fixed__2007', steps=8, date_slice=DFM_slice_2007,
    )

    DFM_B_fix_multistep_for_2007 = getDFMMultiStepAheadForecast(
        'almon__10__6__[1]_fixed__2007', steps=8, date_slice=DFM_slice_2007,
    )

    # Unconditional Mean Forecasts
    UnMean_fix_multistep_for_2007 = UnMean_fix_fit_multistep(
        data=dataset2007,
        steps=FORECAST_STEPS,
    )

    # Low-Frequency AR(1) Forecasts
    lowfreqAR_fix_multistep_for_2007 = lowfreqAR_fix_fit_multistep(
        data=dataset2007,
        steps=FORECAST_STEPS,
        direct=False,
    )

    # S-MFESN A Forecasts
    singleESN_A_agg_cv10_lambda_2007 = singleESN_A.ridge_lambda_cv(
        Y=dataset2007_ext['GDP_fill_data_train'], z=(dataset2007_ext['md_fill_agg_data_train'], ),
        method="ridge-isotropic",
        cv_options="-cv:10-test_size:5",
        steps=1,
        debug=False,
    )

    singleESN_A_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
        esnModel=singleESN_A, 
        Lambda=singleESN_A_agg_cv10_lambda_2007, 
        data=dataset2007,
        steps=FORECAST_STEPS,
        direct=False,
        aggregate=True,
    )

    # S-MFESN B Forecasts
    singleESN_B_agg_cv10_lambda_2007 = singleESN_B.ridge_lambda_cv(
        Y=dataset2007_ext['GDP_fill_data_train'], z=(dataset2007_ext['md_fill_agg_data_train'], ),
        method="ridge-isotropic",
        cv_options="-cv:10-test_size:5",
        steps=1,
        debug=False,
    )

    singleESN_B_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
        esnModel=singleESN_B, 
        Lambda=singleESN_B_agg_cv10_lambda_2007, 
        data=dataset2007,
        steps=FORECAST_STEPS,
        direct=False,
        aggregate=True,
    )

    # M-MFESN A Forecasts
    multiESN_A_cv10_lambda_2007 = multiESN_A.ridge_lambda_cv(
        Y=dataset2007_ext['GDP_data_train'], 
        z=(dataset2007_ext['m_data_train'], dataset2007_ext['d_data_train']),
        method="ridge-isotropic",
        cv_options="-cv:10-test_size:5",
        #cv_options="-cv_min_split_size:80-cv_max_split_size:80-test_size:1",
        steps=1,
        debug=False,
    )

    multiESN_A_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
        esnModel=multiESN_A, 
        Lambda=[multiESN_A_cv10_lambda_2007[0], multiESN_A_cv10_lambda_2007[0]], 
        data=dataset2007,
        steps=FORECAST_STEPS,
        direct=False,
    )

    # M-MFESN B Forecasts
    multiESN_B_cv10_lambda_2007 = multiESN_B.ridge_lambda_cv(
        Y=dataset2007_ext['GDP_data_train'], 
        z=(dataset2007_ext['m_data_train'], dataset2007_ext['d_data_train']),
        method="ridge-isotropic",
        cv_options="-cv:10-test_size:5",
        steps=1,
        debug=False,
    )

    multiESN_B_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
        esnModel=multiESN_B, 
        Lambda=[multiESN_B_cv10_lambda_2007[0], multiESN_B_cv10_lambda_2007[0]], 
        data=dataset2007,
        steps=FORECAST_STEPS,
        direct=False,
    )

    results_fixed_parameters = (
        UnMean_fix_multistep_for_2007,
        lowfreqAR_fix_multistep_for_2007,
        DFM_A_fix_multistep_for_2007,
        DFM_B_fix_multistep_for_2007,
        singleESN_A_agg_fix_multistep_for_2007,
        singleESN_B_agg_fix_multistep_for_2007,
        multiESN_A_fix_multistep_for_2007,
        multiESN_B_fix_multistep_for_2007,
    )
    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_fixed_parameters, 
            "results_M__multistep_fixed_parameters.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    # Export to CSVs
    if SAVE_CSVS:
        os.makedirs(PATH_EXPORTS, exist_ok=True)
        for i, name in enumerate([
            "UnMean_fix_multistep_for_2007",
            "lowfreqAR_fix_multistep_for_2007",
            "DFM_A_fix_multistep_for_2007",
            "DFM_B_fix_multistep_for_2007",
            "singleESN_A_agg_fix_multistep_for_2007",
            "singleESN_B_agg_fix_multistep_for_2007",
            "multiESN_A_fix_multistep_for_2007",
            "multiESN_B_fix_multistep_for_2007",
        ]):
            results_fixed_parameters[i].to_csv(
                os.path.join(PATH_EXPORTS, f"{name}.csv"),
                index=True,
                header=True,
            )
        print("[!] Results saved to exports/ folder as CSVs.")

else:
    # Load results
    results_fixed_parameters = load_from_pickle(
        "results_M__multistep_fixed_parameters.pkl"
    )
    assert not results_fixed_parameters is None

    (UnMean_fix_multistep_for_2007,
        lowfreqAR_fix_multistep_for_2007,
        DFM_A_fix_multistep_for_2007,
        DFM_B_fix_multistep_for_2007,
        singleESN_A_agg_fix_multistep_for_2007,
        singleESN_B_agg_fix_multistep_for_2007,
        multiESN_A_fix_multistep_for_2007,
        multiESN_B_fix_multistep_for_2007,
    ) = results_fixed_parameters

    print("[!] Results loaded from results/ folder as pickle.")

#endregion

#region --- Construct Plots and Tables
#%%%

def multistepStepMSFE_2007(forecasts_df):
    T = len(forecasts_df)
    steps = FORECAST_STEPS
    hStep_mse = np.zeros((steps))
    # for h, c in enumerate(forecasts_df.columns):
    for h in range(steps):
        hStep_mse[h] = (np.mean(np.square(
            dataset2007['GDP_data_test'].to_numpy() - forecasts_df.iloc[:,[h]].to_numpy()
        )[h:(T-FORECAST_STEPS+h+1), :]))
    return pd.DataFrame(data=hStep_mse.reshape(1, -1), columns=range(1, 1+steps))

def hStepMSFE_2007(multistepMSFE, h=0):
    return multistepMSFE.loc[h,0]

UnMean_MSE_multistep_fix_2007_td = multistepStepMSFE_2007(UnMean_fix_multistep_for_2007)
lowfreqAR_MSE_multistep_fix_2007_td = multistepStepMSFE_2007(lowfreqAR_fix_multistep_for_2007)
DFM_A_MSE_multistep_fix_2007_td = multistepStepMSFE_2007(DFM_A_fix_multistep_for_2007)
DMF_B_MSE_multistep_fix_2007_td = multistepStepMSFE_2007(DFM_B_fix_multistep_for_2007)
singleESN_A_MSE_multistep_fix_2007_td = multistepStepMSFE_2007(singleESN_A_agg_fix_multistep_for_2007)
singleESN_B_MSE_multistep_fix_2007_td = multistepStepMSFE_2007(singleESN_B_agg_fix_multistep_for_2007)
multiESN_A_MSE_multistep_fix_2007_td = multistepStepMSFE_2007(multiESN_A_fix_multistep_for_2007)
multiESN_B_MSE_multistep_fix_2007_td = multistepStepMSFE_2007(multiESN_B_fix_multistep_for_2007)

ref_MSE_2007_fix = UnMean_MSE_multistep_fix_2007_td
lowfreqAR_MSE_toRef_2007_fix = lowfreqAR_MSE_multistep_fix_2007_td / ref_MSE_2007_fix
DFM_A_MSE_toRef_2007_fix = DFM_A_MSE_multistep_fix_2007_td / ref_MSE_2007_fix
DMF_B_MSE_toRef_2007_fix = DMF_B_MSE_multistep_fix_2007_td / ref_MSE_2007_fix
singleESN_A_MSE_toRef_2007_fix = singleESN_A_MSE_multistep_fix_2007_td / ref_MSE_2007_fix
singleESN_B_MSE_toRef_2007_fix = singleESN_B_MSE_multistep_fix_2007_td / ref_MSE_2007_fix
multiESN_A_MSE_toRef_2007_fix = multiESN_A_MSE_multistep_fix_2007_td / ref_MSE_2007_fix
multiESN_B_MSE_toRef_2007_fix = multiESN_B_MSE_multistep_fix_2007_td / ref_MSE_2007_fix

# Joint RMSFE, MSFE and relative MSFE table
_tmp_table = Table(title="Fixed-param. Models  //  Table: Forecasting Performance", box=box.MINIMAL)
_tmp_table.add_column("Model", justify="right", no_wrap=True)
for h in OUTPUT_STEPS:
    _tmp_table.add_column(f"Rel. MSFE\n h = {h}", justify="left", no_wrap=True)
    # _tmp_table.add_column(f"Rel. MSFE, h = {h}", justify="center", no_wrap=True)
def _tmp_row_lambda (m, multih_msfe, multih_msfe_toRef, steps=OUTPUT_STEPS): 
    idx = [h-1 for h in steps]
    # _tmp_table.add_row(
    #     "", *[f"[{v1.round(9)}]" for v1 in multih_msfe.to_numpy().flatten()[idx]],
    # )
    # _tmp_table.add_row(
    #     m, *[f"{v2.round(4)}" for v2 in multih_msfe_toRef.to_numpy().flatten()[idx]],
    # )
    _tmp_table.add_row(
        m, *[f"{v2.round(3)} [{v1.round(9)}]" for v2, v1 in zip(
            multih_msfe_toRef.to_numpy().flatten()[idx], multih_msfe.to_numpy().flatten()[idx])
        ],
    )
_tmp_row_lambda("UMean",
    UnMean_MSE_multistep_fix_2007_td,
    pd.DataFrame(data=np.ones((1,8))),
)
_tmp_row_lambda("AR(1)",
    lowfreqAR_MSE_multistep_fix_2007_td,
    lowfreqAR_MSE_toRef_2007_fix,
)
_tmp_row_lambda("DFM [A]",
    DFM_A_MSE_multistep_fix_2007_td,
    DFM_A_MSE_toRef_2007_fix,
)
_tmp_row_lambda("DFM [B]",
    DMF_B_MSE_multistep_fix_2007_td,
    DMF_B_MSE_toRef_2007_fix,
)
_tmp_row_lambda("S-MFESN [A]",
    singleESN_A_MSE_multistep_fix_2007_td,
    singleESN_A_MSE_toRef_2007_fix,
)
_tmp_row_lambda("S-MFESN [B]",
    singleESN_B_MSE_multistep_fix_2007_td,
    singleESN_B_MSE_toRef_2007_fix,
)
_tmp_row_lambda("M-MFESN [A]",
    multiESN_A_MSE_multistep_fix_2007_td,
    multiESN_A_MSE_toRef_2007_fix,
)
_tmp_row_lambda("M-MFESN [B]",
    multiESN_B_MSE_multistep_fix_2007_td,
    multiESN_B_MSE_toRef_2007_fix,
)
console.print(_tmp_table)

# Base models for ensembles
dict_results_MSFE_base_models = {
    "S-MFESN_A": [
        singleESN_A_agg_fix_multistep_for_2007,
        singleESN_A_MSE_multistep_fix_2007_td,
        singleESN_A_MSE_toRef_2007_fix,
    ],
    "S-MFESN_B": [
        singleESN_B_agg_fix_multistep_for_2007,
        singleESN_B_MSE_multistep_fix_2007_td,
        singleESN_B_MSE_toRef_2007_fix,
    ],
    "M-MFESN_A": [
        multiESN_A_fix_multistep_for_2007,
        multiESN_A_MSE_multistep_fix_2007_td,
        multiESN_A_MSE_toRef_2007_fix,
    ],
    "M-MFESN_B": [
        multiESN_B_fix_multistep_for_2007,
        multiESN_B_MSE_multistep_fix_2007_td,
        multiESN_B_MSE_toRef_2007_fix,
    ],
}

#endregion

#endregion

#region // ENSEMBLE ESN W/ DIFFERENT RES. MATRICES ----------

#%%%
#region --- Run Ensembles

print("\n\n// ENSEMBLE ESN W/ DIFFERENT RAND. MATRICES ---------\n")

SEEDS_RP = np.linspace(5, 5000, num=ESN_MATRIX_RESAMPLES)

if DO_FORECASTS:
    print(f"[!] Computing ensemble forecasts... [{ESN_MATRIX_RESAMPLES} resamples]\n")

    print("--- S-MFESN A")
    #--- Single ESN A
    results_singleESN_A_fix_grid_seed = singleESN_A_multistep_fix_grid_seed(
        input_size=int(m_data.shape[1] + d_data.shape[1]),
        reservoir_size=30,
        dataset_cv=dataset2007_ext,
        dataset_fit_for=dataset2007,
        steps=FORECAST_STEPS,
        seed_list=SEEDS_RP,
        direct=False,
    )
    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_singleESN_A_fix_grid_seed, 
            "results_M__multistep_singleESN_A_fix_grid_seed.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    print("--- S-MFESN B")
    #--- Single ESN B
    results_singleESN_B_fix_grid_seed = singleESN_B_multistep_fix_grid_seed(
        input_size=int(m_data.shape[1] + d_data.shape[1]),
        reservoir_size=120,
        dataset_cv=dataset2007_ext,
        dataset_fit_for=dataset2007,
        steps=FORECAST_STEPS,
        seed_list=SEEDS_RP,
        direct=False,
    )
    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_singleESN_B_fix_grid_seed, 
            "results_M__multistep_singleESN_B_fix_grid_seed.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    print("--- M-MFESN A")
    #--- Multi ESN A
    results_multiESN_A_fix_grid_seed = multiESN_A_multistep_fix_grid_seed(
        input_size=(m_data.shape[1], d_data.shape[1]),
        reservoir_size=(100, 20),
        dataset_cv=dataset2007_ext,
        dataset_fit_for=dataset2007,
        steps=FORECAST_STEPS,
        seed_list=SEEDS_RP,
        direct=False,
    )
    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_multiESN_A_fix_grid_seed, 
            "results_M__multistep_multiESN_A_fix_grid_seed.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    print("--- M-MFESN B")
    #--- Multi ESN B
    results_multiESN_B_fix_grid_seed = multiESN_B_multistep_fix_grid_seed(
        input_size=(m_data.shape[1], d_data.shape[1]),
        reservoir_size=(100, 20),
        dataset_cv=dataset2007_ext,
        dataset_fit_for=dataset2007,
        steps=FORECAST_STEPS,
        seed_list=SEEDS_RP,
        direct=False,
    )

    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_multiESN_B_fix_grid_seed,
            "results_M__multistep_multiESN_B_fix_grid_seed.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    # Export to CSVs
    if SAVE_CSVS:
        os.makedirs(PATH_EXPORTS, exist_ok=True)
        for h in OUTPUT_STEPS:
            results_singleESN_A_fix_grid_seed[0][h-1].to_csv(
                os.path.join(PATH_EXPORTS, f"results_singleESN_A_fix_multistep_grid_seed_for_2007_h={h}.csv"),
                index=True,
                header=True,
            )
            results_singleESN_B_fix_grid_seed[0][h-1].to_csv(
                os.path.join(PATH_EXPORTS, f"results_singleESN_B_fix_multistep_grid_seed_for_2007_h={h}.csv"),
                index=True,
                header=True,
            )
            results_multiESN_A_fix_grid_seed[0][h-1].to_csv(
                os.path.join(PATH_EXPORTS, f"results_multiESN_A_fix_multistep_grid_seed_for_2007_h={h}.csv"),
                index=True,
                header=True,
            )
            results_multiESN_B_fix_grid_seed[0][h-1].to_csv(
                os.path.join(PATH_EXPORTS, f"results_multiESN_B_fix_multistep_grid_seed_for_2007_h={h}.csv"),
                index=True,
                header=True,
            )
        print("\n[!] Results exported to results/ folder as CSVs.")
    else:
        print("\n[!] Results not exported, set SAVE_CSVS = True to save them.")

else:
    # Load results
    results_singleESN_A_fix_grid_seed = load_from_pickle(
        "results_M__multistep_singleESN_A_fix_grid_seed.pkl"
    )
    assert not results_singleESN_A_fix_grid_seed is None

    results_singleESN_B_fix_grid_seed = load_from_pickle(
        "results_M__multistep_singleESN_B_fix_grid_seed.pkl"
    )
    assert not results_singleESN_B_fix_grid_seed is None

    results_multiESN_A_fix_grid_seed = load_from_pickle(
        "results_M__multistep_multiESN_A_fix_grid_seed.pkl"
    )
    assert not results_multiESN_A_fix_grid_seed is None

    results_multiESN_B_fix_grid_seed = load_from_pickle(
        "results_M__multistep_multiESN_B_fix_grid_seed.pkl"
    )
    assert not results_multiESN_B_fix_grid_seed is None

    print("[!] Results loaded from results/ folder as pickle.")

dict_results_ens_fix_grid_seed_models = {
    "S-MFESN_A": results_singleESN_A_fix_grid_seed,
    "S-MFESN_B": results_singleESN_B_fix_grid_seed,
    "M-MFESN_A": results_multiESN_A_fix_grid_seed,
    "M-MFESN_B": results_multiESN_B_fix_grid_seed,
}

#endregion

#region --- Utility Functions for Ensemble Forecast Combinations

def multistepStepMSFE_ensemble_2007(forecast_df_list):
    T = forecast_df_list[0].shape[0]
    K = forecast_df_list[0].shape[1]
    steps = FORECAST_STEPS
    hStep_mse = np.zeros((K, steps))
    for h in range(steps):
        for r in range(K):
            hStep_mse[r,h] = (np.mean(np.square(
                dataset2007['GDP_data_test'].to_numpy() - forecast_df_list[h].iloc[:,[r]].to_numpy()
            )[h:(T-FORECAST_STEPS+h+1), :]))
    return pd.DataFrame(data=hStep_mse, columns=range(1, 1+steps))

def multistepRelativeMSFE_ensemble_2007(msfe_df, msfe_df_benchmark):
    K = msfe_df.shape[0]
    relative_msfe = pd.DataFrame(
        data=None, index=range(K), columns=range(1, FORECAST_STEPS+1)
    )
    for i in range(FORECAST_STEPS):
        relative_msfe.iloc[:,i] = (
            msfe_df.iloc[:,i].to_numpy() / msfe_df_benchmark.iloc[:,i].to_numpy()
        ).flatten()
    return relative_msfe

# def multistepRelativeMSFE(msfe_model_list, msfe_df_benchmark):
#     relative_msfe = pd.DataFrame(
#         data=None, index=range(ESN_MATRIX_RESAMPLES), columns=range(1, FORECAST_STEPS+1)
#     )
#     for i in range(FORECAST_STEPS):
#         relative_msfe.iloc[:,i] = (
#             msfe_model_list[i].to_numpy() / msfe_df_benchmark.iloc[i,0]
#         ).flatten()
#     return relative_msfe

def median_for(ens_for):
    cf_list = []
    cw_list = []
    for h in range(FORECAST_STEPS):
        cf, cw = median_forecast(ens_for[h])
        cf.columns = [f"h={h+1}"]
        cf_list.append(cf)
        cw_list.append(cw)
    return pd.concat(cf_list, axis=1), cw_list

def sa_comb_for(ens_for):
    cf_list = []
    cw_list = []
    for h in range(FORECAST_STEPS):
        cf, cw = simple_average_forecast(ens_for[h])
        cf.columns = [f"h={h+1}"]
        cf_list.append(cf)
        cw_list.append(cw)
    return pd.concat(cf_list, axis=1), cw_list

def rollmse_comb_for(ens_for):
    cf_list = []
    cw_list = []
    for h in range(FORECAST_STEPS):
        cf, cw = rollMSE_weights_forecast(
            ens_for[h], 
            targets=dataset2007['GDP_data_test'], 
            h=(h+1), 
            window_size=4
        )
        cf.columns = [f"h={h+1}"]
        cf_list.append(cf)
        cw_list.append(cw)
    return pd.concat(cf_list, axis=1), cw_list

def const_hedge_comb_for(ens_for, ens_sfe):
    cf_list = []
    cw_list = []
    for h in range(FORECAST_STEPS):
        cf, cw = const_hedge_forecast(
            ens_for[h], 
            losses=ens_sfe[h],
            h=(h+1), 
            learning_rate=1e4,
        )
        cf.columns = [f"h={h+1}"]
        cf_list.append(cf)
        cw_list.append(cw)
    return pd.concat(cf_list, axis=1), cw_list

def dec_hedge_comb_for(ens_for, ens_sfe):
    cf_list = []
    cw_list = []
    for h in range(FORECAST_STEPS):
        cf, cw = dec_hedge_forecast(
            ens_for[h], 
            losses=ens_sfe[h],
            h=(h+1), 
            c0=1e4,
        )
        cf.columns = [f"h={h+1}"]
        cf_list.append(cf)
        cw_list.append(cw)
    return pd.concat(cf_list, axis=1), cw_list

def ftl_comb_for(ens_for, ens_sfe):
    cf_list = []
    cw_list = []
    for h in range(FORECAST_STEPS):
        cf, cw = ftl_forecast(
            ens_for[h], 
            losses=ens_sfe[h],
            h=(h+1), 
        )
        cf.columns = [f"h={h+1}"]
        cf_list.append(cf)
        cw_list.append(cw)
    return pd.concat(cf_list, axis=1), cw_list

def adahedge_comb_for(ens_for, ens_sfe):
    cf_list = []
    cw_list = []
    for h in range(FORECAST_STEPS):
        cf, cw = adahedge_forecast(
            ens_for[h], 
            losses=ens_sfe[h],
            h=(h+1), 
        )
        cf.columns = [f"h={h+1}"]
        cf_list.append(cf)
        cw_list.append(cw)
    return pd.concat(cf_list, axis=1), cw_list

def plot_comb_weights(weights_list, h=1, model_name="", method_name=""):
    plt.figure(figsize=(5,3))
    plt.plot(
        np.array(weights_list[h-1]),
        color='C0',
        alpha=np.minimum(0.5, 10/ESN_MATRIX_RESAMPLES),
    )
    plt.title(f'Ensemble Weights: {method_name} -- {ESN_MATRIX_RESAMPLES} Resamplings\n(h={h})')
    plt.xlabel('Resampling Index')
    plt.ylabel('Weight')
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_top_weights_lines(weights_list, top_eps=1e-9, h=1, model_name="", method_name=""):
    weights_array = weights_list[h-1].to_numpy()
    # Get the max weight across models for each resampling
    K = weights_list[h-1].shape[1]
    max_weights = np.max(weights_array, axis=1)
    max_weights_idx = np.argmax(weights_array, axis=1)
    # Find other weights that are within top_eps of the max weight
    close_to_max_weights_idx = []
    for i in range(len(max_weights_idx)):
        isclose_mask = np.isclose(weights_array[i], max_weights[i], atol=top_eps).nonzero()[0]
        # Remove the max index itself
        isclose_mask = isclose_mask[isclose_mask != max_weights_idx[i]]
        close_to_max_weights_idx.append(np.arange(K)[isclose_mask])

    plt.figure(figsize=(5.7,4))
    # Plot line for max weights
    plt.plot(np.arange(h, len(max_weights_idx)), max_weights_idx[h:], color='C5', linewidth=1.6)
    # Add scatter for close-to-max weights
    for i in range(len(close_to_max_weights_idx)):
        plt.scatter(
            [i]*len(close_to_max_weights_idx[i]),
            close_to_max_weights_idx[i],
            color='C5', s=2, alpha=0.8, zorder=5,
        )
    plt.grid(which='major', color="0.85")
    plt.xlim([0, len(max_weights)-1])
    ax_main = plt.gca()
    ax_main.set_xticks(np.arange(len(max_weights), step=5))
    x_q_labels = pd.period_range(start='2008Q1', end='2019Q4', freq='Q').to_numpy().astype(str)
    ax_main.set_xticklabels(x_q_labels[::5], rotation=30, ha="right")  # Set corresponding tick labels with rotation
    plt.ylabel(f'{method_name} Weights (Index)')
    plt.ylim([-5, weights_list[h-1].shape[1]+5])
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)

    if SAVE_PLOTS:
        figname = f'line_weights_{method_name}_{model_name}_multistep_h={h}_seed_leak_MediumMD_fix_2007.pdf' 
        plt.savefig(os.path.join(
            PATH_FIGURES, 
            figname,
        ), bbox_inches="tight")
        print(f"[!] Figure saved to figures/ folder as PDF:\n\t{figname}")

    plt.title(f'{model_name} - {method_name} Weighting (h = {h})')
    plt.gca().annotate(
        f'Note: {ESN_MATRIX_RESAMPLES} total reservoir matrix samples', 
        xy = (1.0, -0.15), xycoords='axes fraction', ha='right', va="center",
        fontsize=8, color="0.3"
    )
    plt.show()

def plot_top_weights_stack(weights_list, rank_list, top_num=5, h=1, model_name="", method_name=""):
    weights_array = weights_list[h-1].to_numpy()
    rank_array = rank_list[h-1]
    # Determine top weights indices
    sorted_indices = np.argsort(rank_array)
    top_idxs = np.arange(len(rank_array))[sorted_indices]
    # Create array of weights
    top_weights_array = np.zeros((weights_array.shape[0], top_num+1))
    for k in range(top_num):
        # idx_k = np.argwhere(rank_array == k+1).item()
        # top_weights_array[:, k] = weights_array[:, idx_k]
        top_weights_array[:, k] = weights_array[:, top_idxs[k]]
    # Compute leftover weights
    top_weights_array[:, -1] = np.sum(weights_array[:, top_idxs[top_num:]], axis=1)

    # print(np.sum(weights_array, axis=1))      # Debug: Check if rows sum to 1
    # print(np.sum(top_weights_array, axis=1))  # Debug: Check if rows sum to 1

    # Plot stacked area chart
    sp_labels = [f"Rank #{r+1}" for r in range(top_num)] + ["Other"] 
    sp_colors = ["C"+str(i) for i in range(top_num)] + ["0.7"]

    plt.rc('axes', axisbelow=False)
    plt.figure(figsize=(5.7,4))
    plt.stackplot(np.arange(len(top_weights_array)), top_weights_array.T, labels=sp_labels, colors=sp_colors)
    plt.grid(which='major', color="0.2", alpha=0.3)
    plt.xlim([0, len(top_weights_array)-1])
    ax_main = plt.gca()
    ax_main.set_xticks(np.arange(len(top_weights_array), step=5))
    x_q_labels = pd.period_range(start='2008Q1', end='2019Q4', freq='Q').to_numpy().astype(str)
    ax_main.set_xticklabels(x_q_labels[::5], rotation=30, ha="right")  # Set corresponding tick labels with rotation
    plt.ylabel(f'{method_name} Weights')
    plt.ylim([0, 1])
    plt.legend(fontsize=10, framealpha=0, fancybox=False, 
               bbox_to_anchor=(0.08, 1.17), loc='upper left',
               ncol=3)
    
    if SAVE_PLOTS:
        figname = f'stack_weights_{method_name}_{model_name}_multistep_h={h}_seed_leak_MediumMD_fix_2007.pdf' 
        plt.savefig(os.path.join(
            PATH_FIGURES, 
            figname,
        ), bbox_inches="tight")
        print(f"[!] Figure saved to figures/ folder as PDF:\n\t{figname}")

    plt.title(f'{model_name} - {method_name} Weighting (h = {h})')
    plt.gca().annotate(
        f'Note: {ESN_MATRIX_RESAMPLES} total reservoir matrix samples', 
        xy = (1.0, -0.15), xycoords='axes fraction', ha='right', va="center",
        fontsize=8, color="0.3"
    )
    plt.show()
    plt.rc('axes', axisbelow=True)

#endregion

#%%%
#region --- Construct Plots and Tables

for model_name, results in dict_results_ens_fix_grid_seed_models.items():
    print(f"\n--- Ensemble Combination // {model_name} ---\n")

    # Unwrap result tuple
    (esnmodel_fix_grid_seed__forecast, 
    esnmodel_fix_grid_seed__sfe,
    esnmodel_fix_grid_seed__msfe,
    ) = results

    esnmodel_fix_grid_seed__MSFE = multistepStepMSFE_ensemble_2007(
        esnmodel_fix_grid_seed__forecast
    )
    esnmodel_fix_grid_seed__RelMSE = multistepRelativeMSFE_ensemble_2007(
        esnmodel_fix_grid_seed__MSFE, ref_MSE_2007_fix
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plt.figure(figsize=(5,3))
        # plt.plot(
        #     esnmodel_fix_grid_seed__RelMSE,
        #     color='lightgray',
        #     alpha=np.minimum(0.5, 10/ESN_MATRIX_RESAMPLES),
        # )
        # # plt.title(f'Single ESN A: Relative MSFE Distribution over {ESN_MATRIX_RESAMPLES} Resamplings\n(Fixed Parameters, Pre-Crisis Evaluation, h={h})')
        # plt.xticks(range(FORECAST_STEPS))
        # # plt.yticks(np.linspace(0.7, 1.1, num=9))
        # plt.ylabel('Relative MSFE')
        # plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
        # # if SAVE_PLOTS:
        # #     plt.savefig(
        # #         os.path.join(
        # #             PATH_FIGURES, 
        # #             f'figure_M__multistep_singleESN_A_fix_grid_seed_relMSFE_h{h}.png'
        # #         ),
        # #         dpi=300,
        # #         bbox_inches='tight',
        # #     )
        # plt.show()
        pass

    # Compute ensemble forecasts
    # Median
    esnmodel_fix_grid_seed__forecast_ens_median, _ = median_for(
        esnmodel_fix_grid_seed__forecast
    )

    # Simple Average
    esnmodel_fix_grid_seed__forecast_ens_sa, _ = sa_comb_for(
        esnmodel_fix_grid_seed__forecast
    )

    # RollMSE
    (esnmodel_fix_grid_seed__forecast_ens_rollmse, 
    esnmodel_fix_grid_seed__weights_ens_rollmse) = rollmse_comb_for(
        esnmodel_fix_grid_seed__forecast
    )

    # Constant Hedge
    (esnmodel_fix_grid_seed__forecast_ens_consthedge, 
    esnmodel_fix_grid_seed__weights_ens_consthedge) = const_hedge_comb_for(
        esnmodel_fix_grid_seed__forecast,
        esnmodel_fix_grid_seed__sfe,
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plot_comb_weights(
        #     esnmodel_fix_grid_seed__weights_ens_consthedge,
        #     h=1,
        #     method_name="Constant Hedge"
        # )
        pass

    # Decreasing Hedge
    (esnmodel_fix_grid_seed__forecast_ens_dechedge, 
    esnmodel_fix_grid_seed__weights_ens_dechedge) = dec_hedge_comb_for(
        esnmodel_fix_grid_seed__forecast,
        esnmodel_fix_grid_seed__sfe,
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plot_comb_weights(
        #     esnmodel_fix_grid_seed__weights_ens_dechedge,
        #     h=1,
        #     method_name="Decreasing Hedge"
        # )
        pass

    # Follow-The-Leader
    (esnmodel_fix_grid_seed__forecast_ens_ftl, 
    esnmodel_fix_grid_seed__weights_ens_ftl) = ftl_comb_for(
        esnmodel_fix_grid_seed__forecast,
        esnmodel_fix_grid_seed__sfe,
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plot_comb_weights(
        #     esnmodel_fix_grid_seed__weights_ens_ftl,
        #     h=1,
        #     method_name="Follow-The-Leader"
        # )
        pass

    # Adaptive Hedge
    (esnmodel_fix_grid_seed__forecast_ens_adahedge, 
    esnmodel_fix_grid_seed__weights_ens_adahedge) = adahedge_comb_for(
        esnmodel_fix_grid_seed__forecast,
        esnmodel_fix_grid_seed__sfe,
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plot_comb_weights(
        #     esnmodel_fix_grid_seed__weights_ens_adahedge,
        #     h=1,
        #     method_name="Adaptive Hedge"
        # )
        pass

    # (Optional) Plot combination methods forecasts against actuals
    if DO_PLOTS and OPTIONAL_PLOTS:
        for h in OUTPUT_STEPS:
            plt.figure(figsize=(10,4))
            plt.plot(
                dataset2007['GDP_data_test'],
                color='black', linewidth=2, label='Target',
            )
            t_idx = list(range(h-1, (dataset2007['GDP_data_test'].shape[0]-FORECAST_STEPS+h)))
            plt.plot(
                dict_results_MSFE_base_models[model_name][0].iloc[t_idx,h-1],
                color='C7', alpha=0.7, linewidth=1.5, label=model_name,
            )
            plt.plot(
                esnmodel_fix_grid_seed__forecast_ens_median.iloc[t_idx,h-1],
                color='C8', alpha=0.7, linewidth=1.5, label='Median',
            )
            plt.plot(
                esnmodel_fix_grid_seed__forecast_ens_sa.iloc[t_idx,h-1],
                color='C0', alpha=0.7, linewidth=1.5, ls=":", label='Simple Average',
            )
            plt.plot(
                esnmodel_fix_grid_seed__forecast_ens_rollmse.iloc[t_idx,h-1],
                color='C1', alpha=0.7, linewidth=1.5, ls=":", label='RollMSE',
            )
            plt.plot(
                esnmodel_fix_grid_seed__forecast_ens_consthedge.iloc[t_idx,h-1],
                color='C2', alpha=0.7, linewidth=1.5, ls="--", label='Const. Hedge',
            )
            plt.plot(
                esnmodel_fix_grid_seed__forecast_ens_dechedge.iloc[t_idx,h-1],
                color='C6', alpha=0.7, linewidth=1.5, ls="--", label='Dec. Hedge',
            )
            plt.plot(
                esnmodel_fix_grid_seed__forecast_ens_ftl.iloc[t_idx,h-1],
                color='C3', alpha=0.7, linewidth=1.5, label='FTL',
            )
            plt.plot(
                esnmodel_fix_grid_seed__forecast_ens_adahedge.iloc[t_idx,h-1],
                color='C4', alpha=0.7, linewidth=1.5, label='AdaHedge',
            )
            plt.title(f"{model_name} Ensemble -- RP // Forecast vs Targets (h={h})")
            plt.xlabel('Time Index')
            plt.ylabel('GDP Growth Rate')
            plt.legend()
            plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
            plt.show()

    # MSFE of ensemble forecast
    esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_median = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed__forecast_ens_median
    )
    esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_sa = multistepStepMSFE_2007(
        esnmodel_fix_grid_seed__forecast_ens_sa
    )
    esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_rollmse = multistepStepMSFE_2007(
        esnmodel_fix_grid_seed__forecast_ens_rollmse
    )
    esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_consthedge = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed__forecast_ens_consthedge
    )
    esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_dechedge = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed__forecast_ens_dechedge
    )
    esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_ftl = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed__forecast_ens_ftl
    )  
    esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_adahedge = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed__forecast_ens_adahedge
    )

    # Relative MSFE of ensemble forecast
    esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_median = (
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_median / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_sa = (
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_sa / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_rollmse = (
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_rollmse / ref_MSE_2007_fix
    )   
    esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_consthedge = (
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_consthedge / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_dechedge = (
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_dechedge / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_ftl = (
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_ftl / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_adahedge = (
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_adahedge / ref_MSE_2007_fix
    )

    # ECDF Plot for ensemble forecasts
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_baseline = (
                dict_results_MSFE_base_models[model_name][2].iloc[:,h-1]
            )

            plt.figure(figsize=(5.7,4))
            plt.ecdf(
                x=esnmodel_fix_grid_seed__RelMSE.iloc[:,h-1].to_numpy().astype(float),
                color="#758E85",
                alpha=0.75,
                linewidth=1.6,
            )
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_baseline.item(), 
                        color='0.2', linestyle='-', linewidth=1.2, 
                        label=model_name)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_median.iloc[:,h-1].item(), 
                        color='0.2', linestyle=':', linewidth=1.4, 
                        label=f'Median')
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_sa.iloc[:,h-1].item(), 
                        color='C0', linestyle='dashed', linewidth=1.7, 
                        label=f'Average')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_sa.iloc[:,h-1].item(), y=0,
                        color='C0', marker="X", markersize=8, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_rollmse.iloc[:,h-1].item(), 
                        color='C1', linestyle='dashed', linewidth=1.7, 
                        label=f'RollMSE')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_rollmse.iloc[:,h-1].item(), y=0.04, 
                        color='C1', marker="o", markersize=8, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_ftl.iloc[:,h-1].item(), 
                        color='C3', linestyle='dashed', linewidth=1.7, 
                        label=f'FTL')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_ftl.iloc[:,h-1].item(), y=0.,
                        color='C3', marker="^", markersize=8, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_consthedge.iloc[:,h-1].item(), 
                        color='C2', linestyle='dashed', linewidth=1.7, 
                        label=f'Hedge')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_consthedge.iloc[:,h-1].item(), y=0.08,
                        color='C2', marker="s", markersize=8, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_dechedge.iloc[:,h-1].item(), 
                        color='C6', linestyle='dashed', linewidth=1.7, 
                        label=f'DecHedge')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_dechedge.iloc[:,h-1].item(), y=0.12,
                        color='C6', marker="v", markersize=7, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_adahedge.iloc[:,h-1].item(),
                        color='C4', linestyle='dashed', linewidth=1.7, 
                        label=f'AdaHedge')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_adahedge.iloc[:,h-1].item(), y=0.,
                        color='C4', marker="D", markersize=7, alpha=0.5)
            # plt.xlim(bin_range)
            if h == 1:
                plt.xlim([0.45, 1.15])
            elif h == 2:
                plt.xlim([0.75, 1.25])
            plt.xlabel('Relative MSFE')
            plt.ylabel('Probability')
            plt.legend(fontsize='small', loc="lower right", framealpha=1, edgecolor='0.85', fancybox=False)
            plt.grid(which='major', color="0.85")
            # if SAVE_PLOTS: 
            #     plt.savefig(os.path.join(
            #         PATH_FIGURES, 
            #         'ecdf_ensembles_multistep_singleESN_A_seed_Medi_MD_fix_2007.pdf'
            #     ), bbox_inches="tight")

            if SAVE_PLOTS:
                figname = f'ecdf_{model_name}_multistep_h={h}_seed_MediumMD_fix_2007.pdf' 
                plt.savefig(os.path.join(
                    PATH_FIGURES, 
                    figname,
                ), bbox_inches="tight")
                print(f"[!] Figure saved to figures/ folder as PDF:\n\t{figname}")

            plt.title(f'Relative MSFE ECDF with Ensembles - {model_name}')
            plt.gca().annotate(
                f'Note: {ESN_MATRIX_RESAMPLES} reservoir matrix samples', 
                xy = (1.0, -0.15), xycoords='axes fraction', ha='right', va="center",
                fontsize=8, color="0.3"
            )
            plt.show()

    # Rank models by final MSFE
    rank_list = []
    for h in range(1, FORECAST_STEPS+1):
        ranks_h = esnmodel_fix_grid_seed__RelMSE.iloc[:,h-1].rank().to_numpy().astype(int)
        rank_list.append(ranks_h)

    # Plot RollMSE weights over time
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            plot_top_weights_stack(
                esnmodel_fix_grid_seed__weights_ens_rollmse, 
                rank_list, 
                top_num=5, 
                h=h, 
                model_name=model_name, 
                method_name="RollMSE"
            )

    # Plot DecHedge weights over time
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            plot_top_weights_stack(
                esnmodel_fix_grid_seed__weights_ens_dechedge, 
                rank_list, 
                top_num=5, 
                h=h, 
                model_name=model_name, 
                method_name="DecHedge"
            )

    # Plot FTL weights over time
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            plot_top_weights_lines(
                esnmodel_fix_grid_seed__weights_ens_ftl, 
                top_eps=1e-9, 
                h=h, 
                model_name=model_name, 
                method_name="FTL"
            )

    # Plot AdaHedge weights over time
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            plot_top_weights_stack(
                esnmodel_fix_grid_seed__weights_ens_adahedge, 
                rank_list, 
                top_num=5, 
                h=h, 
                model_name=model_name, 
                method_name="AdaHedge"
            )

    # Joint RMSFE, MSFE and relative MSFE table for ensemble forecasts
    _tmp_table = Table(title=f"Table: {model_name} Ensemble -- RP [{ESN_MATRIX_RESAMPLES} resamples]", box=box.MINIMAL)
    _tmp_table.add_column("Combination", justify="right", no_wrap=True)
    for h in OUTPUT_STEPS:
        _tmp_table.add_column(f"Rel. MSFE\n h = {h}", justify="left", no_wrap=True)
        # _tmp_table.add_column(f"Rel. MSFE, h = {h}", justify="center", no_wrap=True)
    def _tmp_row_lambda (m, multih_msfe, multih_msfe_toRef, multih_msfe_baseline, steps=OUTPUT_STEPS): 
        idx = [h-1 for h in steps]
        # _tmp_table.add_row(
        #     m, *[f"{v1.round(6)}" for v1 in multih_msfe.to_numpy().flatten()[idx]],
        # )
        # _tmp_table.add_row(
        #     m, *[f"{v2.round(3)}" for v2 in multih_msfe_toRef.to_numpy().flatten()[idx]],
        # )
        _tmp_table.add_row(
            m, *[f"{v2.round(3)} [{v1.round(9)}]" for v2, v1 in zip(
                multih_msfe_toRef.to_numpy().flatten()[idx], multih_msfe.to_numpy().flatten()[idx])
            ],
        )
        multih_perc_change = [
            (
                (multih_msfe_toRef.to_numpy().flatten()[i] - multih_msfe_baseline.to_numpy().flatten()[i])
                / multih_msfe_baseline.to_numpy().flatten()[i]
            ) * 100
            for i in idx
        ]
        _tmp_table.add_row(
            "", *[f"{v.round(2):+}" for v in multih_perc_change],
        )
    _tmp_row_lambda("Baseline",
        dict_results_MSFE_base_models[model_name][1],
        dict_results_MSFE_base_models[model_name][2],
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("Median",
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_median,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_median,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("Simple Average",
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_sa,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_sa,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("RollMSE",
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_rollmse,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_rollmse,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("FTL",
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_ftl,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_ftl,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("Const. Hedge",
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_consthedge,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_consthedge,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("Dec. Hedge",
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_dechedge,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_dechedge,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("AdaHedge",
        esnmodel__MSE_multistep_fix_grid_seed_2007_td_ens_adahedge,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_ens_adahedge,
        dict_results_MSFE_base_models[model_name][2],
    )
    console.print(_tmp_table)

#endregion

#endregion

#region // ENSEMBLE ESN W/ DIFFERENT RES. MATRICS AND LEAK --

#%%%
#region --- Run Ensembles

print("\n\n// ENSEMBLE ESN W/ DIFFERENT RAND. MATRICES AND LEAK RATES ---------\n")

LEAK_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS_RP_LEAK = np.linspace(5, 5000, num=ESN_MATRIX_RESAMPLES//len(LEAK_RATES))

if DO_FORECASTS:
    print(f"[!] Computing ensemble forecasts... [{ESN_MATRIX_RESAMPLES} resamples]\n")

    print("--- S-MFESN A")
    #--- Single ESN A
    results_singleESN_A_fix_grid_seed_leak = singleESN_A_multistep_fix_grid_seed_leak(
        input_size=int(m_data.shape[1] + d_data.shape[1]),
        reservoir_size=30,
        dataset_cv=dataset2007_ext,
        dataset_fit_for=dataset2007,
        steps=FORECAST_STEPS,
        seed_list=SEEDS_RP_LEAK,
        leak_rate_list=LEAK_RATES,
        direct=False,
    )
    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_singleESN_A_fix_grid_seed_leak, 
            "results_M__multistep_singleESN_A_fix_grid_seed_leak.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    print("--- S-MFESN B")
    #--- Single ESN B
    results_singleESN_B_fix_grid_seed_leak = singleESN_B_multistep_fix_grid_seed_leak(
        input_size=int(m_data.shape[1] + d_data.shape[1]),
        reservoir_size=120,
        dataset_cv=dataset2007_ext,
        dataset_fit_for=dataset2007,
        steps=FORECAST_STEPS,
        seed_list=SEEDS_RP_LEAK,
        leak_rate_list=LEAK_RATES,
        direct=False,
    )
    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_singleESN_B_fix_grid_seed_leak, 
            "results_M__multistep_singleESN_B_fix_grid_seed_leak.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    print("--- M-MFESN A")
    #--- Multi ESN A
    results_multiESN_A_fix_grid_seed_leak = multiESN_A_multistep_fix_grid_seed_leak(
        input_size=(m_data.shape[1], d_data.shape[1]),
        reservoir_size=(100, 20),
        dataset_cv=dataset2007_ext,
        dataset_fit_for=dataset2007,
        steps=FORECAST_STEPS,
        seed_list=SEEDS_RP_LEAK,
        leak_rate_list=LEAK_RATES,
        direct=False,
    )
    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_multiESN_A_fix_grid_seed_leak, 
            "results_M__multistep_multiESN_A_fix_grid_seed_leak.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    print("--- M-MFESN B")
    #--- Multi ESN B
    results_multiESN_B_fix_grid_seed_leak = multiESN_B_multistep_fix_grid_seed_leak(
        input_size=(m_data.shape[1], d_data.shape[1]),
        reservoir_size=(100, 20),
        dataset_cv=dataset2007_ext,
        dataset_fit_for=dataset2007,
        steps=FORECAST_STEPS,
        seed_list=SEEDS_RP_LEAK,
        leak_rate_list=LEAK_RATES,
        direct=False,
    )
    # Pickle results
    if SAVE_RESULTS:
        save_to_pickle(
            results_multiESN_B_fix_grid_seed_leak,
            "results_M__multistep_multiESN_B_fix_grid_seed_leak.pkl"
        )
        print("[!] Results saved to results/ folder as pickle.")
    else:
        print("[!] Results not saved, set SAVE_RESULTS = True to save them.")

    # Export to CSVs
    if SAVE_CSVS:
        os.makedirs(PATH_EXPORTS, exist_ok=True)
        for h in OUTPUT_STEPS:
            results_singleESN_A_fix_grid_seed_leak[0][h-1].to_csv(
                os.path.join(PATH_EXPORTS, f"results_singleESN_A_fix_multistep_grid_seed_leak_for_2007_h={h}.csv"),
                index=True,
                header=True,
            )
            results_singleESN_B_fix_grid_seed_leak[0][h-1].to_csv(
                os.path.join(PATH_EXPORTS, f"results_singleESN_B_fix_multistep_grid_seed_leak_for_2007_h={h}.csv"),
                index=True,
                header=True,
            )
            results_multiESN_A_fix_grid_seed_leak[0][h-1].to_csv(
                os.path.join(PATH_EXPORTS, f"results_multiESN_A_fix_multistep_grid_seed_leak_for_2007_h={h}.csv"),
                index=True,
                header=True,
            )
            results_multiESN_B_fix_grid_seed_leak[0][h-1].to_csv(
                os.path.join(PATH_EXPORTS, f"results_multiESN_B_fix_multistep_grid_seed_leak_for_2007_h={h}.csv"),
                index=True,
                header=True,
            )
        print("\n[!] Results exported to results/ folder as CSVs.")
    else:
        print("\n[!] Results not exported, set SAVE_CSVS = True to save them.")

else:
    # Load results
    results_singleESN_A_fix_grid_seed_leak = load_from_pickle(
        "results_M__multistep_singleESN_A_fix_grid_seed_leak.pkl"
    )
    assert not results_singleESN_A_fix_grid_seed_leak is None

    results_singleESN_B_fix_grid_seed_leak = load_from_pickle(
        "results_M__multistep_singleESN_B_fix_grid_seed_leak.pkl"
    )
    assert not results_singleESN_B_fix_grid_seed_leak is None

    results_multiESN_A_fix_grid_seed_leak = load_from_pickle(
        "results_M__multistep_multiESN_A_fix_grid_seed_leak.pkl"
    )
    assert not results_multiESN_A_fix_grid_seed_leak is None

    results_multiESN_B_fix_grid_seed_leak = load_from_pickle(
        "results_M__multistep_multiESN_B_fix_grid_seed_leak.pkl"
    )
    assert not results_multiESN_B_fix_grid_seed_leak is None

    print("[!] Results loaded from results/ folder as pickle.")

dict_results_ens_fix_grid_seed_leak_models = {
    "S-MFESN_A": results_singleESN_A_fix_grid_seed_leak,
    "S-MFESN_B": results_singleESN_B_fix_grid_seed_leak,
    "M-MFESN_A": results_multiESN_A_fix_grid_seed_leak,
    "M-MFESN_B": results_multiESN_B_fix_grid_seed_leak,
}

#endregion

#region --- Utility Functions

def split_by_leak_rate(results_tuple):
    lr_list = results_tuple[3]
    leak_rates = sorted(list(set(lr_list)))
    split_results = []
    for lr in leak_rates:
        idx_lr = [i for i, v in enumerate(lr_list) if v == lr]
        split_results.append({
            "leak_rate": lr,
            "forecast_dfs": [df.iloc[:,idx_lr] for df in results_tuple[0]],
            "sfe_dfs": [df.iloc[:,idx_lr] for df in results_tuple[1]],
            "msfe_dfs": [df.iloc[idx_lr] for df in results_tuple[2]],
        })
    return split_results

def plot_msfe_ecdf_by_leak_rate(results, results_by_leak_list, model_name="",):
    plt.figure(figsize=(5.7,4))
    plt.ecdf(x=results, color="0.2", label="All models", linewidth=1.6)
    for j, rbl in enumerate(results_by_leak_list):
        leak_rate, r = rbl
        plt.ecdf(x=r, color=f"C{j}", label=r"$\alpha =$" + f"{leak_rate}", linewidth=1.6)
    plt.xlim([0.35, 1.25])
    plt.xlabel('Relative MSFE')
    plt.ylabel('Probability')
    plt.legend(fontsize='small', loc="upper left", framealpha=1, edgecolor='0.85', fancybox=False)
    plt.grid(which='major', color="0.85")

    if SAVE_PLOTS: 
        figname = f'ecdf_by_leak_rate_{model_name}_multistep_h={h}_seed_leak_MediumMD_fix_2007.pdf' 
        plt.savefig(os.path.join(
            PATH_FIGURES, 
            figname,
        ), bbox_inches="tight")
        print(f"[!] Figure saved to figures/ folder as PDF:\n\t{figname}")

    plt.title(f'{model_name} - CDF Comparison: Different Reservoir Matrices and Leak Rates')
    plt.gca().annotate(
        f'Note: {ESN_MATRIX_RESAMPLES} total reservoir matrix samples', 
        xy = (1.0, -0.15), xycoords='axes fraction', ha='right', va="center",
        fontsize=8, color="0.3"
    )
    plt.show()

#endregion

#%%%
#region --- Construct Plots and Tables

for model_name, results in dict_results_ens_fix_grid_seed_leak_models.items():
    print(f"\n--- Ensemble Combination // {model_name} ---\n")

    # Unwrap result tuple
    (esnmodel_fix_grid_seed_leak__forecast, 
    esnmodel_fix_grid_seed_leak__sfe,
    esnmodel_fix_grid_seed_leak__msfe,
    esnmodel_fix_grid_seed_leak__lr_list,
    ) = results

    esnmodel_fix_grid_seed_leak__MSFE = multistepStepMSFE_ensemble_2007(
        esnmodel_fix_grid_seed_leak__forecast
    )
    esnmodel_fix_grid_seed_leak__RelMSE = multistepRelativeMSFE_ensemble_2007(
        esnmodel_fix_grid_seed_leak__MSFE, ref_MSE_2007_fix
    )

    if OPTIONAL_PLOTS:
        # plt.figure(figsize=(5,3))
        # plt.plot(
        #     esnmodel_fix_grid_seed_leak__RelMSE,
        #     color='lightgray',
        #     alpha=np.minimum(0.5, 10/ESN_MATRIX_RESAMPLES),
        # )
        # # plt.title(f'Single ESN A: Relative MSFE Distribution over {ESN_MATRIX_RESAMPLES} Resamplings\n(Fixed Parameters, Pre-Crisis Evaluation, h={h})')
        # plt.xticks(range(FORECAST_STEPS))
        # # plt.yticks(np.linspace(0.7, 1.1, num=9))
        # plt.ylabel('Relative MSFE')
        # plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
        # # if SAVE_PLOTS:
        # #     plt.savefig(
        # #         os.path.join(
        # #             PATH_FIGURES, 
        # #             f'figure_M__multistep_singleESN_A_fix_grid_seed_relMSFE_h{h}.png'
        # #         ),
        # #         dpi=300,
        # #         bbox_inches='tight',
        # #     )
        # plt.show()
        pass

    # Compute ensemble forecasts
    # Median
    esnmodel_fix_grid_seed_leak__forecast_ens_median, _ = median_for(
        esnmodel_fix_grid_seed_leak__forecast
    )

    # Simple Average
    esnmodel_fix_grid_seed_leak__forecast_ens_sa, _ = sa_comb_for(
        esnmodel_fix_grid_seed_leak__forecast
    )

    # RollMSE
    (esnmodel_fix_grid_seed_leak__forecast_ens_rollmse, 
    esnmodel_fix_grid_seed_leak__weights_ens_rollmse) = rollmse_comb_for(
        esnmodel_fix_grid_seed_leak__forecast
    )

    # Constant Hedge
    (esnmodel_fix_grid_seed_leak__forecast_ens_consthedge, 
    esnmodel_fix_grid_seed_leak__weights_ens_consthedge) = const_hedge_comb_for(
        esnmodel_fix_grid_seed_leak__forecast,
        esnmodel_fix_grid_seed_leak__sfe,
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plot_comb_weights(
        #     esnmodel_fix_grid_seed_leak__weights_ens_consthedge,
        #     h=1,
        #     method_name="Constant Hedge"
        # )
        pass

    # Decreasing Hedge
    (esnmodel_fix_grid_seed_leak__forecast_ens_dechedge, 
    esnmodel_fix_grid_seed_leak__weights_ens_dechedge) = dec_hedge_comb_for(
        esnmodel_fix_grid_seed_leak__forecast,
        esnmodel_fix_grid_seed_leak__sfe,
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plot_comb_weights(
        #     esnmodel_fix_grid_seed_leak__weights_ens_dechedge,
        #     h=1,
        #     method_name="Decreasing Hedge"
        # )
        pass

    # Follow-The-Leader
    (esnmodel_fix_grid_seed_leak__forecast_ens_ftl, 
    esnmodel_fix_grid_seed_leak__weights_ens_ftl) = ftl_comb_for(
        esnmodel_fix_grid_seed_leak__forecast,
        esnmodel_fix_grid_seed_leak__sfe,
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plot_comb_weights(
        #     esnmodel_fix_grid_seed_leak__weights_ens_ftl,
        #     h=1,
        #     method_name="Follow-The-Leader"
        # )
        pass

    # Adaptive Hedge
    (esnmodel_fix_grid_seed_leak__forecast_ens_adahedge, 
    esnmodel_fix_grid_seed_leak__weights_ens_adahedge) = adahedge_comb_for(
        esnmodel_fix_grid_seed_leak__forecast,
        esnmodel_fix_grid_seed_leak__sfe,
    )

    if DO_PLOTS and OPTIONAL_PLOTS:
        # plot_comb_weights(
        #     esnmodel_fix_grid_seed_leak__weights_ens_adahedge,
        #     h=1,
        #     method_name="Adaptive Hedge"
        # )
        pass

    # (Optional) Plot combination methods forecasts against actuals
    if DO_PLOTS and OPTIONAL_PLOTS:
        for h in OUTPUT_STEPS:
            plt.figure(figsize=(10,4))
            plt.plot(
                dataset2007['GDP_data_test'],
                color='black', linewidth=2, label='Target',
            )
            t_idx = list(range(h-1, (dataset2007['GDP_data_test'].shape[0]-FORECAST_STEPS+h)))
            plt.plot(
                dict_results_MSFE_base_models[model_name][0].iloc[t_idx,h-1],
                color='C7', alpha=0.7, linewidth=1.5, label=model_name,
            )
            plt.plot(
                esnmodel_fix_grid_seed_leak__forecast_ens_median.iloc[t_idx,h-1],
                color='C8', alpha=0.7, linewidth=1.5, label='Median',
            )
            plt.plot(
                esnmodel_fix_grid_seed_leak__forecast_ens_sa.iloc[t_idx,h-1],
                color='C0', alpha=0.7, linewidth=1.5, ls=":", label='Simple Average',
            )
            plt.plot(
                esnmodel_fix_grid_seed_leak__forecast_ens_rollmse.iloc[t_idx,h-1],
                color='C1', alpha=0.7, linewidth=1.5, ls=":", label='RollMSE',
            )
            plt.plot(
                esnmodel_fix_grid_seed_leak__forecast_ens_consthedge.iloc[t_idx,h-1],
                color='C2', alpha=0.7, linewidth=1.5, ls="--", label='Const. Hedge',
            )
            plt.plot(
                esnmodel_fix_grid_seed_leak__forecast_ens_dechedge.iloc[t_idx,h-1],
                color='C6', alpha=0.7, linewidth=1.5, ls="--", label='Dec. Hedge',
            )
            plt.plot(
                esnmodel_fix_grid_seed_leak__forecast_ens_ftl.iloc[t_idx,h-1],
                color='C3', alpha=0.7, linewidth=1.5, label='FTL',
            )
            plt.plot(
                esnmodel_fix_grid_seed_leak__forecast_ens_adahedge.iloc[t_idx,h-1],
                color='C4', alpha=0.7, linewidth=1.5, label='AdaHedge',
            )
            plt.title(f"{model_name} Ensemble -- αRP // Forecast vs Targets (h={h})")
            plt.xlabel('Time Index')
            plt.ylabel('GDP Growth Rate')
            plt.legend()
            plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
            plt.show()

    # MSFE of ensemble forecast
    esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_median = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed_leak__forecast_ens_median
    )
    esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_sa = multistepStepMSFE_2007(
        esnmodel_fix_grid_seed_leak__forecast_ens_sa
    )
    esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_rollmse = multistepStepMSFE_2007(
        esnmodel_fix_grid_seed_leak__forecast_ens_rollmse
    )
    esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_consthedge = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed_leak__forecast_ens_consthedge
    )
    esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_dechedge = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed_leak__forecast_ens_dechedge
    )
    esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_ftl = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed_leak__forecast_ens_ftl
    )  
    esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_adahedge = multistepStepMSFE_2007(    
        esnmodel_fix_grid_seed_leak__forecast_ens_adahedge
    )

    # Relative MSFE of ensemble forecast
    esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_median = (
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_median / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_sa = (
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_sa / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_rollmse = (
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_rollmse / ref_MSE_2007_fix
    )   
    esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_consthedge = (
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_consthedge / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_dechedge = (
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_dechedge / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_ftl = (
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_ftl / ref_MSE_2007_fix
    )
    esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_adahedge = (
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_adahedge / ref_MSE_2007_fix
    )

    # ECDF Plot for ensemble forecasts
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_baseline = (
                dict_results_MSFE_base_models[model_name][2].iloc[:,h-1]
            )

            plt.figure(figsize=(5.7,4))
            plt.ecdf(
                x=esnmodel_fix_grid_seed_leak__RelMSE.iloc[:,h-1].to_numpy().astype(float),
                color="#758E85",
                alpha=0.75,
                linewidth=1.6,
            )
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_2007_td_baseline.item(), 
                        color='0.2', linestyle='-', linewidth=1.2, 
                        label=model_name)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_median.iloc[:,h-1].item() , 
                        color='0.2', linestyle=':', linewidth=1.4, 
                        label=f'Median')
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_sa.iloc[:,h-1].item(), 
                        color='C0', linestyle='dashed', linewidth=1.7, 
                        label=f'Average')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_sa.iloc[:,h-1].item(), y=0,
                        color='C0', marker="X", markersize=8, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_rollmse.iloc[:,h-1].item(), 
                        color='C1', linestyle='dashed', linewidth=1.7, 
                        label=f'RollMSE')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_rollmse.iloc[:,h-1].item(), y=0.04, 
                        color='C1', marker="o", markersize=8, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_ftl.iloc[:,h-1].item(), 
                        color='C3', linestyle='dashed', linewidth=1.7, 
                        label=f'FTL')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_ftl.iloc[:,h-1].item(), y=0.,
                        color='C3', marker="^", markersize=8, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_consthedge.iloc[:,h-1].item(), 
                        color='C2', linestyle='dashed', linewidth=1.7, 
                        label=f'Hedge')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_consthedge.iloc[:,h-1].item(), y=0.08,
                        color='C2', marker="s", markersize=8, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_dechedge.iloc[:,h-1].item(), 
                        color='C6', linestyle='dashed', linewidth=1.7, 
                        label=f'DecHedge')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_dechedge.iloc[:,h-1].item(), y=0.12,
                        color='C6', marker="v", markersize=7, alpha=0.5)
            plt.axvline(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_adahedge.iloc[:,h-1].item(),
                        color='C4', linestyle='dashed', linewidth=1.7, 
                        label=f'AdaHedge')
            plt_xaxis_marker(esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_adahedge.iloc[:,h-1].item(), y=0.,
                        color='C4', marker="D", markersize=7, alpha=0.5)
            # plt.xlim(bin_range)
            if h == 1:
                plt.xlim([0.35, 1.15])
            elif h == 2:
                plt.xlim([0.65, 1.25])
            plt.xlabel('Relative MSFE')
            plt.ylabel('Probability')
            plt.legend(fontsize='small', loc="lower right", framealpha=1, edgecolor='0.85', fancybox=False)
            plt.grid(which='major', color="0.85")
            # if SAVE_PLOTS: 
            #     plt.savefig(os.path.join(
            #         PATH_FIGURES, 
            #         'ecdf_ensembles_multistep_singleESN_A_seed_leak_Medi_MD_fix_2007.pdf'
            #     ), bbox_inches="tight")

            if SAVE_PLOTS:
                figname = f'ecdf_{model_name}_multistep_h={h}_seed_leak_MediumMD_fix_2007.pdf' 
                plt.savefig(os.path.join(
                    PATH_FIGURES, 
                    figname,
                ), bbox_inches="tight")
                print(f"[!] Figure saved to figures/ folder as PDF:\n\t{figname}")

            plt.title(f'Relative MSFE ECDF with Ensembles - {model_name}')
            plt.gca().annotate(
                f'Note: {ESN_MATRIX_RESAMPLES} reservoir matrix samples', 
                xy = (1.0, -0.15), xycoords='axes fraction', ha='right', va="center",
                fontsize=8, color="0.3"
            )
            plt.show()

    # Plot ECDF by leak rate
    if DO_PLOTS:
        results_by_leak_list = []
        for res in split_by_leak_rate(results):
            leak_rate = res["leak_rate"]
            forecast_dfs = res["forecast_dfs"]
            msfe_dfs = multistepStepMSFE_ensemble_2007(forecast_dfs)
            rel_msfe = multistepRelativeMSFE_ensemble_2007(
                msfe_dfs, ref_MSE_2007_fix
            )
            results_by_leak_list.append( (leak_rate, rel_msfe) )

        for h in OUTPUT_STEPS:
            plot_msfe_ecdf_by_leak_rate(
                esnmodel_fix_grid_seed_leak__RelMSE.iloc[:, h-1].to_numpy().astype(float), 
                [(r[0], r[1].iloc[:, h-1].to_numpy().astype(float)) for r in results_by_leak_list], 
                model_name=model_name,
            )

    # Rank models by final MSFE
    rank_list = []
    for h in range(1, FORECAST_STEPS+1):
        ranks_h = esnmodel_fix_grid_seed_leak__RelMSE.iloc[:,h-1].rank().to_numpy().astype(int)
        rank_list.append(ranks_h)

    # Plot RollMSE weights over time
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            plot_top_weights_stack(
                esnmodel_fix_grid_seed_leak__weights_ens_rollmse, 
                rank_list, 
                top_num=5, 
                h=h, 
                model_name=model_name, 
                method_name="RollMSE"
            )

    # Plot DecHedge weights over time
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            plot_top_weights_stack(
                esnmodel_fix_grid_seed_leak__weights_ens_dechedge, 
                rank_list, 
                top_num=5, 
                h=h, 
                model_name=model_name, 
                method_name="DecHedge"
            )

    # Plot FTL weights over time
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            plot_top_weights_lines(
                esnmodel_fix_grid_seed_leak__weights_ens_ftl, 
                top_eps=1e-9, 
                h=h, 
                model_name=model_name, 
                method_name="FTL"
            )

    # Plot AdaHedge weights over time
    if DO_PLOTS:
        for h in OUTPUT_STEPS:
            plot_top_weights_stack(
                esnmodel_fix_grid_seed_leak__weights_ens_adahedge, 
                rank_list, 
                top_num=5, 
                h=h, 
                model_name=model_name, 
                method_name="AdaHedge"
            )

    # Joint RMSFE, MSFE and relative MSFE table for ensemble forecasts
    _tmp_table = Table(title=f"Table: {model_name} Ensemble -- αRP [{ESN_MATRIX_RESAMPLES} resamples]", box=box.MINIMAL)
    _tmp_table.add_column("Combination", justify="right", no_wrap=True)
    for h in OUTPUT_STEPS:
        _tmp_table.add_column(f"Rel. MSFE\n h = {h}", justify="left", no_wrap=True)
        # _tmp_table.add_column(f"Rel. MSFE, h = {h}", justify="center", no_wrap=True)
    def _tmp_row_lambda (m, multih_msfe, multih_msfe_toRef, multih_msfe_baseline, steps=OUTPUT_STEPS): 
        idx = [h-1 for h in steps]
        # _tmp_table.add_row(
        #     m, *[f"{v1.round(6)}" for v1 in multih_msfe.to_numpy().flatten()[idx]],
        # )
        # _tmp_table.add_row(
        #     m, *[f"{v2.round(3)}" for v2 in multih_msfe_toRef.to_numpy().flatten()[idx]],
        # )
        _tmp_table.add_row(
            m, *[f"{v2.round(3)} [{v1.round(9)}]" for v2, v1 in zip(
                multih_msfe_toRef.to_numpy().flatten()[idx], multih_msfe.to_numpy().flatten()[idx])
            ],
        )
        multih_perc_change = [
            (
                (multih_msfe_toRef.to_numpy().flatten()[i] - multih_msfe_baseline.to_numpy().flatten()[i])
                / multih_msfe_baseline.to_numpy().flatten()[i]
            ) * 100
            for i in idx
        ]
        _tmp_table.add_row(
            "", *[f"{v.round(2):+}" for v in multih_perc_change],
        )
    _tmp_row_lambda("Baseline",
        dict_results_MSFE_base_models[model_name][1],
        dict_results_MSFE_base_models[model_name][2],
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("Median",
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_median,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_median,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("Simple Average",
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_sa,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_sa,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("RollMSE",
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_rollmse,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_rollmse,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("FTL",
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_ftl,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_ftl,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("Const. Hedge",
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_consthedge,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_consthedge,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("Dec. Hedge",
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_dechedge,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_dechedge,
        dict_results_MSFE_base_models[model_name][2],
    )
    _tmp_row_lambda("AdaHedge",
        esnmodel__MSE_multistep_fix_grid_seed_leak_2007_td_ens_adahedge,
        esnmodel__MSE_toRef_multistep_fix_grid_seed_leak_2007_td_ens_adahedge,
        dict_results_MSFE_base_models[model_name][2],
    )
    console.print(_tmp_table)

#endregion

#endregion

# EOF ------------------------------------------------------
# %%
