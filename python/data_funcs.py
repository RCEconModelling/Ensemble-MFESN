"""
Data Functions

Author: giob1994@github
"""

import numpy as np
import pandas as pd

def interp1d2(y0, y1, n):
    x = np.linspace(0, 1, n+2)
    y = y0*(1-x) + y1*x
    return y[1:-1]

def daily_begginingInterp(data, start=None, end=None, length=None, endOfMonth=True):
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    
    start_ = data.index[0] if (start is None) else start
    end_   = data.index[-1] if (end is None) else end

    assert (not length is None) and (length > 0), "Choose a valid interpolation length"

    #start_ -= pd.offsets.MonthBegin(1)
    freqrange = pd.date_range(start=start_, end=end_, freq='MS')
    #if endOfMonth:
    #    freqrange = pd.date_range(start=start_, end=end_, freq='MS')
    #else:
    #    freqrange = pd.date_range(start=start_, end=end_, freq='MS')
    
    data_interp = np.full(length * len(freqrange), np.nan)
    #id_i = np.full(length * len(freqrange), np.nan)
    dates_interp = None

    # Initial case
    data_i0 = data.loc[:freqrange[1]]
    L = len(data_i0)
    data_interp[(length-L+2):length] = data_i0.iloc[1:-1]
    data_interp[0:(length-L+2)] = data_i0.iloc[0]
    data_f_last = data_i0[-2]
    if endOfMonth:
        dates_interp = pd.date_range(
            freqrange[1]-pd.DateOffset(days=length), freqrange[1]-pd.DateOffset(days=1)
        )
    else:
        dates_interp = pd.date_range(
            freqrange[0], freqrange[0]+pd.DateOffset(days=length-1)
        )

    # Full cases
    p = length
    for j in range(2, len(freqrange)):
        data_ij = data.loc[freqrange[j-1]:freqrange[j]]
        L = len(data_ij)

        data_interp[(p+length-L+2):(p+length)] = data_ij.iloc[1:-1]
        data_interp[p:(p+length-L+2)] = interp1d2(
            data_f_last, data_ij[0], length-L+2
        )
        if endOfMonth:
            dates_interp = dates_interp.append(pd.date_range(
                freqrange[j]-pd.DateOffset(days=length), freqrange[j]-pd.DateOffset(days=1)
            ))
        else:
            dates_interp = dates_interp.append(pd.date_range(
                freqrange[j-1], freqrange[j-1]+pd.DateOffset(days=length-1)
            ))

        #print(data_ij)
        #print(data_ij[0])

        #id_i[p:(p+length)] = j
        data_f_last = data_ij[-2]
        p += length

    # Last case
    data_ie = data.loc[freqrange[-1]:(freqrange[-1]+pd.tseries.offsets.MonthBegin())]
    L = len(data_ie)
    data_interp[(p+length-L+1):] = data_ie.iloc[1:]
    data_interp[p:(p+length-L+1)] = interp1d2(
        data_f_last, data_ie[0], length-L+1
    )
    if endOfMonth:
        dates_interp = dates_interp.append(pd.date_range(
            freqrange[-1]+pd.tseries.offsets.MonthEnd()-pd.DateOffset(days=length-1), 
            freqrange[-1]+pd.tseries.offsets.MonthEnd()
        ))
    else:
        dates_interp = dates_interp.append(pd.date_range(
            freqrange[-1], freqrange[-1]+pd.DateOffset(days=length-1)
        ))
    
    #data_interp[0:(length-L)] = interp1d2(d_data_pre[0], d_data_pre[1], 24-L)

    data_interp = pd.DataFrame(data=data_interp, index=dates_interp)

    return data_interp

def daily_aggregateByIndex(data, window_size=1, endOfWindow=True, debug=False):
    assert (not window_size is None) and (window_size > 0), "Choose a valid daily aggregation length"

    data_dates = data.index
    # Get beginning and ends of months over data index
    start_ = data_dates[0] 
    end_   = data_dates[-1]
    month_ends = pd.date_range(start=start_, end=end_, freq='M')
    
    start_j = start_
    exceptions = []

    aggregated_data = pd.DataFrame(data=[], columns=data.columns)
    for end_j in month_ends:
        data_j = data.loc[start_j:end_j]
        j_n = 0
        while j_n < len(data_j):
            slice_j_n = data_j.iloc[j_n:(j_n + window_size)]
            if len(slice_j_n) < window_size:
                exceptions.append(slice_j_n.index.to_period("D"))
            if endOfWindow:
                mean_j_n_index = slice_j_n.index[-1]
            else:
                mean_j_n_index = slice_j_n.index[0]
            aggregated_data.loc[mean_j_n_index] = slice_j_n.mean()
            j_n += window_size

        start_j = end_j + pd.DateOffset(days=1)

    if len(exceptions) > 0:
        print("daily_aggregateByIndex()")
        print(("! Found exceptions in aggregation (some aggregation windows " +
                f"have different length than {window_size})"))
        if debug:
            print("+ Exceptions found:")
            for ex in exceptions: print(ex)

    return aggregated_data

def normalize_train_test(train, test, return_mu_sig=False):
    m_train = train.mean()
    s_train = (train - train.mean()).std()
    if return_mu_sig:
        return (train - m_train)/s_train, (test - m_train)/s_train, m_train, s_train
    else:
        return (train - m_train)/s_train, (test - m_train)/s_train
    
def get_Medium_Datasets(data, startSampleDate, endSampleDate, trainTestSplitDate=None):
    assert not trainTestSplitDate is None, "Choose a slicing data for train/test datasets"

    #trainTestSplitDate = pd.to_datetime('2007-12-31')
    trainTestSplitDate = pd.to_datetime(trainTestSplitDate)

    # Unpack data
    GDP_data = data["GDP_data"]
    m_data = data["m_data"]
    d_data_interp = data["d_data_interp"]
    GDP_fill_data = data["GDP_fill_data"]
    md_fill_data = data["md_fill_data"]
    md_fill_agg_data = data["md_fill_agg_data"]

    # Split training and testing sample
    GDP_data_train = GDP_data.loc[startSampleDate:trainTestSplitDate]
    m_data_train = m_data.loc[startSampleDate:trainTestSplitDate]
    d_data_train = d_data_interp.loc[startSampleDate:trainTestSplitDate]
    GDP_fill_data_train  = GDP_fill_data.loc[startSampleDate:trainTestSplitDate]
    md_fill_data_train = md_fill_data.loc[startSampleDate:trainTestSplitDate]
    md_fill_agg_data_train = md_fill_agg_data.loc[startSampleDate:trainTestSplitDate]

    GDP_data_test = GDP_data.loc[(trainTestSplitDate + pd.offsets.Day()):endSampleDate]
    m_data_test = m_data.loc[(trainTestSplitDate - pd.offsets.MonthBegin(1)):endSampleDate]
    d_data_test = d_data_interp.loc[(trainTestSplitDate - pd.offsets.MonthBegin(1) + pd.offsets.Day(23)):endSampleDate]
    GDP_fill_data_test  = GDP_fill_data.loc[(trainTestSplitDate + pd.offsets.Day()):endSampleDate]
    md_fill_data_test = md_fill_data.loc[(trainTestSplitDate):endSampleDate]
    md_fill_agg_data_test = md_fill_agg_data.loc[(trainTestSplitDate):endSampleDate]

    # Normalize
    # NOTE: if adding normalization here, makes sure models forecasts are 
    #       properly adjusted and comparable
    #GDP_data_train, GDP_data_test = normalize_train_test(GDP_data_train, GDP_data_test)
    #m_data_train, m_data_test = normalize_train_test(m_data_train, m_data_test)
    #d_data_train, d_data_test = normalize_train_test(d_data_train, d_data_test)

    #GDP_fill_data_train, GDP_fill_data_test = normalize_train_test(GDP_fill_data_train, GDP_fill_data_test)
    #md_fill_data_train, md_fill_data_test = normalize_train_test(md_fill_data_train, md_fill_data_test)
    #md_fill_agg_data_train, md_fill_agg_data_test = normalize_train_test(md_fill_agg_data_train, md_fill_agg_data_test)

    # MIDAS data format
    GDP_data_midas = np.vstack((
        GDP_data_train.to_numpy(), 
        GDP_data_test.loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()
    ))
    md_data_midas = (
        tuple(
            np.vstack((
                m_data_train[s].to_numpy()[:,None], 
                m_data_test[s].loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()[:,None]
            )) for s in m_data_train.columns
        ) +
        (np.vstack((
            d_data_train.to_numpy(), 
            d_data_test.loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()
        )), )
    )

    dataset = {
        'GDP_data_train':           GDP_data_train, 
        'GDP_data_test':            GDP_data_test, 
        'm_data_train':             m_data_train, 
        'm_data_test':              m_data_test, 
        'd_data_train':             d_data_train, 
        'd_data_test':              d_data_test,
        'GDP_fill_data_train':      GDP_fill_data_train, 
        'GDP_fill_data_test':       GDP_fill_data_test, 
        'md_fill_data_train':       md_fill_data_train, 
        'md_fill_data_test':        md_fill_data_test,
        'md_fill_agg_data_train':   md_fill_agg_data_train, 
        'md_fill_agg_data_test':    md_fill_agg_data_test,
        'GDP_data_midas':           GDP_data_midas, 
        'md_data_midas':            md_data_midas,
    }
    return dataset

def get_Medium_Datasets_extended(preproc_data, startSampleDate, endSampleDate, trainTestSplitDate=None):
    assert not trainTestSplitDate is None, "Choose a slicing data for train/test datasets"

    startSampleDate = pd.to_datetime(startSampleDate)
    endSampleDate = pd.to_datetime(endSampleDate)
    trainTestSplitDate = pd.to_datetime(trainTestSplitDate)

    data_ext = preproc_data.loc[startSampleDate:endSampleDate]
    # NOTE: unlike previous datasets, can not just always drop NAs, but instead 
    #       must slice data correctly according to date frequencies.

    # (i) Non-filled data
    d_data_ext = data_ext[['D'+str(i) for i in range(1, 14)]].replace(np.NaN, 0.)
    m_data_ext = data_ext[['M'+str(i) for i in range(1, 21)]].loc[
        pd.date_range(startSampleDate, endSampleDate, freq='BM'),
    ].replace(np.NaN, 0.)
    GDP_data_ext = data_ext[['Q1']].dropna()

    # Interpolate non-filled daily data
    d_data_ext_interp = pd.DataFrame()
    for c in d_data_ext.columns:
        d_data_ext_interp = d_data_ext_interp.join(
            daily_begginingInterp(d_data_ext[c], length=24, endOfMonth=False).rename(columns={0: c}),
            how='right',
        )

    # (ii) Filled data
    m_data_ext_prefill = preproc_data[['M'+str(i) for i in range(1, 21)]].loc[
    (startSampleDate - pd.offsets.MonthBegin()):endSampleDate
    ].loc[
        (startSampleDate - pd.offsets.MonthBegin()):endSampleDate
    ].loc[
        pd.date_range((startSampleDate - pd.offsets.MonthBegin()), endSampleDate, freq='BM'),
    ].replace(np.NaN, 0.)
    GDP_data_ext_prefill = preproc_data[['Q1']].loc[
        (startSampleDate - pd.offsets.QuarterBegin()):endSampleDate
    ].dropna()

    # Interpolate non-filled daily data to month end
    d_data_ext_fill_interp = pd.DataFrame()
    for c in d_data_ext.columns:
        d_data_ext_fill_interp = d_data_ext_fill_interp.join(
            daily_begginingInterp(d_data_ext[c], length=24, endOfMonth=True).rename(columns={0: c}),
            how='right',
        )

    # Aggregate interpolated daily to mean over 6 days windows
    d_data_ext_fill_interp_agg = daily_aggregateByIndex(
        d_data_ext_fill_interp, window_size=6, endOfWindow=True
    )

    # Shift monthly, quarterly data to solar end-of-month date
    m_data_ext_prefill.index = m_data_ext_prefill.index + pd.offsets.MonthEnd(0)
    GDP_data_ext_prefill.index = GDP_data_ext_prefill.index + pd.offsets.QuarterEnd(0)

    # Re-align filled data to daily indexes, join dataframe
    pre_index = pd.date_range(
        start=startSampleDate - pd.offsets.YearBegin(), end=endSampleDate, freq='D'
    )
    pre_data_interp = pd.DataFrame(
        data=np.full((len(pre_index), 1), np.nan), index=pre_index
    )

    # Make a new dataframe for filled explanatory variables
    m_data_ext_interp = pre_data_interp.join(m_data_ext_prefill).ffill()
    md_fill_data_ext = d_data_ext_fill_interp.join(m_data_ext_interp.iloc[:,1:])
    md_fill_agg_data_ext = d_data_ext_fill_interp_agg.join(m_data_ext_interp.iloc[:,1:])

    # Shift target data
    GDP_fill_data_ext = GDP_data_ext.copy()
    GDP_fill_data_ext.index = GDP_fill_data_ext.index + pd.offsets.QuarterEnd(0)
    
    # (iii) Split training and testing sample
    GDP_data_train = GDP_data_ext.loc[:trainTestSplitDate]
    m_data_train = m_data_ext.loc[:trainTestSplitDate]
    d_data_train = d_data_ext_interp.loc[:trainTestSplitDate]
    GDP_fill_data_train  = GDP_fill_data_ext.loc[:trainTestSplitDate]
    md_fill_data_train = md_fill_data_ext.loc[:trainTestSplitDate]
    md_fill_agg_data_train = md_fill_agg_data_ext.loc[:trainTestSplitDate]

    GDP_data_test = GDP_data_ext.loc[(trainTestSplitDate + pd.offsets.Day()):]
    m_data_test = m_data_ext.loc[(trainTestSplitDate - pd.offsets.MonthBegin(1)):]
    d_data_test = d_data_ext_interp.loc[(trainTestSplitDate - pd.offsets.MonthBegin(1) + pd.offsets.Day(23)):]
    GDP_fill_data_test  = GDP_fill_data_ext.loc[(trainTestSplitDate + pd.offsets.Day()):]
    md_fill_data_test = md_fill_data_ext.loc[(trainTestSplitDate):]
    md_fill_agg_data_test = md_fill_agg_data_ext.loc[(trainTestSplitDate):]

    # Normalize
    # NOTE: if adding normalization here, makes sure models forecasts are 
    #       properly adjusted and comparable
    #GDP_data_train, GDP_data_test = normalize_train_test(GDP_data_train, GDP_data_test)
    #m_data_train, m_data_test = normalize_train_test(m_data_train, m_data_test)
    #d_data_train, d_data_test = normalize_train_test(d_data_train, d_data_test)

    #GDP_fill_data_train, GDP_fill_data_test = normalize_train_test(GDP_fill_data_train, GDP_fill_data_test)
    #md_fill_data_train, md_fill_data_test = normalize_train_test(md_fill_data_train, md_fill_data_test)
    #md_fill_agg_data_train, md_fill_agg_data_test = normalize_train_test(md_fill_agg_data_train, md_fill_agg_data_test)

    # MIDAS data format
    GDP_data_midas = np.vstack((
        GDP_data_train.to_numpy(), 
        GDP_data_test.loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()
    ))
    md_data_midas = (
        tuple(
            np.vstack((
                m_data_train[s].to_numpy()[:,None], 
                m_data_test[s].loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()[:,None]
            )) for s in m_data_train.columns
        ) +
        (np.vstack((
            d_data_train.to_numpy(), 
            d_data_test.loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()
        )), )
    )

    dataset = {
        'GDP_data_train':           GDP_data_train, 
        'GDP_data_test':            GDP_data_test, 
        'm_data_train':             m_data_train, 
        'm_data_test':              m_data_test, 
        'd_data_train':             d_data_train, 
        'd_data_test':              d_data_test,
        'GDP_fill_data_train':      GDP_fill_data_train, 
        'GDP_fill_data_test':       GDP_fill_data_test, 
        'md_fill_data_train':       md_fill_data_train, 
        'md_fill_data_test':        md_fill_data_test,
        'md_fill_agg_data_train':   md_fill_agg_data_train, 
        'md_fill_agg_data_test':    md_fill_agg_data_test,
        'GDP_data_midas':           GDP_data_midas, 
        'md_data_midas':            md_data_midas,
    }

    return dataset
