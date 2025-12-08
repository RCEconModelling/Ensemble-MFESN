"""
Combination Methods

Authors:    andrew-li-yc@github
            giob1994@github
"""

import numpy as np
import pandas as pd

# -----------------------------------------------------------
# UTILITY FUNCTIONS FOR COMBINATION METHODS

def _unpack(data):
    """Unpack input into a numpy array and list of names."""
    if type(data) is list:
        array = np.column_stack(data)
        index = np.arange(array.shape[0])
        names = [f"model_{i+1}" for i in range(len(data))]
    
    elif type(data) is np.ndarray:
        array = data
        index = np.arange(data.shape[0])
        names = [f"model_{i+1}" for i in range(data.shape[1])]

    elif type(data) is pd.DataFrame:
        array = data.to_numpy()
        index = data.index
        names = data.columns.tolist()

    else:
        raise ValueError("Input must be a list, pandas DataFrame, or numpy array.")
    
    return array, index, names

# -----------------------------------------------------------
# COMBINATION METHODS

def median_forecast(forecasts):
    """
    Combination method: Median (MED) method
        Calculates the median forecast by taking the median of multiple forecast series.

    Returns:
        pandas.DataFrame: DataFrame containing the weighted forecast
        numpy.ndarray: Weight matrix used for the weighted forecast
    """
    forecasts_array, forecasts_index, _ = _unpack(forecasts)
    
    T, K = forecasts_array.shape

    # Compute weights and combined forecast
    weights = np.zeros((T, K))
    comb_forecast = np.median(forecasts_array, axis=1)

    for t in range(T):
        median_idx = np.isclose(forecasts_array[t, :], comb_forecast[t], atol=1e-9)
        for k in range(K):
            if median_idx[k]:
                weights[t, k] = 1.0 / np.sum(median_idx)

    comb_forecast_df = pd.DataFrame(
        comb_forecast.reshape(-1, 1), columns=['MEDIAN_Forecast'], index=forecasts_index
    )
    weight_df = pd.DataFrame(
        weights, columns=[f'w_{i+1}' for i in range(K)], index=forecasts_index
    )
    return comb_forecast_df, weight_df

def simple_average_forecast(forecasts):
    """
    Combination method: Simple Average (SA) method
        Calculates the equal-weighted forecast by taking the mean of multiple forecast series.

    Returns:
        pandas.DataFrame: DataFrame containing the weighted forecast
        numpy.ndarray: Weight matrix used for the weighted forecast
    """
    forecasts_array, forecasts_index, _ = _unpack(forecasts)
    
    T, K = forecasts_array.shape

    # Compute weights and combined forecast
    weights = np.full((T, K), 1/K)
    comb_forecast = np.mean(forecasts_array, axis=1)

    comb_forecast_df = pd.DataFrame(
        comb_forecast.reshape(-1, 1), columns=['SA_Forecast'], index=forecasts_index
    )
    weight_df = pd.DataFrame(
        weights, columns=[f'w_{i+1}' for i in range(K)], index=forecasts_index
    )
    return comb_forecast_df, weight_df

def rollMSE_weights_forecast(forecasts, targets, h=1, window_size=4):
    """
    Combination method: Roll MSE (rollMSE) method
    Calculates the weighted forecast based on rolling mean squared forecast errors.

    Returns:
        pandas.DataFrame: DataFrame containing the weighted forecast
        numpy.ndarray: Weight matrix used for the weighted forecast
    """
    forecasts_array, forecasts_index, _ = _unpack(forecasts)
    targets_array, targets_index, _ = _unpack(targets)

    if not np.array_equal(forecasts_index, targets_index):
        raise ValueError("Forecasts and targets must have the same index.")
    
    T, K = forecasts_array.shape
    
    # Compute rolling mean squared forecast errors
    msfe = np.square(forecasts_array - targets_array.reshape(-1, 1))
    rolling_msfe_means = pd.DataFrame(msfe).rolling(window=window_size, min_periods=1).mean().to_numpy()

    # Calculate inverse rolling MSFE
    inverses = 1 / (rolling_msfe_means + 1e-9)  # Add a small constant to avoid division by zero
    norm_rolling_msfe_means = inverses / inverses.sum(axis=1, keepdims=True)

    # Update weights based on past performance
    weights = np.full((T, K), 1/K)
    for t in range(T):
        if t < window_size + h - 1:
            continue
        else:
            weights[t, :] = norm_rolling_msfe_means[t-h, :]

    # Calculate weighted forecast
    comb_forecast = np.sum(forecasts_array * weights, axis=1)

    comb_forecast_df = pd.DataFrame(
        comb_forecast.reshape(-1, 1), columns=['rollMSE_Forecast'], index=forecasts_index)
    weight_df = pd.DataFrame(
        weights, columns=[f'w_{i+1}' for i in range(K)], index=forecasts_index
    )
    return comb_forecast_df, weight_df

def const_hedge_forecast(forecasts, losses, h=1, learning_rate=0.5):
    """
    Combination method: Constant Hedge (ConstHedge) method
    Calculates the weighted forecast using the Constant Hedge algorithm.

    Returns:
        pandas.DataFrame: DataFrame containing the weighted forecast
        numpy.ndarray: Weight matrix used for the weighted forecast
    """
    forecasts_array, forecasts_index, _ = _unpack(forecasts)
    losses_array, losses_index, _ = _unpack(losses)

    if not np.array_equal(forecasts_index, losses_index):
        raise ValueError("Forecasts and losses must have the same index.")
    
    T, K = forecasts_array.shape

    # Calculate iterative weight updates
    update = np.exp(-learning_rate * losses_array)

    # Initialize weights
    weights = np.full((T, K), 1/K)

    # Update weights based on past performance
    for t in range(1, T):
        if t < h:
            continue
        else:
            weights[t, :] = weights[t-h, :] * update[t-h, :]
            weights[t, :] /= np.sum(weights[t, :])  # Normalize weights

    # Calculate weighted forecast
    comb_forecast = np.sum(forecasts_array * weights, axis=1)

    comb_forecast_df = pd.DataFrame(
        comb_forecast.reshape(-1, 1), columns=['ConstHedge_Forecast'], index=forecasts_index)
    weight_df = pd.DataFrame(
        weights, columns=[f'w_{i+1}' for i in range(K)], index=forecasts_index
    )
    return comb_forecast_df, weight_df

def dec_hedge_forecast(forecasts, losses, h=1, c0=2.0):
    """
    Combination method: Decreasing Hedge (DecHedge) method
    Calculates the weighted forecast using the Constant Hedge algorithm.

    Returns:
        pandas.DataFrame: DataFrame containing the weighted forecast
        numpy.ndarray: Weight matrix used for the weighted forecast
    """
    forecasts_array, forecasts_index, _ = _unpack(forecasts)
    losses_array, losses_index, _ = _unpack(losses)

    if not np.array_equal(forecasts_index, losses_index):
        raise ValueError("Forecasts and losses must have the same index.")
    
    T, K = forecasts_array.shape

    # Calculate decreasing learning rates (for default choice, see Chernov and Zhdanov, 2010)
    learning_rates = c0 * np.sqrt(np.log(K) / (np.arange(1, T+1).reshape(-1, 1)))
    
    # Calculate cumulative weight updates
    cumulative_loss = np.cumsum(losses_array, axis=0)
    min_cum_loss = np.min(cumulative_loss, axis=1, keepdims=True)
    adjusted_loss = cumulative_loss - min_cum_loss
    update = np.exp(-learning_rates * adjusted_loss)

    # Initialize weights
    weights = np.full((T, K), 1/K)

    # Update weights based on past performance
    for t in range(1, T):
        if t < h:
            continue
        else:
            weights[t, :] = update[t-h, :]
            weights[t, :] /= np.sum(weights[t, :])  # Normalize weights

    # Calculate weighted forecast
    comb_forecast = np.sum(forecasts_array * weights, axis=1)

    comb_forecast_df = pd.DataFrame(
        comb_forecast.reshape(-1, 1), columns=['DecHedge_Forecast'], index=forecasts_index)
    weight_df = pd.DataFrame(
        weights, columns=[f'w_{i+1}' for i in range(K)], index=forecasts_index
    )
    return comb_forecast_df, weight_df

def ftl_forecast(forecasts, losses, h=1):
    """
    Combination method: Follow The Leader (FTL) method
    Calculates the weighted forecast using the Follow The Leader algorithm.

    Returns:
        pandas.DataFrame: DataFrame containing the weighted forecast
        numpy.ndarray: Weight matrix used for the weighted forecast
    """
    forecasts_array, forecasts_index, _ = _unpack(forecasts)
    losses_array, losses_index, _ = _unpack(losses)

    if not np.array_equal(forecasts_index, losses_index):
        raise ValueError("Forecasts and losses must have the same index.")
    
    T, K = forecasts_array.shape

    # Calculate cumulative weight updates
    cumulative_loss = np.cumsum(losses_array, axis=0)
    min_cum_loss = np.min(cumulative_loss, axis=1, keepdims=True)
    ftl_set = 1.0 * np.isclose(cumulative_loss, min_cum_loss, atol=1e-9)
    update = ftl_set / ftl_set.sum(axis=1, keepdims=True)

    # Initialize weights
    weights = np.full((T, K), 1/K)

    # Update weights based on past performance
    for t in range(1, T):
        if t < h:
            continue
        else:
            weights[t, :] = update[t-h, :]

    # Calculate weighted forecast
    comb_forecast = np.sum(forecasts_array * weights, axis=1)

    comb_forecast_df = pd.DataFrame(
        comb_forecast.reshape(-1, 1), columns=['FTL_Forecast'], index=forecasts_index)
    weight_df = pd.DataFrame(
        weights, columns=[f'w_{i+1}' for i in range(K)], index=forecasts_index
    )
    return comb_forecast_df, weight_df

def adahedge_forecast(forecasts, losses, h=1):
    """
    Combination method: Adaptive Hedge (AdaHedge) method
    Calculates the weighted forecast using the Adaptive Hedge algorithm.

    Returns:
        pandas.DataFrame: DataFrame containing the weighted forecast
        numpy.ndarray: Weight matrix used for the weighted forecast
    """
    forecasts_array, forecasts_index, _ = _unpack(forecasts)
    losses_array, losses_index, _ = _unpack(losses)

    if not np.array_equal(forecasts_index, losses_index):
        raise ValueError("Forecasts and losses must have the same index.")
    
    T, K = forecasts_array.shape

    # Compute adaptive learning rates
    cumulative_loss = np.cumsum(losses_array, axis=0)
    min_cum_loss = np.min(cumulative_loss, axis=1, keepdims=True)

    # Initialize weights and learning rate
    weights = np.full((T, K), 1/K)
    nabla = np.zeros((T, 1))
    M = np.zeros((T, 1))
    learning_rate = np.zeros((T, 1))
    hedge_loss = np.zeros((T, 1))

    # Initial update
    hedge_loss[0] = np.sum(weights[0, :] * losses_array[0, :])
    M[0] = min_cum_loss[0]
    nabla[0] = np.maximum(0, hedge_loss[0] - M[0])  # Initial mixability gap

    # Update weights based on past performance
    for t in range(1, T):
        if t < h:
            # continue
            learning_rate[t] = np.inf
        else:
            if nabla[t-h] == 0:
                # Stay with FTL weights 
                learning_rate[t] = np.inf  # Set to infinity if nabla is zero
                v_0 = 1.0 * np.isclose(cumulative_loss[t-h,:], min_cum_loss[t-h,:], atol=1e-9)
            else:
                # Update learning rate and weights 
                learning_rate[t] = np.log(K) / nabla[t-h]
                v_0 = np.exp(-learning_rate[t] * (cumulative_loss[t-h, :] - min_cum_loss[t-h, :]))
                
            weights[t, :] =  v_0 / np.sum(v_0)

            # Update mixability gap and M (using original update)
            # v_1 = np.exp(-learning_rate * (cumulative_loss[t-h+1, :] - min_cum_loss[t-h+1, :]))
            # w_1 = v_1 / np.sum(v_1)
            # M[t] = min_cum_loss[t] - np.log(np.sum(v_1) / K) / learning_rate
            # l_1 = np.sum(w_1 * losses_array[t, :])
            # nabla[t] = nabla[t-1] + np.maximum(0, (l_1 - (M[t] - M[t-1])))

        # Update mixability gap and M (simple upper bound update, see Lemma 2 in de Rooij et al., 2014)
        hedge_loss[t] = np.sum(weights[t, :] * losses_array[t, :])
        if learning_rate[t] == np.inf:
            M[t] = min_cum_loss[t]
        else:
            M[t] = min_cum_loss[t] - np.log(np.sum(v_0) / K) / learning_rate[t]
        nabla[t] = nabla[t-1] + np.maximum(0, (hedge_loss[t] - (M[t] - M[t-1])))

    # Calculate weighted forecast
    comb_forecast = np.sum(forecasts_array * weights, axis=1)

    comb_forecast_df = pd.DataFrame(
        comb_forecast.reshape(-1, 1), columns=['AdaHedge_Forecast'], index=forecasts_index)
    weight_df = pd.DataFrame(
        weights, columns=[f'w_{i+1}' for i in range(K)], index=forecasts_index
    )
    return comb_forecast_df, weight_df


