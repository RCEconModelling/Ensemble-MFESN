"""
ESN Ensemble Functions with Fixed Ridge Penalties

Authors:    andrew-li-yc@github
            giob1994@github
"""

import numpy as np
import pandas as pd

from tqdm import trange

from newToolbox_ESN import ESN, stateMatrixGenerator
from newToolbox_ESN_Multi import ESNMultiFrequency

from esn_multistep_ensemble_funcs import (
    esnSingle_fix_fit_multistep, esnMulti_fix_fit_multistep, multistepStepPointSFE,
)


# Assigned ridge penalties; no per-expert ridge-penalty tuning.
# Expert order: seed, ridge penalty; or seed, leak rate, ridge penalty.
# The fitting sample and forecast dates are supplied in dataset_fit_for.


# region // SINGLE-STATE MFESN

# Different reservoir parameters and ridge penalties

def singleESN_A_multistep_fix_grid_seed_lambda(
    input_size,
    reservoir_size,
    dataset_fit_for,
    steps=1,
    seed_list=np.linspace(5, 5000, num=200),
    lambda_list=np.logspace(-2, 2, num=5),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]

    for i in trange(len(seed_list)):
        seed = seed_list[i]

        # Create ESN model
        esn_S_A = ESN(
            N=reservoir_size,
            A=stateMatrixGenerator(
                (reservoir_size, reservoir_size),
                dist='sparse_normal', sparsity=10/reservoir_size, normalize='eig',
                seed=seed
            ),
            C=stateMatrixGenerator(
                (reservoir_size, input_size),
                dist='sparse_uniform', sparsity=10/reservoir_size, normalize='norm2',
                seed=seed
            ),
            rho=0.5,
            gamma=1,
            leak_rate=0.1,
            activation=np.tanh,
        )
        esnSingle_A = ESNMultiFrequency((esn_S_A,), ar=False)

        # Retain one expert for each assigned ridge penalty.
        for ridge_lambda in lambda_list:
            # Fit and forecast
            singleESN_A_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
                esnModel=esnSingle_A,
                Lambda=[ridge_lambda],
                data=dataset_fit_for,
                steps=steps,
                direct=direct,
                aggregate=True,
            )

            singleESN_A_agg_fix_multistep_sfe_2007 = multistepStepPointSFE(
                data_df=dataset_fit_for['GDP_data_test'],
                forecasts_df=singleESN_A_agg_fix_multistep_for_2007
            )

            # Change column names to reflect steps
            singleESN_A_agg_fix_multistep_for_2007.columns = list(range(1, 1+steps))

            # Append results to the lists
            for s in range(steps):
                forecast_list[s].append(
                    singleESN_A_agg_fix_multistep_for_2007.iloc[:,s]
                )
                sfe_list[s].append(
                    singleESN_A_agg_fix_multistep_sfe_2007.iloc[:,s]
                )


    # Create DataFrames from the lists
    forecast_dfs = []
    sfe_dfs = []
    for s in range(steps):
        forecast_dfs.append(pd.concat(forecast_list[s], axis=1))
        sfe_dfs.append(pd.concat(sfe_list[s], axis=1))
    # MSFE
    msfe_dfs = []
    for s in range(steps):
        msfe_dfs.append(sfe_dfs[s].mean(axis=0))

    return forecast_dfs, sfe_dfs, msfe_dfs

def singleESN_B_multistep_fix_grid_seed_lambda(
    input_size,
    reservoir_size,
    dataset_fit_for,
    steps=1,
    seed_list=np.linspace(5, 5000, num=200),
    lambda_list=np.logspace(-2, 2, num=5),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]

    for i in trange(len(seed_list)):
        seed = seed_list[i]

        # Create ESN model
        esn_S_B = ESN(
            N=reservoir_size,
            A=stateMatrixGenerator(
                (reservoir_size, reservoir_size),
                dist='sparse_normal', sparsity=10/reservoir_size, normalize='eig',
                seed=seed
            ),
            C=stateMatrixGenerator(
                (reservoir_size, input_size),
                dist='sparse_uniform', sparsity=10/reservoir_size, normalize='norm2',
                seed=seed
            ),
            rho=0.5,
            gamma=1,
            leak_rate=0.1,
            activation=np.tanh,
        )
        esnSingle_B = ESNMultiFrequency((esn_S_B,), ar=False)

        # Retain one expert for each assigned ridge penalty.
        for ridge_lambda in lambda_list:
            # Fit and forecast
            singleESN_B_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
                esnModel=esnSingle_B,
                Lambda=[ridge_lambda],
                data=dataset_fit_for,
                steps=steps,
                direct=direct,
                aggregate=True,
            )

            singleESN_B_agg_fix_multistep_sfe_2007 = multistepStepPointSFE(
                data_df=dataset_fit_for['GDP_data_test'],
                forecasts_df=singleESN_B_agg_fix_multistep_for_2007
            )

            # Change column names to reflect steps
            singleESN_B_agg_fix_multistep_for_2007.columns = list(range(1, 1+steps))
            # Append results to the lists
            for s in range(steps):
                forecast_list[s].append(
                    singleESN_B_agg_fix_multistep_for_2007.iloc[:,s]
                )
                sfe_list[s].append(
                    singleESN_B_agg_fix_multistep_sfe_2007.iloc[:,s]
                )


    # Create DataFrames from the lists
    forecast_dfs = []
    sfe_dfs = []
    for s in range(steps):
        forecast_dfs.append(pd.concat(forecast_list[s], axis=1))
        sfe_dfs.append(pd.concat(sfe_list[s], axis=1))
    # MSFE
    msfe_dfs = []
    for s in range(steps):
        msfe_dfs.append(sfe_dfs[s].mean(axis=0))

    return forecast_dfs, sfe_dfs, msfe_dfs

# Different reservoir parameters, leak rates and ridge penalties


def singleESN_A_multistep_fix_grid_seed_leak_lambda(
    input_size,
    reservoir_size,
    dataset_fit_for,
    steps=1,
    seed_list=np.linspace(5, 5000, num=40),
    leak_rate_list=[0.1, 0.3, 0.5, 0.7, 0.9],
    lambda_list=np.logspace(-2, 2, num=5),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    lr_list = []

    for i in trange(len(seed_list)):
        seed = seed_list[i]
        # Keep the same matrices across leak rates by using the same seed.
        for leak_rate in leak_rate_list:

            # Create ESN model
            esn_S_A = ESN(
                N=reservoir_size,
                A=stateMatrixGenerator(
                    (reservoir_size, reservoir_size),
                    dist='sparse_normal', sparsity=10/reservoir_size, normalize='eig',
                    seed=seed
                ),
                C=stateMatrixGenerator(
                    (reservoir_size, input_size),
                    dist='sparse_uniform', sparsity=10/reservoir_size, normalize='norm2',
                    seed=seed
                ),
                rho=0.5,
                gamma=1,
                leak_rate=leak_rate,
                activation=np.tanh,
            )
            esnSingle_A = ESNMultiFrequency((esn_S_A,), ar=False)

            # Retain one expert for each assigned ridge penalty.
            for ridge_lambda in lambda_list:
                lr_list.append(leak_rate)
                # Fit and forecast
                singleESN_A_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
                    esnModel=esnSingle_A,
                    Lambda=[ridge_lambda],
                    data=dataset_fit_for,
                    steps=steps,
                    direct=direct,
                    aggregate=True,
                )

                singleESN_A_agg_fix_multistep_sfe_2007 = multistepStepPointSFE(
                    data_df=dataset_fit_for['GDP_data_test'],
                    forecasts_df=singleESN_A_agg_fix_multistep_for_2007
                )

                # Change column names to reflect steps
                singleESN_A_agg_fix_multistep_for_2007.columns = list(range(1, 1+steps))

                # Append results to the lists
                for s in range(steps):
                    forecast_list[s].append(
                        singleESN_A_agg_fix_multistep_for_2007.iloc[:,s]
                    )
                    sfe_list[s].append(
                        singleESN_A_agg_fix_multistep_sfe_2007.iloc[:,s]
                    )


    # Create DataFrames from the lists
    forecast_dfs = []
    sfe_dfs = []
    for s in range(steps):
        forecast_dfs.append(pd.concat(forecast_list[s], axis=1))
        sfe_dfs.append(pd.concat(sfe_list[s], axis=1))
    # MSFE
    msfe_dfs = []
    for s in range(steps):
        msfe_dfs.append(sfe_dfs[s].mean(axis=0))

    return forecast_dfs, sfe_dfs, msfe_dfs, lr_list

def singleESN_B_multistep_fix_grid_seed_leak_lambda(
    input_size,
    reservoir_size,
    dataset_fit_for,
    steps=1,
    seed_list=np.linspace(5, 5000, num=40),
    leak_rate_list=[0.1, 0.3, 0.5, 0.7, 0.9],
    lambda_list=np.logspace(-2, 2, num=5),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    lr_list = []

    for i in trange(len(seed_list)):
        seed = seed_list[i]
        # Keep the same matrices across leak rates by using the same seed.
        for leak_rate in leak_rate_list:

            # Create ESN model
            esn_S_B = ESN(
                N=reservoir_size,
                A=stateMatrixGenerator(
                    (reservoir_size, reservoir_size),
                    dist='sparse_normal', sparsity=10/reservoir_size, normalize='eig',
                    seed=seed
                ),
                C=stateMatrixGenerator(
                    (reservoir_size, input_size),
                    dist='sparse_uniform', sparsity=10/reservoir_size, normalize='norm2',
                    seed=seed
                ),
                rho=0.5,
                gamma=1,
                leak_rate=leak_rate,
                activation=np.tanh,
            )
            esnSingle_B = ESNMultiFrequency((esn_S_B,), ar=False)

            # Retain one expert for each assigned ridge penalty.
            for ridge_lambda in lambda_list:
                lr_list.append(leak_rate)
                # Fit and forecast
                singleESN_B_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
                    esnModel=esnSingle_B,
                    Lambda=[ridge_lambda],
                    data=dataset_fit_for,
                    steps=steps,
                    direct=direct,
                    aggregate=True,
                )

                singleESN_B_agg_fix_multistep_sfe_2007 = multistepStepPointSFE(
                    data_df=dataset_fit_for['GDP_data_test'],
                    forecasts_df=singleESN_B_agg_fix_multistep_for_2007
                )

                # Change column names to reflect steps
                singleESN_B_agg_fix_multistep_for_2007.columns = list(range(1, 1+steps))
                # Append results to the lists
                for s in range(steps):
                    forecast_list[s].append(
                        singleESN_B_agg_fix_multistep_for_2007.iloc[:,s]
                    )
                    sfe_list[s].append(
                        singleESN_B_agg_fix_multistep_sfe_2007.iloc[:,s]
                    )


    # Create DataFrames from the lists
    forecast_dfs = []
    sfe_dfs = []
    for s in range(steps):
        forecast_dfs.append(pd.concat(forecast_list[s], axis=1))
        sfe_dfs.append(pd.concat(sfe_list[s], axis=1))
    # MSFE
    msfe_dfs = []
    for s in range(steps):
        msfe_dfs.append(sfe_dfs[s].mean(axis=0))

    return forecast_dfs, sfe_dfs, msfe_dfs, lr_list

# endregion

# region // MULTI-STATE MFESN

# Different reservoir parameters and ridge penalties


def multiESN_A_multistep_fix_grid_seed_lambda(
    input_size,
    reservoir_size,
    dataset_fit_for,
    steps=1,
    seed_list=np.linspace(5, 5000, num=200),
    lambda_list=np.logspace(-2, 2, num=5),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]

    for i in trange(len(seed_list)):
        seed = seed_list[i]

        # Create ESN model
        esn_M_A = ESN(
            N=reservoir_size[0],
            A=stateMatrixGenerator(
                (reservoir_size[0], reservoir_size[0]),
                dist='sparse_normal', sparsity=10/reservoir_size[0], normalize='eig',
                seed=seed
            ),
            C=stateMatrixGenerator(
                (reservoir_size[0], input_size[0]),
                dist='sparse_uniform', sparsity=10/reservoir_size[0], normalize='norm2',
                seed=seed
            ),
            rho=0.5,
            gamma=1.5,
            leak_rate=0,
            activation=np.tanh,
        )

        esn_D_A = ESN(
            N=reservoir_size[1],
            A=stateMatrixGenerator(
                (reservoir_size[1], reservoir_size[1]),
                dist='sparse_normal', sparsity=10/reservoir_size[1], normalize='eig',
                seed=seed
            ),
            C=stateMatrixGenerator(
                (reservoir_size[1], input_size[1]),
                dist='sparse_uniform', sparsity=10/reservoir_size[1], normalize='norm2',
                seed=seed
            ),
            rho=0.5,
            gamma=0.5,
            leak_rate=0.1,
            activation=np.tanh,
        )
        esnMulti_A = ESNMultiFrequency((esn_M_A, esn_D_A), ar=False)

        # Retain one expert for each assigned ridge penalty.
        for ridge_lambda in lambda_list:
            # Fit and forecast
            multiESN_A_agg_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
                esnModel=esnMulti_A,
                Lambda=[ridge_lambda, ridge_lambda],
                data=dataset_fit_for,
                steps=steps,
                direct=direct,
            )

            multiESN_A_agg_fix_multistep_sfe_2007 = multistepStepPointSFE(
                data_df=dataset_fit_for['GDP_data_test'],
                forecasts_df=multiESN_A_agg_fix_multistep_for_2007
            )

            # Change column names to reflect steps
            multiESN_A_agg_fix_multistep_for_2007.columns = list(range(1, 1+steps))
            # Append results to the lists
            for s in range(steps):
                forecast_list[s].append(
                    multiESN_A_agg_fix_multistep_for_2007.iloc[:,s]
                )
                sfe_list[s].append(
                    multiESN_A_agg_fix_multistep_sfe_2007.iloc[:,s]
                )


    # Create DataFrames from the lists
    forecast_dfs = []
    sfe_dfs = []
    for s in range(steps):
        forecast_dfs.append(pd.concat(forecast_list[s], axis=1))
        sfe_dfs.append(pd.concat(sfe_list[s], axis=1))
    # MSFE
    msfe_dfs = []
    for s in range(steps):
        msfe_dfs.append(sfe_dfs[s].mean(axis=0))

    return forecast_dfs, sfe_dfs, msfe_dfs

def multiESN_B_multistep_fix_grid_seed_lambda(
    input_size,
    reservoir_size,
    dataset_fit_for,
    steps=1,
    seed_list=np.linspace(5, 5000, num=200),
    lambda_list=np.logspace(-2, 2, num=5),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]

    for i in trange(len(seed_list)):
        seed = seed_list[i]

        # Create ESN model
        esn_M_B = ESN(
            N=reservoir_size[0],
            A=stateMatrixGenerator(
                (reservoir_size[0], reservoir_size[0]),
                dist='sparse_normal', sparsity=10/reservoir_size[0], normalize='eig',
                seed= seed
            ),
            C=stateMatrixGenerator(
                (reservoir_size[0], input_size[0]),
                dist='sparse_uniform', sparsity=10/reservoir_size[0], normalize='norm2',
                seed= seed
            ),
            rho=0.08,
            gamma=0.25,
            leak_rate=0.3,
            activation=np.tanh,
        )

        esn_D_B = ESN(
            N=reservoir_size[1],
            A=stateMatrixGenerator(
                (reservoir_size[1], reservoir_size[1]),
                dist='sparse_normal', sparsity=10/reservoir_size[1], normalize='eig',
                seed= seed
            ),
            C=stateMatrixGenerator(
                (reservoir_size[1], input_size[1]),
                dist='sparse_uniform', sparsity=10/reservoir_size[1], normalize='norm2',
                seed= seed
            ),
            rho=0.01,
            gamma=0.01,
            leak_rate=0.99,
            activation=np.tanh,
        )
        esnMulti_B = ESNMultiFrequency((esn_M_B, esn_D_B), ar=False)

        # Retain one expert for each assigned ridge penalty.
        for ridge_lambda in lambda_list:
            # Fit and forecast
            multiESN_B_agg_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
                esnModel=esnMulti_B,
                Lambda=[ridge_lambda, ridge_lambda],
                data=dataset_fit_for,
                steps=steps,
                direct=direct,
            )

            multiESN_B_agg_fix_multistep_sfe_2007 = multistepStepPointSFE(
                data_df=dataset_fit_for['GDP_data_test'],
                forecasts_df=multiESN_B_agg_fix_multistep_for_2007
            )

            # Change column names to reflect steps
            multiESN_B_agg_fix_multistep_for_2007.columns = list(range(1, 1+steps))
            # Append results to the lists
            for s in range(steps):
                forecast_list[s].append(
                    multiESN_B_agg_fix_multistep_for_2007.iloc[:,s]
                )
                sfe_list[s].append(
                    multiESN_B_agg_fix_multistep_sfe_2007.iloc[:,s]
                )


    # Create DataFrames from the lists
    forecast_dfs = []
    sfe_dfs = []
    for s in range(steps):
        forecast_dfs.append(pd.concat(forecast_list[s], axis=1))
        sfe_dfs.append(pd.concat(sfe_list[s], axis=1))
    # MSFE
    msfe_dfs = []
    for s in range(steps):
        msfe_dfs.append(sfe_dfs[s].mean(axis=0))

    return forecast_dfs, sfe_dfs, msfe_dfs

# Different reservoir parameters, leak rates and ridge penalties


def multiESN_A_multistep_fix_grid_seed_leak_lambda(
    input_size,
    reservoir_size,
    dataset_fit_for,
    steps=1,
    seed_list=np.linspace(5, 5000, num=40),
    leak_rate_list=[0.1, 0.3, 0.5, 0.7, 0.9],
    lambda_list=np.logspace(-2, 2, num=5),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    lr_list = []

    for i in trange(len(seed_list)):
        seed = seed_list[i]
        # Keep the same matrices across leak rates by using the same seed.
        for leak_rate in leak_rate_list:

            # Create ESN model
            esn_M_A = ESN(
                N=reservoir_size[0],
                A=stateMatrixGenerator(
                    (reservoir_size[0], reservoir_size[0]),
                    dist='sparse_normal', sparsity=10/reservoir_size[0], normalize='eig',
                    seed=seed
                ),
                C=stateMatrixGenerator(
                    (reservoir_size[0], input_size[0]),
                    dist='sparse_uniform', sparsity=10/reservoir_size[0], normalize='norm2',
                    seed=seed
                ),
                rho=0.5,
                gamma=1.5,
                leak_rate=leak_rate,
                activation=np.tanh,
            )

            esn_D_A = ESN(
                N=reservoir_size[1],
                A=stateMatrixGenerator(
                    (reservoir_size[1], reservoir_size[1]),
                    dist='sparse_normal', sparsity=10/reservoir_size[1], normalize='eig',
                    seed=seed
                ),
                C=stateMatrixGenerator(
                    (reservoir_size[1], input_size[1]),
                    dist='sparse_uniform', sparsity=10/reservoir_size[1], normalize='norm2',
                    seed=seed
                ),
                rho=0.5,
                gamma=0.5,
                leak_rate=leak_rate,
                activation=np.tanh,
            )
            esnMulti_A = ESNMultiFrequency((esn_M_A, esn_D_A), ar=False)

            # Retain one expert for each assigned ridge penalty.
            for ridge_lambda in lambda_list:
                lr_list.append(leak_rate)
                # Fit and forecast
                multiESN_A_agg_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
                    esnModel=esnMulti_A,
                    Lambda=[ridge_lambda, ridge_lambda],
                    data=dataset_fit_for,
                    steps=steps,
                    direct=direct,
                )

                multiESN_A_agg_fix_multistep_sfe_2007 = multistepStepPointSFE(
                    data_df=dataset_fit_for['GDP_data_test'],
                    forecasts_df=multiESN_A_agg_fix_multistep_for_2007
                )

                # Change column names to reflect steps
                multiESN_A_agg_fix_multistep_for_2007.columns = list(range(1, 1+steps))
                # Append results to the lists
                for s in range(steps):
                    forecast_list[s].append(
                        multiESN_A_agg_fix_multistep_for_2007.iloc[:,s]
                    )
                    sfe_list[s].append(
                        multiESN_A_agg_fix_multistep_sfe_2007.iloc[:,s]
                    )


    # Create DataFrames from the lists
    forecast_dfs = []
    sfe_dfs = []
    for s in range(steps):
        forecast_dfs.append(pd.concat(forecast_list[s], axis=1))
        sfe_dfs.append(pd.concat(sfe_list[s], axis=1))
    # MSFE
    msfe_dfs = []
    for s in range(steps):
        msfe_dfs.append(sfe_dfs[s].mean(axis=0))

    return forecast_dfs, sfe_dfs, msfe_dfs, lr_list

def multiESN_B_multistep_fix_grid_seed_leak_lambda(
    input_size,
    reservoir_size,
    dataset_fit_for,
    steps=1,
    seed_list=np.linspace(5, 5000, num=40),
    leak_rate_list=[0.1, 0.3, 0.5, 0.7, 0.9],
    lambda_list=np.logspace(-2, 2, num=5),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    lr_list = []

    for i in trange(len(seed_list)):
        seed = seed_list[i]
        # Keep the same matrices across leak rates by using the same seed.
        for leak_rate in leak_rate_list:

            # Create ESN model
            esn_M_B = ESN(
                N=reservoir_size[0],
                A=stateMatrixGenerator(
                    (reservoir_size[0], reservoir_size[0]),
                    dist='sparse_normal', sparsity=10/reservoir_size[0], normalize='eig',
                    seed= seed
                ),
                C=stateMatrixGenerator(
                    (reservoir_size[0], input_size[0]),
                    dist='sparse_uniform', sparsity=10/reservoir_size[0], normalize='norm2',
                    seed= seed
                ),
                rho=0.08,
                gamma=0.25,
                leak_rate=leak_rate,
                activation=np.tanh,
            )

            esn_D_B = ESN(
                N=reservoir_size[1],
                A=stateMatrixGenerator(
                    (reservoir_size[1], reservoir_size[1]),
                    dist='sparse_normal', sparsity=10/reservoir_size[1], normalize='eig',
                    seed= seed
                ),
                C=stateMatrixGenerator(
                    (reservoir_size[1], input_size[1]),
                    dist='sparse_uniform', sparsity=10/reservoir_size[1], normalize='norm2',
                    seed= seed
                ),
                rho=0.01,
                gamma=0.01,
                leak_rate=leak_rate,
                activation=np.tanh,
            )
            esnMulti_B = ESNMultiFrequency((esn_M_B, esn_D_B), ar=False)

            # Retain one expert for each assigned ridge penalty.
            for ridge_lambda in lambda_list:
                lr_list.append(leak_rate)
                # Fit and forecast
                multiESN_B_agg_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
                    esnModel=esnMulti_B,
                    Lambda=[ridge_lambda, ridge_lambda],
                    data=dataset_fit_for,
                    steps=steps,
                    direct=direct,
                )

                multiESN_B_agg_fix_multistep_sfe_2007 = multistepStepPointSFE(
                    data_df=dataset_fit_for['GDP_data_test'],
                    forecasts_df=multiESN_B_agg_fix_multistep_for_2007
                )

                # Change column names to reflect steps
                multiESN_B_agg_fix_multistep_for_2007.columns = list(range(1, 1+steps))
                # Append results to the lists
                for s in range(steps):
                    forecast_list[s].append(
                        multiESN_B_agg_fix_multistep_for_2007.iloc[:,s]
                    )
                    sfe_list[s].append(
                        multiESN_B_agg_fix_multistep_sfe_2007.iloc[:,s]
                    )


    # Create DataFrames from the lists
    forecast_dfs = []
    sfe_dfs = []
    for s in range(steps):
        forecast_dfs.append(pd.concat(forecast_list[s], axis=1))
        sfe_dfs.append(pd.concat(sfe_list[s], axis=1))
    # MSFE
    msfe_dfs = []
    for s in range(steps):
        msfe_dfs.append(sfe_dfs[s].mean(axis=0))

    return forecast_dfs, sfe_dfs, msfe_dfs, lr_list

# endregion
