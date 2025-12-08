"""
ESN Ensemble Functions

Authors:    andrew-li-yc@github
            giob1994@github
"""

import numpy as np
import pandas as pd

from tqdm import trange

from newToolbox_ESN import ESN, stateMatrixGenerator
from newToolbox_ESN_Multi import ESNMultiFrequency

from data_funcs import normalize_train_test


# region // UTILITY FUNCTIONS

def multistepStepPointSFE(data_df, forecasts_df):
    T = len(forecasts_df)
    steps = len(forecasts_df.columns)
    hStep_p_sfe = np.full((T, steps), np.nan)
    for h, c in enumerate(forecasts_df.columns):
        hStep_p_sfe[:,h] = np.square(
            data_df.to_numpy() - forecasts_df[[c]].to_numpy()
        ).flatten()
    return pd.DataFrame(data=hStep_p_sfe, columns=range(1, 1+steps), index=forecasts_df.index)

# endregion

# region // SINGLE-STATE MFESN

# Different reservoir parameters only

def esnSingle_fix_fit_multistep(esnModel, Lambda, steps, data, direct=False, aggregate=False):
    GDP_fill_data_train = data['GDP_fill_data_train']
    GDP_fill_data_test = data['GDP_fill_data_test']
    if not aggregate:
        md_fill_data_train = data['md_fill_data_train']
        md_fill_data_test = data['md_fill_data_test']
    else:
        md_fill_data_train = data['md_fill_agg_data_train']
        md_fill_data_test = data['md_fill_agg_data_test']

    GDP_test_dates = GDP_fill_data_test.index

    GDP_fill_data_train, GDP_fill_data_test, GDP_mu_train, GDP_sig_train = (
        normalize_train_test(GDP_fill_data_train, GDP_fill_data_test,
            return_mu_sig=True)
    )
    md_fill_data_train, md_fill_data_test = normalize_train_test(md_fill_data_train, md_fill_data_test)

    esnSingle_fit = esnModel.fit(
        Y=GDP_fill_data_train, z=(md_fill_data_train, ), 
        method='ridge',
        Lambda=Lambda,
        full=True,
        debug=False,
        steps=(steps if direct else 1)
    )

    if direct:
        esnSingle_for = esnModel.fixedParamsForecast(
            Yf=GDP_fill_data_test, zf=(md_fill_data_test, ),
            fit=esnSingle_fit,
        )
    else:
        esnSingle_for = esnModel.multistepForecast(
            Yf=GDP_fill_data_test, zf=(md_fill_data_test, ),
            fit=esnSingle_fit,
            steps=steps
        )

    forecast_fix_multistep = np.zeros((len(GDP_test_dates), steps))
    for s in range(steps):
        if direct:
            forecast_fix_multistep[:,s] = np.squeeze(esnSingle_for['Forecast'][s]['Y_for'].to_numpy())
        else:
            forecast_fix_multistep[:,s] = np.squeeze(esnSingle_for['multistepForecast'][s]['Y_for'].to_numpy())

    forecast_fix_multistep = forecast_fix_multistep * GDP_sig_train.to_numpy() + GDP_mu_train.to_numpy()

    return pd.DataFrame(data=forecast_fix_multistep, index=GDP_test_dates, columns=range(steps))

def singleESN_A_multistep_fix_grid_seed(
    input_size,
    reservoir_size,
    dataset_cv,
    dataset_fit_for,
    steps=2,
    seed_list=np.linspace(2, 200, num=100),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    
    # for seed in seed_list:
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

        # Cross-validate
        esnSingle_A_agg_cv10_lambda_2007 = esnSingle_A.ridge_lambda_cv(
            Y=dataset_cv['GDP_fill_data_train'], 
            z=(dataset_cv['md_fill_agg_data_train'], ),
            method="ridge-isotropic",
            cv_options="-cv:10-test_size:5",
            steps=1,
            debug=False,
        )

        # Fit and forecast
        singleESN_A_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
            esnModel=esnSingle_A, 
            Lambda=esnSingle_A_agg_cv10_lambda_2007, 
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
            # print(f"Seed {seed} - Step {s+1}")
            # print(singleESN_A_agg_fix_multistep_for_2007.head())
            # print(singleESN_A_agg_fix_multistep_sfe_2007.head())
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

def singleESN_B_multistep_fix_grid_seed(
    input_size,
    reservoir_size,
    dataset_cv,
    dataset_fit_for,
    steps=2,
    seed_list=np.linspace(2, 200, num=100),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    
    # for seed in seed_list:
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

        # Cross-validate
        esnSingle_B_agg_cv10_lambda_2007 = esnSingle_B.ridge_lambda_cv(
            Y=dataset_cv['GDP_fill_data_train'], 
            z=(dataset_cv['md_fill_agg_data_train'], ),
            method="ridge-isotropic",
            cv_options="-cv:10-test_size:5",
            steps=1,
            debug=False,
        )

        # Fit and forecast
        singleESN_B_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
            esnModel=esnSingle_B, 
            Lambda=esnSingle_B_agg_cv10_lambda_2007, 
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
            # print(f"Seed {seed} - Step {s+1}")
            # print(singleESN_B_agg_fix_multistep_for_2007.head())
            # print(singleESN_B_agg_fix_multistep_sfe_2007.head())
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

# Different reservoir parameters and leak rates

def singleESN_A_multistep_fix_grid_seed_leak(
    input_size,
    reservoir_size,
    dataset_cv,
    dataset_fit_for,
    steps=2,
    seed_list=np.linspace(2, 200, num=100),
    leak_rate_list=[0.5, 0.9],
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    lr_list = []
    
    # for seed in seed_list:
    for i in trange(len(seed_list)):
        seed = seed_list[i]
        for leak_rate in leak_rate_list:
            lr_list.append(leak_rate)

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

            # Cross-validate
            esnSingle_A_agg_cv10_lambda_2007 = esnSingle_A.ridge_lambda_cv(
                Y=dataset_cv['GDP_fill_data_train'], 
                z=(dataset_cv['md_fill_agg_data_train'], ),
                method="ridge-isotropic",
                cv_options="-cv:10-test_size:5",
                steps=1,
                debug=False,
            )

            # Fit and forecast
            singleESN_A_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
                esnModel=esnSingle_A, 
                Lambda=esnSingle_A_agg_cv10_lambda_2007, 
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
                # print(f"Seed {seed} - Step {s+1}")
                # print(singleESN_A_agg_fix_multistep_for_2007.head())
                # print(singleESN_A_agg_fix_multistep_sfe_2007.head())
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

def singleESN_B_multistep_fix_grid_seed_leak(
    input_size,
    reservoir_size,
    dataset_cv,
    dataset_fit_for,
    steps=2,
    seed_list=np.linspace(2, 200, num=100),
    leak_rate_list=[0.5, 0.9],
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    lr_list = []
    
    # for seed in seed_list:
    for i in trange(len(seed_list)):
        seed = seed_list[i]
        for leak_rate in leak_rate_list:
            lr_list.append(leak_rate)

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

            # Cross-validate
            esnSingle_B_agg_cv10_lambda_2007 = esnSingle_B.ridge_lambda_cv(
                Y=dataset_cv['GDP_fill_data_train'], 
                z=(dataset_cv['md_fill_agg_data_train'], ),
                method="ridge-isotropic",
                cv_options="-cv:10-test_size:5",
                steps=1,
                debug=False,
            )

            # Fit and forecast
            singleESN_B_agg_fix_multistep_for_2007 = esnSingle_fix_fit_multistep(
                esnModel=esnSingle_B, 
                Lambda=esnSingle_B_agg_cv10_lambda_2007, 
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
                # print(f"Seed {seed} - Step {s+1}")
                # print(singleESN_B_agg_fix_multistep_for_2007.head())
                # print(singleESN_B_agg_fix_multistep_sfe_2007.head())
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

# Different reservoir parameters only

def esnMulti_fix_fit_multistep(esnModel, Lambda, steps, data, direct=False):
    GDP_data_train = data['GDP_data_train']
    GDP_data_test = data['GDP_data_test']
    m_data_train = data['m_data_train']
    m_data_test = data['m_data_test']
    d_data_train = data['d_data_train']
    d_data_test = data['d_data_test']

    GDP_test_dates = GDP_data_test.index

    GDP_data_train, GDP_data_test, GDP_mu_train, GDP_sig_train = (
        normalize_train_test(GDP_data_train, GDP_data_test,
            return_mu_sig=True)
    )
    m_data_train, m_data_test = normalize_train_test(m_data_train, m_data_test)
    d_data_train, d_data_test = normalize_train_test(d_data_train, d_data_test)

    esnMulti_fit = esnModel.fit(
        Y=GDP_data_train, z=(m_data_train, d_data_train),
        method='ridge',
        Lambda=Lambda,
        full=True,
        debug=False,
        steps=(steps if direct else 1)
    )

    if direct:
        esnMulti_for = esnModel.fixedParamsForecast(
            Yf=GDP_data_test, zf=(m_data_test, d_data_test),
            fit=esnMulti_fit,
        )
    else:
        esnMulti_for = esnModel.multistepForecast(
            Yf=GDP_data_test, zf=(m_data_test, d_data_test),
            fit=esnMulti_fit,
            steps=steps
        )

    forecast_fix_multistep = np.zeros((len(GDP_test_dates), steps))
    for s in range(steps):
        if direct:
            forecast_fix_multistep[:,s] = np.squeeze(esnMulti_for['Forecast'][s]['Y_for'].to_numpy())
        else:
            forecast_fix_multistep[:,s] = np.squeeze(esnMulti_for['multistepForecast'][s]['Y_for'].to_numpy())

    forecast_fix_multistep = forecast_fix_multistep * GDP_sig_train.to_numpy() + GDP_mu_train.to_numpy()

    return pd.DataFrame(data=forecast_fix_multistep, index=GDP_test_dates, columns=range(steps))

def multiESN_A_multistep_fix_grid_seed(
    input_size,
    reservoir_size,
    dataset_cv,
    dataset_fit_for,
    steps=2,
    seed_list=np.linspace(2, 200, num=100),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    
    # for seed in seed_list:
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
                seed=20220623
            ),
            C=stateMatrixGenerator(
                (reservoir_size[1], input_size[1]), 
                dist='sparse_uniform', sparsity=10/reservoir_size[1], normalize='norm2',
                seed=20220623
            ),
            rho=0.5,
            gamma=0.5,
            leak_rate=0.1,
            activation=np.tanh,
        )
        esnMulti_A = ESNMultiFrequency((esn_M_A, esn_D_A), ar=False)

        # Cross-validate
        esnMulti_A_cv10_lambda_2007 = esnMulti_A.ridge_lambda_cv(
            Y=dataset_cv['GDP_fill_data_train'], 
            z=(dataset_cv['m_data_train'], dataset_cv['d_data_train']),
            method="ridge-isotropic",
            cv_options="-cv:10-test_size:5",
            steps=1,
            debug=False,
        )

        # Fit and forecast
        multiESN_A_agg_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
            esnModel=esnMulti_A, 
            Lambda=[esnMulti_A_cv10_lambda_2007[0], esnMulti_A_cv10_lambda_2007[0]], 
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
            # print(f"Seed {seed} - Step {s+1}")
            # print(multiESN_A_agg_fix_multistep_for_2007.head())
            # print(multiESN_A_agg_fix_multistep_sfe_2007.head())
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

def multiESN_B_multistep_fix_grid_seed(
    input_size,
    reservoir_size,
    dataset_cv,
    dataset_fit_for,
    steps=2,
    seed_list=np.linspace(2, 200, num=100),
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    
    # for seed in seed_list:
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

        # Cross-validate
        esnMulti_B_cv10_lambda_2007 = esnMulti_B.ridge_lambda_cv(
            Y=dataset_cv['GDP_fill_data_train'], 
            z=(dataset_cv['m_data_train'], dataset_cv['d_data_train']),
            method="ridge-isotropic",
            cv_options="-cv:10-test_size:5",
            steps=1,
            debug=False,
        )

        # Fit and forecast
        multiESN_B_agg_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
            esnModel=esnMulti_B, 
            Lambda=[esnMulti_B_cv10_lambda_2007[0], esnMulti_B_cv10_lambda_2007[0]], 
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
            # print(f"Seed {seed} - Step {s+1}")
            # print(multiESN_B_agg_fix_multistep_for_2007.head())
            # print(multiESN_B_agg_fix_multistep_sfe_2007.head())
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

# Different reservoir parameters and leak rates

def multiESN_A_multistep_fix_grid_seed_leak(
    input_size,
    reservoir_size,
    dataset_cv,
    dataset_fit_for,
    steps=2,
    seed_list=np.linspace(2, 200, num=100),
    leak_rate_list=[0.5, 0.9],
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    lr_list = []
    
    # for seed in seed_list:
    for i in trange(len(seed_list)):
        seed = seed_list[i]
        for leak_rate in leak_rate_list:
            lr_list.append(leak_rate)

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
                    seed=20220623
                ),
                C=stateMatrixGenerator(
                    (reservoir_size[1], input_size[1]), 
                    dist='sparse_uniform', sparsity=10/reservoir_size[1], normalize='norm2',
                    seed=20220623
                ),
                rho=0.5,
                gamma=0.5,
                leak_rate=leak_rate,
                activation=np.tanh,
            )
            esnMulti_A = ESNMultiFrequency((esn_M_A, esn_D_A), ar=False)

            # Cross-validate
            esnMulti_A_cv10_lambda_2007 = esnMulti_A.ridge_lambda_cv(
                Y=dataset_cv['GDP_fill_data_train'], 
                z=(dataset_cv['m_data_train'], dataset_cv['d_data_train']),
                method="ridge-isotropic",
                cv_options="-cv:10-test_size:5",
                steps=1,
                debug=False,
            )

            # Fit and forecast
            multiESN_A_agg_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
                esnModel=esnMulti_A, 
                Lambda=[esnMulti_A_cv10_lambda_2007[0], esnMulti_A_cv10_lambda_2007[0]], 
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
                # print(f"Seed {seed} - Step {s+1}")
                # print(multiESN_A_agg_fix_multistep_for_2007.head())
                # print(multiESN_A_agg_fix_multistep_sfe_2007.head())
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

def multiESN_B_multistep_fix_grid_seed_leak(
    input_size,
    reservoir_size,
    dataset_cv,
    dataset_fit_for,
    steps=2,
    seed_list=np.linspace(2, 200, num=100),
    leak_rate_list=[0.5, 0.9],
    direct=False,
):
    # change float array to integer array
    seed_list = [int(i) for i in seed_list]
    # Create an empty list to store the individual's forecast points and errors
    forecast_list = [[] for _ in range(steps)]
    sfe_list = [[] for _ in range(steps)]
    lr_list = []
    
    # for seed in seed_list:
    for i in trange(len(seed_list)):
        seed = seed_list[i]
        for leak_rate in leak_rate_list:
            lr_list.append(leak_rate)

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

            # Cross-validate
            esnMulti_B_cv10_lambda_2007 = esnMulti_B.ridge_lambda_cv(
                Y=dataset_cv['GDP_fill_data_train'], 
                z=(dataset_cv['m_data_train'], dataset_cv['d_data_train']),
                method="ridge-isotropic",
                cv_options="-cv:10-test_size:5",
                steps=1,
                debug=False,
            )

            # Fit and forecast
            multiESN_B_agg_fix_multistep_for_2007 = esnMulti_fix_fit_multistep(
                esnModel=esnMulti_B, 
                Lambda=[esnMulti_B_cv10_lambda_2007[0], esnMulti_B_cv10_lambda_2007[0]], 
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
                # print(f"Seed {seed} - Step {s+1}")
                # print(multiESN_B_agg_fix_multistep_for_2007.head())
                # print(multiESN_B_agg_fix_multistep_sfe_2007.head())
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