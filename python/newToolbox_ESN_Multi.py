# newToolbox_ESN_Multi
#   Multi-frequency ESN updated toolbox
#
# Minor update: July 2024
# ================================================================

from math import floor, ceil, inf
#import datetime as dt
import pandas as pd
import numpy as np
#from numpy import random
import re
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from scipy.optimize import minimize as scipy_minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA

from newToolbox_ESN import ESN
from newToolbox_ESN import matrh, vech
from newToolbox_ESN import esn_data_to_nparray, ls_ridge, jack_ridge, r_ls

# ----------------------------------------------------------------
# Preamble

def closest_past_date(list_date, base_date, cutoff=0):
    """
    list_date: collections of dates to compare.
    base_date: reference date to compate to for closest match.
    """
    #return min([i for i in list_date if i <= base_date], key=lambda x: abs(x - base_date)), 0
    try:
        d_min = min([i for i in list_date[cutoff:] if i <= base_date], key=lambda x: abs(x - base_date))
    except ValueError:
        print(f"Error at: {base_date} with cutoff: {cutoff}")
        raise ValueError
    return d_min, list_date.get_loc(d_min)
    

def closest_future_date(list_date, base_date, cutoff=None):
    #return min([i for i in list_date if i >= base_date], key=lambda x: abs(x - base_date)), 0
    try:
        d_min = min([i for i in list_date[:cutoff] if i >= base_date], key=lambda x: abs(x - base_date))
    except ValueError:
        print(f"Error at: {base_date} with cutoff: {cutoff}")
        raise ValueError
    return d_min, list_date.get_loc(d_min)

def infer_periods(freq, periods=10**4, scale=100):
    """
    freq : str pandas frequency alias.
    periods : numeric, given freq, should create many years. 
    scale: scale of years to group by (century = 100).
    """
    
    while True:
        try:
            s = pd.Series(data=pd.date_range('1970-01-01', freq=freq, periods=periods))
            break
        # If periods is too large
        except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError): 
            periods = periods/10
    
    return s.groupby(s.dt.year // scale * scale).size().value_counts().index[0]

def compare_pandas_freq(f1, f2):
    p1 = infer_periods(f1)
    p2 = infer_periods(f2)
    return (f1 if p1 > p2 else f2)

def _scalpel_loss(t, c=1.0):
    return abs(t) if abs(t) <= abs(c) else t**2 / abs(c)
scalpel_loss = np.vectorize(_scalpel_loss, excluded=['c'])

def _hammer_loss(t, c=1.0):
    return 0 if abs(t) <= abs(c) else (abs(t) - abs(c))**2
hammer_loss = np.vectorize(_hammer_loss, excluded=['c'])

#class ShortTimeSeriesSplit:
#    def __init__(self, split_size=1, n_splits=None):
#        assert split_size > 0
#        self.split_size_ = split_size
#        self.n_splits_   = n_splits
#
#    def split(self, data):
#        assert isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)
#
#        T = len(data)
#        # NOTE: if 'test_size' was set to None, use the last 10% 
#        # of data as testing splits
#        n_splits = self.n_splits_ if (self.n_splits_ > 1) else max((T // 10) // self.split_size_, 1)
#        split_size = self.split_size_
#        min_train_size = T - n_splits * split_size
#
#        train_idxs = []
#        test_idxs  = []
#        for i in range(n_splits):
#            t_i = min_train_size + i * split_size
#            train_idxs.append(tuple(range(0, t_i)))
#            test_idxs.append(tuple(range(t_i, t_i + split_size)))
#
#        return tuple(zip(train_idxs, test_idxs))

class ShiftTimeSeriesSplit:
    def __init__(self, min_split_size, test_size=1, max_split_size=None):
        assert min_split_size > 0
        if not max_split_size is None:
            assert max_split_size > 0, "Maximum split size must be positive"
            assert max_split_size >= min_split_size, "Maximum split size must be greater or equal to minimum split size"

        self.min_split_size_ = min_split_size
        self.test_size_ = test_size
        self.max_split_size_ = max_split_size
        
    def split(self, data):
        #assert isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)

        flag_mss = False
        if not self.max_split_size_ is None:
            flag_mss = True

        T = len(data)
        # Compute number of splits
        n_splits = T - self.min_split_size_ - self.test_size_ + 1

        train_idxs = []
        test_idxs  = []
        t_i = self.min_split_size_
        for i in range(n_splits):
            start_idx = max(0, t_i - self.max_split_size_) if flag_mss else 0
            train_idxs.append(list(range(start_idx, t_i)))
            test_idxs.append(list(range(t_i, t_i + self.test_size_)))
            t_i += 1

        return tuple(zip(train_idxs, test_idxs))

# ----------------------------------------------------------------
# ESN Multi-Frequency Class
#

class ESNMultiFrequency:
    def __init__(self, models, ar=False, states_join='align', states_lags=None):
        self.models_ = tuple(models)
        self.ar_ = (ar is True)
        self.states_join_ = states_join
        self.states_lags_ = states_lags

        # Checks
        for m in self.models_:
            assert isinstance(m, ESN), 'All models must be ESN models'

        assert self.states_join_ in ('align', 'lag_stack'), "State joining specification must be one of: 'align', 'lag_stack'"

        if not self.states_lags_ is None:
            assert len(self.states_lags_) == len(self.models_), 'State lags specification must be of same length as models'
            for l in self.states_lags_:
                assert type(l) is int and l >= 0, 'State lags must be integers and non-negative'

        # Inherited properties
        models_N = []
        for m in self.models_:
            models_N.append(m.N_)
        self.models_N_ = models_N
        self.M_ = sum(models_N)
        if not self.states_lags_ is None:
            self.states_N_ = [int(n * (1 + l)) for n, l in zip(self.models_N_, self.states_lags_)]
        else:
            self.states_N_ = self.models_N_

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 
    def prep_data(self, Y, z):
        Y_dates = Y.index
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")
        assert isinstance(Y_dates, pd.DatetimeIndex)
        
        z_dates = []
        z_ = [] 
        #assert len(z) > 0
        if len(z) > 0:
            for zj in z:
                zj_dates = zj.index
                zj_ = esn_data_to_nparray(zj)
                if zj_ is None:
                    raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")
                assert isinstance(zj_dates, pd.DatetimeIndex)

                z_dates.append(zj_dates)
                z_.append(zj_)
        # Make immutable
        z_dates = tuple(z_dates)
        z_ = tuple(z_)

        return Y_, Y_dates, z_, z_dates

    def multifreq_states(self, Y, z, init, washout_len, Y_dates=None, z_dates=None):
        # States
        X = []
        
        # Y states [if autoregressive]
        #X0 = self.models_[0].base_generate_states(
        #    z=Y, init=init[0], washout_len=washout_len
        #)
        #X0 = pd.DataFrame(X0, index=Y_dates)
        #X.append(X0)

        # z states
        Z = (Y, ) + z if self.ar_ else z
        dates = (Y_dates, ) + z_dates if self.ar_ else z_dates
        for j, zj_ in enumerate(Z):
            Xj = self.models_[j].base_generate_states(
                z=zj_, init=init[j], washout_len=washout_len
            )
            # Add dates to states
            Xj = pd.DataFrame(
                Xj, 
                #columns=tuple(map(lambda x : 'C'+str(x), range(Xj.shape[1]))),
                index=dates[j][washout_len:],
            )
            #
            X.append(Xj)

        return X

    def multifreq_states_to_matrix(self, ref_dates, states, lags=None):
        # Multifrequency state matrix
        X_multi = None

        if self.states_join_ == 'align':
            # High-frequency states are aligned, i.e. for each frequency
            # only the closest past / contemporary state to the low-freuency 
            # target is used as regressor.
            X_multi = np.full((len(ref_dates), self.M_), np.nan)

            p = 0
            for j, Xj in enumerate(states):
                #if j == 0:
                #    X_multi[:,0:self.models_N_[0]] = np.squeeze(Xj)
                #else:
                #    for t, lf_date_t in enumerate(ref_dates):
                #        cpd, _ = closest_past_date(states_dates[j-1], lf_date_t)
                #        X_multi[t,p:(p+self.models_N_[j])] = np.squeeze(Xj.loc[cpd])

                kt = 0
                for t, lf_date_t in enumerate(ref_dates):
                    #cpd, _ = closest_past_date(states_dates[j-1], lf_date_t)
                    cpd, kt = closest_past_date(Xj.index, lf_date_t, cutoff=kt)
                    X_multi[t,p:(p+self.models_N_[j])] = np.squeeze(Xj.loc[cpd])
                #
                p += self.models_N_[j]

        elif self.states_join_ == 'lag_stack':
            # High-frequency states are stacked with lags, i.e. for each frequency
            # the closest past / contemporary state to the low-freuency 
            # target + lagged states are used together as regressors.
            if lags is None:
                lags = np.array(self.states_lags_).astype(int)
            else:
                assert len(lags) == len(self.models_), "State stacking: lags specification not of same length as models"
                for l in lags:
                    assert type(l) is int and l >= 0, 'State lags must be integers and non-negative'

            X_multi = np.full((len(ref_dates), int(sum(self.models_N_ * (1 + lags)))), np.nan)

            p = 0
            for j, Xj in enumerate(states):
                kt = 0
                for t, lf_date_t in enumerate(ref_dates):
                    cpd, kt = closest_past_date(Xj.index, lf_date_t, cutoff=kt)
                    X_multi[t,p:(p+self.models_N_[j])] = np.squeeze(Xj.loc[cpd])
                    # Lags
                    q = p + self.models_N_[j]
                    for l in range(lags[j]):
                        cpd_lag = Xj.index[kt-l-1]
                        X_multi[t,q:(q+self.models_N_[j])] = np.squeeze(Xj.loc[cpd_lag])
                        q += self.models_N_[j]
                #
                p += q # = self.models_N_[j] * int(1 + lags[j])

        else:
            raise ValueError("Multifrequency state joining method not defined")

        return X_multi
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Multi-Frequency Fitting

    def fit(self, Y, z, steps=1, method='ridge', Lambda=0, full=True, Lambda_comps=None, options='',
                init=None, washout_len=0, debug=True):
        # Flatter data and extract datetime indexes
        Y_, Y_dates, z_, z_dates = self.prep_data(Y, z)

        # Check model number vs data
        assert len(self.models_) == int(self.ar_) + len(z), "Number of ESN models does not correspond to number of data series"

        # Decompose options strings
        method_name = re.findall("^\w+(?=-)?", method)
        method_opts = re.findall("(?<=-)\w+(?=:)?", method)
        method_nums = re.findall("(?<=:)(\d+(?:\.\d+)?)", method)

        method = method_name[0]

        if debug:
            print(". Method:")
            print(method_name)
            print(method_opts)
            print(method_nums)

        opts_opts = re.findall("(?<=-)\w+(?=:)?", options)
        opts_nums = re.findall("(?<=:)(\d+(?:\.\d+)?)", options)

        if debug:
            print(". Options:")
            print(opts_opts)
            print(opts_nums)
        
        # States
        init = [None for _ in range(int(self.ar_) + len(z))] if init is None else init
        assert len(init) == len(self.models_), "Length of initialization 'init' must be equal to the number of ESN models"
        
        X = self.multifreq_states(
            Y=Y_, z=z_, Y_dates=Y_dates, z_dates=z_dates,
            init=init, washout_len=0,
        )

        # States matrix
        X_multi = self.multifreq_states_to_matrix(
            ref_dates=Y_dates, states=X,
        )

        # OPTIONAL: reduce state dimensionalty using PCA
        pca = None
        if 'pca' in method_opts:
            pca_factor = float(method_nums[method_opts.index('pca')])
            assert pca_factor > 0
            if pca_factor > 1:
                pca_factor = int(pca_factor)

            # PCA
            pca = PCA(n_components=pca_factor)
            X_multi = pca.fit_transform(X_multi)
            # NOTE: below is the "wrong" time-dimension "PCA"
            #X_multi = pca.fit(X_multi.T).components_.T

            if debug: print(f". PCA: states space size reduced to N = {pca.n_components_}")

        if not method in (None, 'none', 'least_squares', 'r_ls'):
            Lambda_ = np.atleast_1d(np.array(Lambda))
            if pca is None:
                assert len(Lambda_) == len(self.models_), "Lambda is not scalar, must have same length as number of models"
                d = np.zeros(sum(self.states_N_))
                i = 0
                for n, k in enumerate(self.states_N_):
                    d[i:(i+k)] = Lambda[n] * np.ones(k)
                    i += k
                Lambda_ = np.diagflat(d)
            else:
                if not len(Lambda_) == 1:
                    print("PCA requires scalar ridge penalty, selecting only first component of 'Lambda'")
                    Lambda_ = Lambda_[0]

        # Fit
        assert steps > 0

        W = []
        for s in range(steps):
            # Slice matrices
            Ys = Y_[(1+s):,]
            Xs = X_multi[0:(-1-s),]

            Ws = None
            if (method == 'none') or (method is None):
                Ws = np.ones((X.shape[1], 1))   
            elif method == 'least_squares':
                Ws = ls_ridge(Y=Ys, X=Xs, Lambda=0)
            elif method == 'ridge':
                Ws = ls_ridge(Y=Ys, X=Xs, Lambda=Lambda_)
            elif method == 'ridge_jackknife':
                Ws = jack_ridge(Y=Ys, X=Xs, Lambda=Lambda_)
            elif method == 'rls':
                min_train_size = 1
                if 'min_train_size' in opts_opts:
                    min_train_size = int(opts_nums[opts_opts.index('min_train_size')])
                    assert min_train_size >= 0
                V0 = np.hstack((np.ones((min_train_size, 1)), Xs[0:min_train_size,]))
                P0 = np.linalg.pinv(V0.T @ V0)
                W0 = ls_ridge(Y=Ys[0:min_train_size,], X=Xs[0:min_train_size,], Lambda=0)
                Ws, _ = r_ls(Y=Ys[min_train_size:,], X=Xs[min_train_size:,], W0=W0, P0=P0)
                Ws = [W0,] + Ws
            else:
                raise ValueError("Fitting method not defined")

            W.append(Ws)

        # OPTIONAL: Fit full individual autoregressive models 
        # (Necessary for multi-step forecasting)
        W_comps_ar = []
        if full:
            if Lambda_comps is None:
                Lambda_comps = Lambda

            for j, Xj in enumerate(X):
                if self.ar_:
                    Ys_j = Y_[1:,] if j == 0 else z_[j-1][1:,]
                else:
                    Ys_j = z_[j][1:,]
                Xs_j = Xj.iloc[0:-1,].to_numpy()
                # Fit individual weights
                if (method == 'none') or (method is None):
                    W_comps_ar.append(np.ones((Xs_j.shape[1], 1)))
                elif method == 'least_squares':
                    W_comps_ar.append(ls_ridge(Y=Ys_j, X=Xs_j, Lambda=0))
                elif method == 'ridge':
                    W_comps_ar.append(ls_ridge(Y=Ys_j, X=Xs_j, Lambda=Lambda_comps[j]))
                else:
                    raise ValueError("Fitting method not defined")
                #
                #if method == 'ridge':
                #    eps = 1.3
                #    fac = 1
                #    max_eig = np.Inf
                #    l = 0
                #    while max_eig > 1.2 and l <= 1e5:
                #        fac *= eps
                #        W_j = ls_ridge(Y=Ys_j, X=Xs_j, Lambda=(Lambda[j] * fac))
                #        max_eig = np.max(np.abs(np.linalg.eig(
                #            self.models_[j].A_ + self.models_[j].C_ @ W_j[1:,].T
                #        )[0]))
                #        l += 1
                #    W_comps_ar.append(W_j)


        # Fit objects
        esnmulti_fit_for = []
        for s in range(steps):
            # Slice matrices
            X_s     = X_multi[0:(-1-s),]
            V_fit_s = np.hstack((np.ones((X_s.shape[0], 1)), X_s))

            # Compute fits
            if method in ('least_squares', 'ridge', 'ridge_jackknife'):
                Y_fit_s = pd.DataFrame(data=V_fit_s @ W[s], index=Y_dates[(1+s):])
            elif method == 'rls':
                Y_fit_r_ls_s = np.zeros(Y_[(1+s):,].shape)
                for t in range(len(Y_fit_r_ls_s)):
                    if t < min_train_size:
                        Y_fit_r_ls_s[t,] = V_fit_s[t,] @ Ws[0]
                    else:
                        Y_fit_r_ls_s[t,] = V_fit_s[t,] @ Ws[t-min_train_size]
                Y_fit_s = pd.DataFrame(data=Y_fit_r_ls_s, index=Y_dates[(1+s):])
        
            Y_s         = pd.DataFrame(data=Y_[(1+s):,], index=Y_dates[(1+s):])
            Residuals_s = Y_s - Y_fit_s
            RSS_s       = np.sum(Residuals_s.to_numpy() ** 2)

            esnmulti_fit_for.append({
                's':            s+1,
                'W':            W[s],
                'Y_fit':        Y_fit_s,
                'Residuals':    Residuals_s,
                'RSS':          RSS_s,
                'Y':            Y_s,
                'V':            V_fit_s,
                'X':            X_s,
            })

        # Output
        fit_out = {
            'model':        'ESNMultiFrequency',
            'fit_for':      esnmulti_fit_for,
            'fit_comp_ar':  W_comps_ar,
            'steps':        steps,
            'dates':        Y_dates,
            'models':       len(self.models_),
            'model_N':      self.models_N_,
            'method':       method,
            'Lambda':       Lambda,
            'init':         init,
            'state_tuple':  X,
            'pca':          pca,
            #'end_state':    X_multi[-1,],
            'washout_len':  washout_len,
        }

        return fit_out

    def fit_components_ar(self, z, method='ridge', Lambda=0, init=None, washout_len=0):
        """
        Fit a full (i.e. all series) autoregressive model for each indvidual model/dataset
        """

        # Flatter data and extract datetime indexes
        z_dates = []
        z_ = [] 
        #assert len(z) > 0
        if len(z) > 0:
            for zj in z:
                zj_dates = zj.index
                zj_ = esn_data_to_nparray(zj)
                if zj_ is None:
                    raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")
                assert isinstance(zj_dates, pd.DatetimeIndex)

                z_dates.append(zj_dates)
                z_.append(zj_)
        # Make immutable
        z_dates = tuple(z_dates)
        z_ = tuple(z_)

        # Check model number vs data
        assert len(self.models_) == len(z), "Number of ESN models does not correspond to number of data series"
        
        # States
        init = [None for _ in range(len(z))] if init is None else init
        assert len(init) == len(self.models_), "Length of initialization 'init' must be equal to the number of ESN models"

        assert len(Lambda) == len(self.models_), "Lambda is not scalar, must have same length as number of models"

        # Fit
        esnmulti_fit_comp_ar = []
        for j, zj_ in enumerate(z_):
            # States
            Xj = self.models_[j].base_generate_states(
                z=zj_, init=init[j], washout_len=washout_len
            )
            # Regression
            Ys = zj_[1:,]
            Xs = Xj[0:-1,]

            Ws = None
            if (method == 'none') or (method is None):
                Ws = np.ones((Xj.shape[1], 1))   
            elif method == 'ridge':
                Ws = ls_ridge(Y=Ys, X=Xs, Lambda=Lambda[j])
            else:
                raise ValueError("Fitting method not defined")

            # Fit objects
            V_fit_s = np.hstack((np.ones((Xs.shape[0], 1)), Xs))

            Ys          = pd.DataFrame(data=Ys, index=z_dates[j][1:])
            Y_fit_s     = pd.DataFrame(data=V_fit_s @ Ws, index=z_dates[j][1:])
            Residuals_s = Ys - Y_fit_s
            RSS_s       = np.sum(Residuals_s.to_numpy() ** 2)

            esnmulti_fit_comp_ar.append({
                'j':            j,
                'W':            Ws,
                'Y_fit':        Y_fit_s,
                'Residuals':    Residuals_s,
                'RSS':          RSS_s,
                'Y':            Ys,
                'V':            V_fit_s,
                'X':            Xs,
            })

        # Output
        fit_out = {
            'model':        'ESNMultiFrequency_FullComponentsAR',
            'fit_comp_ar':  esnmulti_fit_comp_ar,
            'models':       len(self.models_),
            'model_N':      self.models_N_,
            'method':       method,
            'Lambda':       Lambda,
            'init':         init,
            'washout_len':  washout_len,
        }

        return fit_out


    def fit_multistep(self, Y, z, steps=1, method='ridge', Lambda=0, 
                init=None, washout_len=0, debug=True):
        """
        Fit an ESN multi-frequency model for autonomous multiple-step forecasting. 
        States are first collected, and a "full" target regression is fitted to
        allow for autonomous state iteration. Then for each step (i.e. horizon)
        a specific target weigth matrix is estimated.
        """
        # Flatter data and extract datetime indexes
        Y_, Y_dates, z_, z_dates = self.prep_data(Y, z)

        # Check model number vs data
        assert len(self.models_) == int(self.ar_) + len(z), "Number of ESN models does not correspond to number of data series"

        # States
        init = [None for _ in range(int(self.ar_) + len(z))] if init is None else init
        assert len(init) == len(self.models_), "Length of initialization 'init' must be equal to the number of ESN models"
        
        X = self.multifreq_states(
            Y=Y_, z=z_, Y_dates=Y_dates, z_dates=z_dates,
            init=init, washout_len=0,
        )

        # States matrix
        #X_multi = self.multifreq_states_to_matrix(
        #    ref_dates=Y_dates, states=X,
        #)

        # Ridge penalty for each step
        Lambda_s = []
        if not Lambda is None:
            if len(tuple(Lambda)) == 1:
                Lambda = [np.atleast_1d(np.array(Lambda)) for _ in range(steps)]
            else:
                assert len(Lambda) == steps, f"Lambda must have same length as steps, {steps}, found {len(Lambda)}"
            for j in range(len(Lambda)):
                assert len(Lambda[j]) == len(self.models_), f"Penalty array Lambda[{j}] should have length {len(self.models_)}, found {len(Lambda[j])}"
                d = np.zeros(sum(self.states_N_))
                i = 0
                for n, k in enumerate(self.states_N_):
                    d[i:(i+k)] = (Lambda[j])[n] * np.ones(k)
                    i += k
                Lambda_s.append(np.diagflat(d))
                #print(d)

        # (1) Fit "full" individual reservoir models
        W_comps_full = []
        for j, Xj in enumerate(X):
            if self.ar_:
                Z1s_j = Y_[1:,] if j == 0 else z_[j-1][1:,]
            else:
                Z1s_j = z_[j][1:,]
            X0s_j = Xj.iloc[0:-1,].to_numpy()
            # 
            if (method == 'none') or (method is None):
                W_comps_full.append(np.zeros((X0s_j.shape[1], 1)))
            elif method == 'least_squares':
                W_comps_full.append(ls_ridge(Y=Z1s_j, X=X0s_j, Lambda=0))
            elif method == 'ridge':
                W_comps_full.append(ls_ridge(Y=Z1s_j, X=X0s_j, Lambda=Lambda_s[0][j]))
            else:
                raise ValueError("Fitting method not defined")

        # Fit
        assert steps > 0, "Forecasting steps must be > 0"

        # State reference dates
        ref_dates = Y_dates
        ref_dates_0 = Y_dates

        W_multistep = []
        esnmulti_fit_multistep = []
        for s in range(steps):
            # Construct the correct low-freq state index
            ref_dates_s = ref_dates[0:(-1-s)]

            # States
            X_ms_multi = np.full((len(ref_dates_s), self.M_), np.nan)
            p = 0
            for j, x_j in enumerate(X):
                ks = 0
                for i, d in enumerate(ref_dates_s):
                    cpd, ks = closest_past_date(x_j.index, d, cutoff=ks)
                    # Target date
                    tgd = ref_dates_0[i]
                    # State forward iterations 
                    iter_x_j = len(x_j.loc[cpd:tgd,])
                    # Generate states
                    init_x_j = np.squeeze(x_j.loc[cpd,].to_numpy())
                    Xj, _ = self.models_[j].base_generate_autostates(
                        #T=s+1,
                        T=iter_x_j, 
                        W=W_comps_full[j], init=init_x_j
                    )
                    #
                    #if i == s:
                        #plt.plot(Xj)
                        #plt.show()
                    #
                    X_ms_multi[i,p:(p+self.models_N_[j])] = np.squeeze(Xj[-1,])
                p += self.models_N_[j]

            # Target
            Y_ms = Y_[(1+s):,]

            # Fit
            if (method == 'none') or (method is None):
                W_multistep.append(np.zeros((1 + X_ms_multi.shape[1], 1)))   
            elif method == 'least_squares':
                W_multistep.append(ls_ridge(Y=Y_ms, X=X_ms_multi, Lambda=0))
            elif method == 'ridge':
                W_multistep.append(ls_ridge(Y=Y_ms, X=X_ms_multi, Lambda=Lambda_s[s]))
            else:
                raise ValueError("Fitting method not defined")

            # Compute fit objects
            V_fit_ms     = np.hstack((np.ones((X_ms_multi.shape[0], 1)), X_ms_multi))
            Y_fit_ms     = pd.DataFrame(data=V_fit_ms @ W_multistep[s], index=Y_dates[(1+s):])
            Y_ms         = pd.DataFrame(data=Y_ms, index=Y_dates[(1+s):])
            Residuals_ms = Y_ms - Y_fit_ms
            RSS_ms       = np.sum(Residuals_ms.to_numpy() ** 2)

            esnmulti_fit_multistep.append({
                's':            s+1,
                'W':            W_multistep[s],
                'Y_fit':        Y_fit_ms,
                'Residuals':    Residuals_ms,
                'RSS':          RSS_ms,
                'Y':            Y_ms,
                'V':            V_fit_ms,
                'X':            X_ms_multi,
            })

        # Output
        fit_out = {
            'model':        'ESNMultiFrequency_MultiStep',
            'fit_for':      esnmulti_fit_multistep,
            'fit_comp_ar':  W_comps_full,
            'steps':        steps,
            'dates':        Y_dates,
            'models':       len(self.models_),
            'model_N':      self.models_N_,
            'method':       method,
            'Lambda':       Lambda,
            'init':         init,
            'washout_len':  washout_len,
            'state_tuple':  X,
            'pca':          None,
            #'end_state':    X_multi[-1,],
        }

        return fit_out

    def fit_now(self, Y, z, method='ridge', Lambda=0, init=None, washout_len=0):
        # Flatter data and extract datetime indexes
        Y_, Y_dates, z_, z_dates = self.prep_data(Y, z)

        # Check model number
        assert len(self.models_) == 1 + len(z), "Number of ESN models needs to be exactly 1 + [number of regressors in z]"

        # States
        init = [None for _ in range(1+len(z))] if init is None else init
        assert len(init) == 1 + len(z), "Length of initialization 'init' must be exactly 1 + [number of regressors in z]"

        X = self.multifreq_states(
            Y=Y_, z=z_, Y_dates=Y_dates, z_dates=z_dates,
            init=init, washout_len=0,
        )

        # Add state init as pre-stat
        # NOTE: this is currently a hack, as the date assigned 
        # to pre-states is always '1800-01-01' to make sure they
        # always have the earliest Datetime.
        X_mod = []
        for Xj in X:
            pre_Xj = pd.DataFrame(
                data=(np.zeros(Xj.shape[1]),), columns=Xj.columns,
                index=pd.DatetimeIndex(('1800-01-01',))
            )
            X_mod.append(pre_Xj.append(Xj))
        #
        X = X_mod

        # Find maximal frequency dates
        max_freq = pd.infer_freq(z_dates[0])
        max_periods = infer_periods(max_freq)
        max_freq_dates = z_dates[0]
        for d in z_dates:
            d_f = pd.infer_freq(d)
            p_f = infer_periods(d_f)
            if p_f > max_periods:
                max_freq = d_f
                max_periods = p_f
                max_freq_dates = d

        # States matrix
        #X_multi = self.multifreq_states_to_matrix(
        #    ref_dates=max_freq_dates, states=X, states_dates=z_dates,
        #)

        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Forecasting
    def fixedParamsForecast(self, Yf, zf, fit, steps=None, init=None):
        """
        Given a multi-frequency ESNMulti fit compute forecasts of future
        data (Yf, zf) without updating parameters.
        """

        # Flatter data and extract datetime indexes
        Yf_, Yf_dates, zf_, zf_dates = self.prep_data(Yf, zf)

        # Check model number vs data
        assert len(self.models_) == int(self.ar_) + len(zf), "Number of ESN models does not correspond to number of data series"

        # States
        if init is None:
            init = []
            for_dates = (Yf_dates, ) + zf_dates if self.ar_ else zf_dates
            for j, x_j in enumerate(fit['state_tuple']):
                d = for_dates[j][0]
                cpd, _ = closest_past_date(x_j.index, d - pd.Timedelta(microseconds=1))
                init.append(np.squeeze(x_j.loc[cpd].to_numpy()))

        Xf = self.multifreq_states(
            Y=Yf_, z=zf_, Y_dates=Yf_dates, z_dates=zf_dates,
            init=init, washout_len=0
        )

        # Multifrequency state matrix
        Xf_dates = fit['dates'][[-1]].union(Yf_dates)
        Xf_multi = self.multifreq_states_to_matrix(
            ref_dates=Xf_dates, states=Xf,
        )

        # Stack initial step
        #Xf_multi = np.vstack(((fit['X'])[-1,], Xf_multi[:-1,]))

        # Load training states (necessary for steps > 1)
        V_train = fit['fit_for'][0]['V']

        # PCA
        if not fit['pca'] is None:
            Xf_multi = fit['pca'].transform(Xf_multi)
            # NOTE: below is the "wrong" time-dimension "PCA"
            #pcaf = PCA(n_components=len(fit['fit_for'][0]['W'])-1)
            #Xf_multi = pcaf.fit(Xf_multi.T).components_.T

        # Compute forecast
        steps = steps if (not steps is None) and (int(steps) <= fit['steps']) else fit['steps']
        assert steps > 0, "Forecasting steps must be > 0"

        Forecast = []
        for s in range(steps):
            # Slice matrices
            Xf_s     = Xf_multi[0:(-1-s),]
            Vf_for_s = np.hstack((np.ones((Xf_s.shape[0], 1)), Xf_s))
            if s > 0:
                Vf_for_s = np.vstack((V_train[(-s):,], Vf_for_s))[:len(Yf_dates),]
            
            W_s = fit['fit_for'][s]['W']
            
            #Yf_s     = pd.DataFrame(data=Yf_[s:,], index=Yf_dates[s:])
            #Yf_for_s = pd.DataFrame(data=Vf_for_s @ W_s, index=Yf_dates[s:])
            Yf_s     = Yf
            Yf_for_s = pd.DataFrame(data=Vf_for_s @ W_s, index=Yf_dates)
            Errors_s = Yf_s - Yf_for_s
            FESS_s   = np.sum(Errors_s.to_numpy() ** 2)

            Forecast.append({
                's':            s+1,
                'W':            W_s,
                'Y_for':        Yf_for_s,
                'Errors':       Errors_s,
                'FESS':         FESS_s,
                'Y':            Yf_s,
                'V':            Vf_for_s,
                'X':            Xf_s,
            })

        # Output
        for_out = {
            'model':        'ESNMultiFrequency',
            'Forecast':     Forecast,
            'dates':        Yf_dates,
            'models':       len(self.models_),
            'model_N':      self.models_N_,
            'method':       fit['method'],
            'init':         init,
        }

        return for_out
    
    #New Function- Andrew 
    #A modified function to get the L2 norm of the beta of ridgeless
    def fixedParamsForecast_ridgeless(self, Yf, zf, fit, steps=None, init=None):
        """
        Given a multi-frequency ESNMulti fit compute forecasts of future
        data (Yf, zf) without updating parameters.
        """

        # Flatter data and extract datetime indexes
        Yf_, Yf_dates, zf_, zf_dates = self.prep_data(Yf, zf)

        # Check model number vs data
        assert len(self.models_) == int(self.ar_) + len(zf), "Number of ESN models does not correspond to number of data series"

        # States
        if init is None:
            init = []
            for_dates = (Yf_dates, ) + zf_dates if self.ar_ else zf_dates
            for j, x_j in enumerate(fit['state_tuple']):
                d = for_dates[j][0]
                cpd, _ = closest_past_date(x_j.index, d - pd.Timedelta(microseconds=1))
                init.append(np.squeeze(x_j.loc[cpd].to_numpy()))

        Xf = self.multifreq_states(
            Y=Yf_, z=zf_, Y_dates=Yf_dates, z_dates=zf_dates,
            init=init, washout_len=0
        )

        # Multifrequency state matrix
        Xf_dates = fit['dates'][[-1]].union(Yf_dates)
        Xf_multi = self.multifreq_states_to_matrix(
            ref_dates=Xf_dates, states=Xf,
        )

        # Stack initial step
        #Xf_multi = np.vstack(((fit['X'])[-1,], Xf_multi[:-1,]))

        # Load training states (necessary for steps > 1)
        V_train = fit['fit_for'][0]['V']

        # PCA
        if not fit['pca'] is None:
            Xf_multi = fit['pca'].transform(Xf_multi)
            # NOTE: below is the "wrong" time-dimension "PCA"
            #pcaf = PCA(n_components=len(fit['fit_for'][0]['W'])-1)
            #Xf_multi = pcaf.fit(Xf_multi.T).components_.T

        # Compute forecast
        steps = steps if (not steps is None) and (int(steps) <= fit['steps']) else fit['steps']
        assert steps > 0, "Forecasting steps must be > 0"

        Forecast = []
        for s in range(steps):
            # Slice matrices
            Xf_s     = Xf_multi[0:(-1-s),]
            Vf_for_s = np.hstack((np.ones((Xf_s.shape[0], 1)), Xf_s))
            if s > 0:
                Vf_for_s = np.vstack((V_train[(-s):,], Vf_for_s))[:len(Yf_dates),]

            W_s = fit['fit_for'][s]['W']

            #Yf_s     = pd.DataFrame(data=Yf_[s:,], index=Yf_dates[s:])
            #Yf_for_s = pd.DataFrame(data=Vf_for_s @ W_s, index=Yf_dates[s:])
            Yf_s     = Yf
            Yf_for_s = pd.DataFrame(data=Vf_for_s @ W_s, index=Yf_dates)
            Errors_s = Yf_s - Yf_for_s
            FESS_s   = np.sum(Errors_s.to_numpy() ** 2)

            # Calculate L2 norm of W_s
            L2_norm_W_s = np.linalg.norm(W_s, ord=2)

            Forecast.append({
                's':            s+1,
                'W':            W_s,
                'Y_for':        Yf_for_s,
                'Errors':       Errors_s,
                'FESS':         FESS_s,
                'Y':            Yf_s,
                'V':            Vf_for_s,
                'X':            Xf_s,
                'L2_norm_W_s':  L2_norm_W_s  # Add L2 norm of W_s to the result
            })

        # Output
        for_out = {
            'model':        'ESNMultiFrequency',
            'Forecast':     Forecast,
            'dates':        Yf_dates,
            'models':       len(self.models_),
            'model_N':      self.models_N_,
            'method':       fit['method'],
            'init':         init,
        }

        return for_out
    ####
    
    
    def fixedParamsHighFreqForecast(self, Yf, zf, fit, steps=None, init=None):
        """
        Given a multi-frequency ESNMulti fit compute high-frequency forecasts 
        of future data (Yf, zf) without updating parameters.
        """

        # Flatter data and extract datetime indexes
        Yf_, Yf_dates, zf_, zf_dates = self.prep_data(Yf, zf)

        # Check model number vs data
        assert len(self.models_) == int(self.ar_) + len(zf), "Number of ESN models does not correspond to number of data series"
        
        # States
        X_dates = [] # need to recover fit state dates
        if init is None:
            init = []
            for_dates = (Yf_dates, ) + zf_dates if self.ar_ else zf_dates
            for j, x_j in enumerate(fit['state_tuple']):
                d = for_dates[j][0]
                cpd, _ = closest_past_date(x_j.index, d - pd.Timedelta(microseconds=1))
                init.append(np.squeeze(x_j.loc[cpd].to_numpy()))
                X_dates.append(x_j.index)

        Xf = self.multifreq_states(
            Y=Yf_, z=zf_, Y_dates=Yf_dates, z_dates=zf_dates,
            init=init, washout_len=0
        )

        # Stack training states
        Xf_w_init = []
        for j, xf_j in enumerate(Xf):
            #Xf_w_init.append(fit['state_tuple'][j].iloc[:-1,].append(xf_j))
            Xf_w_init.append(pd.concat([fit['state_tuple'][j].iloc[:-1,], xf_j]))

        # Multifrequency state matrix
        steps = steps if (not steps is None) and (int(steps) <= fit['steps']) else fit['steps']
        assert steps > 0, "Forecasting steps must be > 0"

        # Find maximal frequency dates
        max_freq = pd.infer_freq(X_dates[0])
        max_periods = infer_periods(max_freq)
        max_freq_idx = 0
        for i, d in enumerate(X_dates):
            d_f = pd.infer_freq(d)
            p_f = infer_periods(d_f)
            if p_f > max_periods:
                max_freq = d_f
                max_periods = p_f
                max_freq_idx = i
        max_freq_dates = zf_dates[max_freq_idx]

        # NOTE: we must make a union with the fit high-freq dates 
        max_freq_dates_w_init = (X_dates[max_freq_idx][
            X_dates[max_freq_idx] > fit['dates'][-1-steps]
        ])[:-1].union(max_freq_dates)
        #max_freq_dates_w_init = X_dates[max_freq_idx][:-1].union(max_freq_dates)
        low_freq_dates_w_init = fit['dates'][-1-steps:].union(Yf_dates)

        Xf_multi_hf = self.multifreq_states_to_matrix(
            ref_dates=max_freq_dates_w_init, states=Xf_w_init,
        )

        # Re-add dates to track states and observations easily
        Xf_hf = pd.DataFrame(data=Xf_multi_hf, index=max_freq_dates_w_init)

        # Compute high-frequency forecasts
        # NOTE: we also must identify the max freq dates that are closest to
        #       to low freq target dates for slicing
        max2low_freq_dates_w_init = X_dates[max_freq_idx][[0]]
        kt = 0
        for lf_date in low_freq_dates_w_init:
            cpd, kt = closest_past_date(X_dates[max_freq_idx], lf_date, cutoff=kt)
            max2low_freq_dates_w_init = max2low_freq_dates_w_init.union([cpd])
        max2low_freq_dates_w_init = max2low_freq_dates_w_init[1:]
        
        # NOTE: need to be careful to appropriately repeat low-freq
        # target when making high-freq dataframe
        Yf_hf_dates = Xf_hf.index[
            Xf_hf.index >= closest_past_date(zf_dates[max_freq_idx], Yf_dates[0])[0]
        ]
        Yf_hf_s = pd.DataFrame(columns=[0], index=Yf_dates.union(Yf_hf_dates))
        Yf_hf_s.loc[Yf_dates] = Yf
        # Re-align correctly to high-freq 
        Yf_hf_s = Yf_hf_s.backfill().loc[Yf_hf_dates]

        highFrequencyForecast = []
        for s in range(steps):
            #slice_hs_s = Xf_hf.index[
            #    (Xf_hf.index > fit['dates'][(-2-s)]) & (Xf_hf.index <= low_freq_dates_w_init[(-2-s)])
            #]
            slice_hs_s = Xf_hf.index[
                (Xf_hf.index >= max2low_freq_dates_w_init[(-1-s)]) 
                & 
                (Xf_hf.index <= low_freq_dates_w_init[(-2-s)])
            ]

            # Slice matrices
            Xf_hf_s = Xf_hf.loc[slice_hs_s].to_numpy()
            Vf_hf_for_s = np.hstack((np.ones((Xf_hf_s.shape[0], 1)), Xf_hf_s))

            W_s = fit['fit_for'][s]['W']

            Yf_hf_for_s = pd.DataFrame(data=Vf_hf_for_s @ W_s, index=Yf_hf_dates)
            Errors_s = Yf_hf_s - Yf_hf_for_s
            FESS_s   = np.sum(Errors_s.to_numpy() ** 2)

            highFrequencyForecast.append({
                's':            s+1,
                'W':            W_s,
                'Y_for':        Yf_hf_for_s,
                'Errors':       Errors_s,
                'FESS':         FESS_s,
                'Y':            Yf_hf_s,
                'V':            Vf_hf_for_s,
                'X':            Xf_hf_s,
            })

        # Output
        for_hf_out = {
            'model':                    'ESNMultiFrequency',
            'highFrequencyForecast':    highFrequencyForecast,
            'dates':                    Yf_dates,
            'models':                   len(self.models_),
            'model_N':                  self.models_N_,
            'method':                   fit['method'],
            'init':                     init,
        }

        return for_hf_out

    def fixedParamsKFForecast(self, Yf, zf, fit, steps=None):
        # Flatter data and extract datetime indexes
        Yf_, Yf_dates, zf_, zf_dates = self.prep_data(Yf, zf)

        # Check model number vs data
        assert len(self.models_) == int(self.ar_) + len(zf), "Number of ESN models does not correspond to number of data series"

        # Build "dupe" states, only need associated indexes
        #pre_Xf = self.multifreq_states(
        #    Y=Yf_, z=zf_, Y_dates=Yf_dates, z_dates=zf_dates,
        #    init=[None for _ in range(int(self.ar_) + len(zf))], 
        #    washout_len=0
        #)

        # States indexes
        Xf_multi_idx = []
        for j, zf_dj in enumerate(zf_dates):
            if not self.states_lags_ is None:
                idx_j = np.full((1+self.states_lags_[j], len(Yf_dates)), np.nan)
            else:
                idx_j = np.full((1, len(Yf_dates)), np.nan)
            kt = 0
            for t, lf_date_t in enumerate(Yf_dates):
                _, kt = closest_past_date(zf_dj, lf_date_t, cutoff=kt)
                idx_j[0,t] = kt 
                if not self.states_lags_ is None:
                    for l in range(self.states_lags_[j]):
                        idx_j[1+l,t] = kt-1
                # Add 0 index to accound for forecasting inputs
            idx_j = np.c_[np.atleast_2d(np.zeros(1)), idx_j]
            Xf_multi_idx.append(idx_j.astype(int))

        # MF-EKF filtering
        Zf = (Yf_, ) + zf_ if self.ar_ else zf_

        Yfs1 = Yf_

        Ny = Yf_.shape[1]
        Nz = [zfj_.shape[1] for zfj_ in Zf]
        Tf = len(Yf_dates)

        # Filter initialization from KF fit
        m0 = fit['fitKF_for'][0]['M_t']
        P0 = fit['fitKF_for'][0]['P_t']

        W_a = fit['fitKF_for'][0]['W'][0,:]
        W_w = fit['fitKF_for'][0]['W'][1:,:]

        Sigma_eps = fit['fitKF_for'][0]['Sigma_eps']
        Sigma_eta = fit['fitKF_for'][0]['Sigma_eta']

        # ESN model parameter aliases
        rho_       = [model.rho_ for model in self.models_]
        A_         = [model.A_ for model in self.models_]
        gamma_     = [model.gamma_ for model in self.models_]
        C_         = [model.C_ for model in self.models_]
        zeta_      = [model.zeta_ for model in self.models_]
        leak_rate_ = [model.leak_rate_ for model in self.models_]

        # Multi-Frequency Extended Kalman Filter
        X_prd = np.zeros((self.M_, Tf))
        X_flt = np.zeros((self.M_, Tf))
        L_l_t = np.zeros(Tf)
        LogLike = 0

        # Prediction variables
        M0_t = []
        P0_t = P0.copy()

        # Update variables
        P_t = P0.copy()
        M_t = m0.copy()

        # NOTE: for now, implement only the 'canonical' linearization
        sl = [0 for _ in range(len(zf_))]
        # For each low-frequency period (t-index) from 0 to T-1
        for t in range(Tf):
            m0_t = np.full((self.M_, 1), np.nan)
            # For each reservoir component...
            p = 0
            for j, zfj_ in enumerate(Zf):
                # Prediction: iterate state equations forwards at own freqency (s index)
                m0_s = M_t[j]
                s_range = range(sl[j], Xf_multi_idx[j][0][t]) if (t > 0) else [0,]
                for s in s_range:
                    u0_s = (rho_[j] * A_[j]) @ m0_s + (gamma_[j] * C_[j]) @ zfj_[[s],:].T + zeta_[j]
                    m0_s = leak_rate_[j] * m0_s + (1 - leak_rate_[j]) * np.tanh(u0_s)
                    D_u0_t = 1 - (np.tanh(u0_s) ** 2)
                    F_x = leak_rate_[j] * np.eye(self.models_N_[j]) + (1 - leak_rate_[j]) * D_u0_t * rho_[j] * A_[j]
                    F_q = (1 - leak_rate_[j]) * D_u0_t * gamma_[j] * C_[j]
                    P0_s = F_x @ P_t[j] @ F_x.T + F_q @ np.diagflat(Sigma_eps[j]) @ F_q.T
                    P0_s = (P0_s + P0_s.T) / 2
                m0_t[p:(p+self.models_N_[j]),] = m0_s
                P0_t[j] = P0_s
                p += self.models_N_[j]
                # Move s index forward
                sl[j] = Xf_multi_idx[j][0][t]
            # Update: use only last prediction step (s=0 for all j)
            v_t = Yfs1[[t],:].T - (W_w.T @ m0_t + W_a[:,None])
            S_t = W_w.T @ block_diag(*P0_t) @ W_w + np.diagflat(Sigma_eta)
            K_t = block_diag(*P0_t) @ W_w @ np.linalg.inv(S_t)
            m_u_t = m0_t + K_t @ v_t
            P_u_t = block_diag(*P0_t) - K_t @ S_t @ K_t.T
            P_u_t = (P_u_t + P_u_t.T) / 2
            if np.linalg.norm(P_u_t, ord=np.inf) <= 1e-11:
                P_u_t = 1e-12 * np.eye(self.M_)
            # Slice updates to individual reservoirs
            p = 0
            for j in range(len(Zf)):
                M_t[j] = m_u_t[p:(p+self.models_N_[j]),]
                P_t[j] = P_u_t[p:(p+self.models_N_[j]),p:(p+self.models_N_[j])]
                p += self.models_N_[j]
            # Save states
            X_prd[:,t] = np.squeeze(m0_t)
            X_flt[:,t] = np.squeeze(m_u_t)
            # Compute log-likelihood
            L_l_t[t] = multivariate_normal.pdf(
                np.squeeze(Yfs1[[t],:]), 
                mean=np.squeeze(W_w.T @ m0_t + W_a[:,None]), cov=S_t
            )
        #
        LogLike = np.sum(np.log(L_l_t + 1e-12))

        # Output
        Yf_1        = pd.DataFrame(data=Yfs1, index=Yf_dates)
        Yf_forKF_1  = pd.DataFrame(
                data=np.reshape(W_w.T @ X_prd + W_a[:,None], Yfs1.shape),
                index=Yf_dates,
            )
        Errors_KF_1 = Yf_1 - Yf_forKF_1
        FESS_KF_1   = np.sum(Errors_KF_1 ** 2)

        Forecast =[{
            's':            1,
            'W':            fit['fitKF_for'][0]['W'],
            'Y_for':        Yf_forKF_1,
            'Errors':       Errors_KF_1,
            'FESS':         FESS_KF_1,
            'Y':            Yf_1,
            #'V':            Vf_for_s,
            #'X':            Xf_s,
        }]

        # Output
        forKF_out = {
            'model':        'ESNMultiFrequency',
            'Forecast':     Forecast,
            'dates':        Yf_dates,
            'models':       len(self.models_),
            'model_N':      self.models_N_,
            'method':       fit['method'],
            #'init':         init,
        }

        return forKF_out

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Multi-step Forecasting
    def multistepForecast(self, Yf, zf, fit, steps, init=None):
        """
        Compute multi-step forecasts.
        """
        # NOTE: to compure multi-step forecasts all single-frequency ESNs
        # within a ESNMultiFrequency object must be additionally trained 
        # to fit *all* input components. This is necessary to ensure the
        # state equations can be run autonomously.

        # Check fit object
        assert 'fit_comp_ar' in fit

        # Flatter data and extract datetime indexes
        Yf_, Yf_dates, zf_, zf_dates = self.prep_data(Yf, zf)

        # Check model number vs data
        assert len(self.models_) == int(self.ar_) + len(zf), "Number of ESN models does not correspond to number of data series"

        # States
        if init is None:
            init = []
            for_dates = (Yf_dates, ) + zf_dates if self.ar_ else zf_dates
            for j, x_j in enumerate(fit['state_tuple']):
                d = for_dates[j][0]
                cpd, _ = closest_past_date(x_j.index, d - pd.Timedelta(microseconds=1))
                init.append(np.squeeze(x_j.loc[cpd].to_numpy()))

        Xf = self.multifreq_states(
            Y=Yf_, z=zf_, Y_dates=Yf_dates, z_dates=zf_dates,
            init=init, washout_len=0
        )

        # Infer regressors frequencies
        #Xf_freq = []
        #for xf_j in Xf:
        #    Xf_freq.append(pd.infer_freq(xf_j.index))

        # Infer Yf offsets to date multi-steps forecasts
        #Yf_freq = pd.infer_freq(Yf_dates)
        #Yf_date_offset = pd.tseries.frequencies.to_offset(Yf_freq)

        # Stack training states
        # NOTE: to produce a consistent multistep forecast (exactly 'steps'
        # forecasts for each target in the 'Yf' argument), we additionally
        # stack the training states to the testing states
        Xf_w_init = []
        for j, xf_j in enumerate(Xf):
            #Xf_w_init.append(fit['state_tuple'][j].iloc[:-1,].append(xf_j))
            Xf_w_init.append(pd.concat([fit['state_tuple'][j].iloc[:-1,], xf_j]))

        # Compute multi-step forecast
        #steps = steps if (not steps is None) and (int(steps) <= fit['steps']) else fit['steps']
        assert steps > 0, "Forecasting steps must be > 0"

        # State reference dates
        ref_dates = fit['dates'].append(Yf_dates)
        ref_index = len(fit['dates']) - 1
        ref_dates_0 = ref_dates[ref_index:]

        multistepForecast = []
        for s in range(steps):
            # Construct the correct low-freq state index
            #ref_dates_s = fit['dates'][(-1-s):].append(Yf_dates[0:(-1-s)])
            ref_dates_s = ref_dates[(ref_index - s):(-1-s)]

            # States
            #X_ms = []
            X_ms_multi = np.full((len(ref_dates_s), self.M_), np.nan)
            p = 0
            for j, xf_j in enumerate(Xf_w_init):
                ks = 0
                for i, d in enumerate(ref_dates_s):
                    cpd, ks = closest_past_date(xf_j.index, d, cutoff=ks)
                    # Target date
                    tgd = ref_dates_0[i]
                    # State forward iterations 
                    iter_xf_j = len(xf_j.loc[cpd:tgd,])
                    # Generate states
                    init_xf_j = np.squeeze(xf_j.loc[cpd,].to_numpy())
                    Xj, _ = self.models_[j].base_generate_autostates(
                        #T=s+1,
                        T=iter_xf_j, 
                        W=fit['fit_comp_ar'][j], init=init_xf_j
                    )
                    #
                    #if False:
                    #if s == 2:
                    #if i == s and s > 0:
                    #    plt.figure(figsize=(5, 2))
                    #    plt.plot(Xj)
                    #    plt.show()
                    #
                    X_ms_multi[i,p:(p+self.models_N_[j])] = np.squeeze(Xj[-1,])
                p += self.models_N_[j]

            # PCA
            if not fit['pca'] is None:
                X_ms_multi = fit['pca'].transform(X_ms_multi)

            # Load coefficients
            if fit['model'] == 'ESNMultiFrequency':
                Ws = fit['fit_for'][0]['W']
            elif fit['model'] == 'ESNMultiFrequency_MultiStep':
                Ws = fit['fit_for'][s]['W']
            else:
                raise TypeError("Model type of fit object not recognized")
            
            # Forecasts
            V_ms_s = np.hstack((np.ones((X_ms_multi.shape[0], 1)), X_ms_multi))
            Y_ms_for_s = pd.DataFrame(data=V_ms_s @ Ws, index=Yf_dates)
            
            multistepForecast.append({
                's':            s+1,
                'ref_dates':    ref_dates_s,
                'Y_for':        Y_ms_for_s,
                'V':            V_ms_s,
                'X':            X_ms_multi,
                #'state_tuple':  X_ms,
            })

        
        #multistepForecast = []
        #for t, d in enumerate(Yf_dates[1:]):
        #    d_plus_steps = d + (steps-1)*Yf_date_offset
        #    ref_dates_t = pd.date_range(d, d_plus_steps, freq=Yf_freq)

        #    # States
        #    X_ms       = []
        #    X_ms_multi = np.full((len(ref_dates_t), self.M_), np.nan)
        #    p = 0
        #    for j, xf_j in enumerate(Xf_w_init):
        #        # Compute dates range
        #        d_range_j = pd.date_range(d, d_plus_steps, freq=Xf_freq[j])
        #        # NOTE: this is a hack to make sure the cdp is strictly in the past
        #        cpd, _ = closest_past_date(xf_j.index, d - pd.Timedelta(microseconds=1))
        #        # Generate states
        #        init_xf_j = np.squeeze(xf_j.loc[cpd,].to_numpy())
        #        Xj, _ = self.models_[j].base_generate_autostates(
        #            T=len(d_range_j), 
        #            W=fit['fit_comp_ar'][j], 
        #            init=init_xf_j
        #        )
        #        # Add dates to states
        #        Xj = pd.DataFrame(
        #            Xj, columns=tuple(map(lambda x : 'C'+str(x), range(Xj.shape[1]))),
        #            index=d_range_j
        #        )
        #        X_ms.append(Xj)
        #        #
        #        ks = 0
        #        for s, lf_date_t in enumerate(ref_dates_t):
        #            #cpd, _ = closest_past_date(states_dates[j-1], lf_date_t)
        #            cpd, ks = closest_past_date(Xj.index, lf_date_t, cutoff=ks)
        #            X_ms_multi[s,p:(p+self.models_N_[j])] = np.squeeze(Xj.loc[cpd])
        #        #
        #        p += self.models_N_[j]

        #    # PCA
        #    if not fit['pca'] is None:
        #        X_ms_multi = fit['pca'].transform(X_ms_multi)
        #    
        #    # Forecasts
        #    V_ms_t = np.hstack((np.ones((X_ms_multi.shape[0], 1)), X_ms_multi))
        #    Y_ms_for_t = pd.DataFrame(data=V_ms_t @ W0, index=ref_dates_t)
        #    
        #    multistepForecast.append({
        #        't':            t,
        #        'start_date':   d,
        #        'Y_for':        Y_ms_for_t,
        #        'V':            V_ms_t,
        #        'X':            X_ms_multi,
        #        'state_tuple':  X_ms,
        #    })
        
        
        # Output
        for_out = {
            'model':                'ESNMultiFrequency',
            'multistepForecast':    multistepForecast,
            'Yf':                   Yf,
            'models':               len(self.models_),
            'model_N':              self.models_N_,
            'method':               fit['method'],
            'init':                 init,
        }

        return for_out

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Nowcasting
    def fixedParamsNowcast(self, Yf, zf, fit, steps=None, init=None, debug=False):
        if True:
            """
            !!!
            TODO: need to correctly implement fit_now() method, because below 
            we are not computing the nowcast but a high-frequency forecast
            """
            print("Error: method needs to be updated!")
            return None

        # Flatter data and extract datetime indexes
        Yf_, Yf_dates, zf_, zf_dates = self.prep_data(Yf, zf)

        # Check model number vs data
        assert len(self.models_) == int(self.ar_) + len(zf), "Number of ESN models does not correspond to number of data series"

        # Find maximal frequency dates
        max_freq = pd.infer_freq(zf_dates[0])
        max_periods = infer_periods(max_freq)
        max_freq_dates = zf_dates[0]
        for d in zf_dates:
            d_f = pd.infer_freq(d)
            p_f = infer_periods(d_f)
            if p_f > max_periods:
                max_freq = d_f
                max_periods = p_f
                max_freq_dates = d

        # States
        #if init is None:
        #    init = []
        #    for Xj in fit['state_tuple']:
        #        init.append(np.squeeze(Xj.iloc[-1,].to_numpy()))

        if init is None:
            init = []
            init_dates = (Yf_dates, ) + zf_dates if self.ar_ else zf_dates
            for j, x_j in enumerate(fit['state_tuple']):
                d = init_dates[j][0]
                cpd, _ = closest_past_date(x_j.index, d - pd.Timedelta(microseconds=1))
                init.append(np.squeeze(x_j.loc[cpd].to_numpy()))

        Xf = self.multifreq_states(
            Y=Yf_, z=zf_, Y_dates=Yf_dates, z_dates=zf_dates,
            init=init, washout_len=0
        )

        # Add state init as pre-stat
        #Xf_w_init = []
        #for j, Xfj in enumerate(Xf):
        #    Xf_w_init.append(
        #        fit['state_tuple'][j].iloc[-1:,].append(Xfj)
        #    )
        
        # States matrix
        Xf_multi_hf = self.multifreq_states_to_matrix(
            ref_dates=max_freq_dates, states=Xf,
        )
        
        # Re-add dates to track states and observations easily
        Yf_    = pd.DataFrame(data=Yf_, index=Yf_dates)
        Xf_now = pd.DataFrame(data=Xf_multi_hf, index=max_freq_dates)

        # Compute nowcast
        W_n = fit['fit_for'][0]['W']

        Y_now  = pd.DataFrame(data=np.full(len(max_freq_dates), np.nan), index=max_freq_dates)
        Y      = Y_now.copy()
        Errors = Y_now.copy()

        for d in max_freq_dates:
            Xf_now_d = Xf_now.loc[d].to_numpy().reshape(1, -1)

            # PCA
            if not fit['pca'] is None:
                Xf_now_d = fit['pca'].transform(Xf_now_d)

            cfd, _ = closest_future_date(Yf_dates, d)

            Y_now_d = np.hstack((np.ones((1, 1)), Xf_now_d)) @ W_n
            Y_d     = Yf_.loc[cfd]
            Error_d = Y_d - np.squeeze(Y_now_d)
            
            Y_now.loc[d] = Y_now_d
            Y.loc[d] = Y_d
            Errors.loc[d] = Error_d
        NESS = np.sum(Errors.to_numpy() ** 2)

        Nowcast = [{
            'Y_now':        Y_now,
            'Errors':       Errors,
            'NESS':         NESS,
            'Y':            Y,
            'V':            Y_now,
            'X':            Xf_multi_hf, 
        }]

        # Output
        now_out = {
            'model':        'ESNMultiFrequency',
            'Nowcast':      Nowcast,
            'dates':        max_freq_dates,
            'models':       len(self.models_),
            'model_N':      self.models_N_,
            'method':       fit['method'],
            'init':         init,
        }

        return now_out

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Multi-frequency ridge penalty cross-validation
        
    def ridge_lambda_cv(
        self, 
        Y, 
        z, 
        method='ridge', 
        cv_options='-cv:5', 
        steps=1, 
        Lambda0=1e-1,
        steps_weights=None, 
        Lambda_comps=None, 
        init=None, 
        washout_len=0, 
        debug=True
    ):
        if debug: print("ESNMultiFrequency.ridge_lambda_cv()")

        # Flatter data and extract datetime indexes
        Y_, Y_dates, z_, z_dates = self.prep_data(Y, z)

        # Decompose optimization type string
        method_name = re.findall("^\w+(?=-)?", method)
        method_opts = re.findall("(?<=-)\w+(?=:)?", method)
        method_nums = re.findall("(?<=:)(\d+(?:\.\d+)?)", method)

        assert method_name[0] == 'ridge', "Only supported method is 'ridge'"

        if debug:
            print(". Method:")
            print(method_name)
            print(method_opts)
            print(method_nums)

        cv_opts = re.findall("(?<=-)\w+(?=:)?", cv_options)
        cv_nums = re.findall("(?<=:)(\d+(?:\.\d+)?)", cv_options)

        if debug:
            print(". CV Options:")
            print(cv_opts)
            print(cv_nums)

        # States
        init = [None for _ in range(int(self.ar_) + len(z))] if init is None else init
        assert len(init) == len(self.models_), "Length of initialization 'init' must be equal to the number of ESN models"
        
        X = self.multifreq_states(
            Y=Y_, z=z_, Y_dates=Y_dates, z_dates=z_dates,
            init=init, washout_len=washout_len,
        )

        # States matrix
        X_multi = self.multifreq_states_to_matrix(
            ref_dates=Y_dates, states=X,
        )

        # OPTIONAL: reduce state dimensionalty using PCA
        pca = None
        if 'pca' in method_opts:
            pca_factor = float(method_nums[method_opts.index('pca')])
            assert pca_factor > 0
            if pca_factor > 1:
                pca_factor = int(pca_factor)

            # PCA
            pca = PCA(n_components=pca_factor)
            X_multi = pca.fit_transform(X_multi)
            # NOTE: below is the "wrong" time-dimension "PCA"
            #X_multi = pca.fit(X_multi.T).components_.T

            if debug: print(f". PCA: states space size reduced to N = {pca.n_components_}")

        # Fallback parameters
        cv_splits = 2
        test_size = None
        cv_min_split_size = None
        cv_max_split_size = None
        if 'cv' in cv_opts:    
            cv_splits = int(cv_nums[cv_opts.index('cv')])
            assert cv_splits >= 0
        if 'test_size' in cv_opts:
            test_size = int(cv_nums[cv_opts.index('test_size')])
            assert test_size >= 0
        if 'cv_min_split_size' in cv_opts:
            if 'cv' in cv_opts:
                print("[!] Option cv_min_split_size overwrites baseline CV folds option")
            cv_min_split_size = int(cv_nums[cv_opts.index('cv_min_split_size')])
            assert cv_min_split_size >= 0
        if 'cv_max_split_size' in cv_opts:
            cv_max_split_size = int(cv_nums[cv_opts.index('cv_max_split_size')])
            assert cv_max_split_size >= 0

        # CV method variations
        isotropic_cv = False
        multistep_cv = False
        shift_cv = False
        if 'isotropic' in method_opts:
            isotropic_cv = True
        if 'multistep' in method_opts:
            multistep_cv = True
            if test_size is None:
                test_size = steps
            elif test_size > steps:
                print("[!] Multi-step ridge penalty cross-validation, alert:")
                print(f" :  CV test split size option is {test_size}, but steps is {steps}")
                print(f" -> Reducing CV test split size to {steps}")
                test_size = steps
        if 'shift' in method_opts:
            shift_cv = True
            if cv_min_split_size is None:
                raise ValueError("To use the single-shift CV option cv_min_split_size needs to be set")

        # Cross-validation object
        if (cv_splits > 2) and (not shift_cv):
            tscv = TimeSeriesSplit(
                n_splits=cv_splits, 
                test_size=test_size,
                max_train_size=cv_max_split_size, 
                #gap=0,
            )
        elif shift_cv:
            tscv = ShiftTimeSeriesSplit(
                min_split_size=cv_min_split_size,
                test_size=test_size,
                max_split_size=cv_max_split_size,
            )
        else:
            raise ValueError("Please choose a number of CV splits > 2 (option '-cv:__')")

        #for train_index, test_index in tscv.split(X_multi):
        #    print(train_index)
        #    print(test_index)
        #    print("---------------------------")

        assert steps > 0
        if multistep_cv:
            if steps_weights is None:
                steps_weights = np.ones(steps) / steps
            else:
                assert len(steps_weights) == steps
                steps_weights = np.squeeze(np.asarray(steps_weights))

        # Ridge penalty cross-validation objective
        if multistep_cv:
            # State reference dates
            #ref_dates = Y_dates
            #ref_index = len(fit['dates']) - 1
            #ref_dates_0 = ref_dates[ref_index:]

            # Slice matrices
            Ys = Y_[1:,]
            Xs = X_multi[0:(-1),]

            # Prepare date indexes for correct state iteration
            cpd_x_j = []
            tgd_x_j = []
            kt_j = [0 for _ in range(len(X))]
            for train_index, test_index in tscv.split(X_multi[0:(-1),]):
                dates_split = Y_dates[test_index]
                cpd_split = []
                tgd_split = []
                for j, x_j in enumerate(X):
                    cpd, kt = closest_past_date(x_j.index, dates_split[0], cutoff=kt_j[j])
                    cpd_split.append(cpd)
                    kt_j[j] = kt
                    #
                    tgd, _ = closest_past_date(x_j.index, dates_split[-1], cutoff=kt-1)
                    tgd_split.append(tgd)
                cpd_x_j.append(cpd_split)
                tgd_x_j.append(tgd_split)

                #print("...........")
                #print(train_index)
                #print(test_index)

            # Multi-step ahead cross-validation
            def CV_obj(p_lambda, p_lambda_comps):
                # Rescale penalty
                p_lambda = 10 ** (p_lambda)
                #p_lambda_out   = p_lambda[0:len(self.models_)]
                #p_lambda_comps = p_lambda[len(self.models_):]

                if isotropic_cv:
                    Lambda_ = p_lambda[0]
                else:
                    d = np.zeros(sum(self.states_N_))
                    i = 0
                    for n, k in enumerate(self.states_N_):
                        d[i:(i+k)] = p_lambda[n] * np.ones(k)
                        i += k
                    Lambda_ = np.diagflat(d)

                Obj = 0
                l = 0
                for train_index, test_index in tscv.split(Xs):
                    Ws = ls_ridge(Y=Ys[train_index,], X=Xs[train_index,], Lambda=Lambda_)

                    #j_dates = Y_dates[np.hstack((train_index[-1], test_index[0:-1]))]
                    #j_dates = Y_dates[test_index]
                    #cpd = j_dates[0]
                    #tgd = j_dates[-1]

                    # Iterate states forward
                    #test_dates = ref_dates[test_index]
                    X_ms_multi = np.full((len(test_index), self.M_), np.nan)
                    p = 0
                    for j, x_j in enumerate(X):
                        cpd = cpd_x_j[l][j]
                        tgd = tgd_x_j[l][j]

                        # Forward iterations indeces
                        dates_x_j = x_j.loc[cpd:tgd,].index
                        slice_x_j = dates_x_j.get_indexer(Y_dates[test_index])
                        iters_x_j = len(dates_x_j)

                        # Autonomous weigths
                        comps_cpd = Y_dates[train_index[0]]
                        comps_tgd = Y_dates[train_index[-1]]
                        if self.ar_:
                            if j == 0:
                                Ys_j = Y.loc[comps_cpd:comps_tgd,].to_numpy()
                            else:
                                Ys_j = z[j-1].loc[comps_cpd:comps_tgd,].to_numpy()
                        else:
                            Ys_j = z[j].loc[comps_cpd:comps_tgd,].to_numpy()
                        Xs_j = x_j.loc[comps_cpd:comps_tgd,].to_numpy()
                        Ws_j = ls_ridge(Y=Ys_j[1:,], X=Xs_j[0:-1,], Lambda=p_lambda_comps[n])

                        # Generate states
                        init_x_j = np.squeeze(x_j.loc[cpd,].to_numpy())
                        Xj, _ = self.models_[j].base_generate_autostates(
                            T=iters_x_j, 
                            W=Ws_j, init=init_x_j
                        )
                        #
                        X_ms_multi[:,p:(p+self.models_N_[j])] = Xj[slice_x_j,]
                        p += self.models_N_[j]
                    #
                    l += 1

                    Ys_fit = np.hstack((np.ones((X_ms_multi.shape[0], 1)), X_ms_multi)) @ Ws
                    Obj += np.sum(steps_weights * np.squeeze(Ys[test_index,] - Ys_fit) ** 2) / len(test_index)

                    #print("-----------------------")
                    #print(Y_dates[train_index])
                    #print(Y_dates[test_index])
                
                return Obj

            if debug and (isotropic_cv or len(self.models_) == 1):
                tmp_lambda_ls = np.linspace(-5, 5, 30)
                tmp_CV_obj = np.array([CV_obj(np.array((l, )), [Lambda0,]) for l in tmp_lambda_ls],)
                plt.figure(figsize=(5,2))
                plt.plot(tmp_lambda_ls, np.log10(tmp_CV_obj))
                plt.grid()
                plt.xlabel("$\log_{10}(\lambda)$")
                plt.ylabel("$\log_{10}(Loss(\lambda))$")
                plt.show()

            #print(CV_obj(np.ones(2), [1e-2, 1e-2]))
            #return [-1,]

        else:
            # Standard one-step-ahead cross-validation

            # Infer target frequency
            # NOTE: unfortunately this is needed to be able to correctly slice
            #       the regressors, which are of a higher frequency than Y
            #       and thus start from a previous point in time
            Y_freq = pd.infer_freq(Y_dates)
            Y_dates_offset = Y_dates - pd.tseries.frequencies.to_offset(Y_freq)

            # NOTE: to properly estimate CV loss, one needs to normalize data at
            #       each split and re-compute states. To make obj. fun. evaluation
            #       feasible, pre-compute all state matrices.
            X_multi_by_split = []
            Y_target_by_split = []
            for train_index, test_index in tscv.split(range(len(Y_dates)-1)):
                # Dates
                train_state_dates = Y_dates[train_index]
                test_state_dates = Y_dates[test_index]
                train_state_dates_offset = Y_dates_offset[train_index[[0]]]
                #test_state_dates_offset = Y_dates_offset[test_index]

                train_target_dates = Y_dates[[i + 1 for i in train_index]]
                test_target_dates = Y_dates[[i + 1 for i in test_index]]

                #print(f"+ -------------------------------------------")
                #print(f"train_state_dates: \n{train_state_dates}\n~")
                #print(f"test_state_dates: \n{test_state_dates}\n~")
                #print(f"train_state_dates_offset: \n{train_state_dates_offset}\n~")
                #print(f"train_target_dates: \n{train_target_dates}\n~")
                #print(f"test_target_dates: \n{test_target_dates}\n~")

                # Slice
                z_split = []
                #z_split_dates = []
                for j, zj in enumerate(z):
                    zj_split_train = zj.loc[train_state_dates_offset[0]:train_state_dates[-1],]
                    mean_zj = zj_split_train.mean()
                    std_zj = zj_split_train.std()
                    zj_split = (zj.loc[
                        train_state_dates_offset[0]:test_state_dates[-1],
                    ] - mean_zj) / std_zj

                    z_split.append(zj_split)
                    #z_split_dates.append(zj_split.index)
  
                mean_Y = Y.loc[train_state_dates[0]:train_target_dates[-1],].mean()
                std_Y = Y.loc[train_state_dates[0]:train_target_dates[-1],].std()
                Y_split_state = (Y.loc[
                    train_state_dates[0]:test_state_dates[-1],
                ] - mean_Y) / std_Y
                #Y_split_state_dates = Y_split_state.index

                # Flatten
                Y_split_state, Y_split_state_dates, z_split, z_split_dates = (
                    self.prep_data(Y_split_state, z_split)
                )

                # States
                X_split = self.multifreq_states(
                    Y=Y_split_state, z=z_split, Y_dates=Y_split_state_dates, z_dates=z_split_dates,
                    init=init, washout_len=washout_len,
                )

                X_split_multi = self.multifreq_states_to_matrix(
                    ref_dates=Y_split_state_dates, states=X_split,
                )
                X_split_multi_train = X_split_multi[train_index,]
                X_split_multi_test = X_split_multi[test_index,]

                # Targets
                Y_split_targets_train = ((Y.loc[train_target_dates] - mean_Y) / std_Y).to_numpy()
                Y_split_targets_test = ((Y.loc[test_target_dates] - mean_Y) / std_Y).to_numpy()

                # Save split slices
                X_multi_by_split.append(
                    (X_split_multi_train, X_split_multi_test)
                )
                Y_target_by_split.append(
                    (Y_split_targets_train, Y_split_targets_test)
                )

            if debug: print(". Folds built")

            def CV_obj(p_lambda, s):
                # Rescale penalty
                p_lambda = 10**(p_lambda)

                if isotropic_cv:
                    Lambda_ = p_lambda[0]
                else:
                    d = np.zeros(sum(self.states_N_))
                    i = 0
                    for n, k in enumerate(self.states_N_):
                        d[i:(i+k)] = p_lambda[n] * np.ones(k)
                        i += k
                    Lambda_ = np.diagflat(d)

                Obj = 0

                # Slice matrices
                #Ys = Y_[(1+s):,]
                #Xs = X_multi[0:(-1-s),]
                #for train_index, test_index in tscv.split(Xs):
                #    Ws = ls_ridge(Y=Ys[train_index,], X=Xs[train_index,], Lambda=Lambda_)
                #
                #    Ys_fit = np.hstack((np.ones((Xs[test_index,].shape[0], 1)), Xs[test_index,])) @ Ws
                #    Obj += np.sum((Ys[test_index,] - Ys_fit) ** 2) / len(test_index)
                #
                #    #print("-----------------------")
                #    #print(Y_dates[train_index])
                #    #print(Y_dates[test_index])

                for j, X_multi_split_j in enumerate(X_multi_by_split):
                    Ws = ls_ridge(Y=Y_target_by_split[j][0], X=X_multi_split_j[0], Lambda=Lambda_)

                    Ys_fit = np.hstack((np.ones((X_multi_split_j[1].shape[0], 1)), X_multi_split_j[1])) @ Ws
                    Ys_target = Y_target_by_split[j][1]

                    Obj += np.mean((Ys_target - Ys_fit) ** 2)
                
                return Obj


            #print(CV_obj(np.ones(1), 0))
            #return [-1,]

            if debug and (isotropic_cv or len(self.models_) == 1):
                tmp_lambda_ls = np.linspace(-7, 5, 50)
                tmp_CV_obj = np.array([CV_obj(np.array((l, )), 0) for l in tmp_lambda_ls])
                plt.figure(figsize=(5,2))
                plt.plot(tmp_lambda_ls, np.log10(tmp_CV_obj))
                plt.grid()
                plt.xlabel("$\log_{10}(\lambda)$")
                plt.ylabel("$\log_{10}(Loss(\lambda))$")
                plt.show()

        # Initialization and bounds
        Lambda0 = np.atleast_1d(np.asarray(np.log10(Lambda0)))
        xl = -5 * np.ones(Lambda0.shape),
        xu = +5 * np.ones(Lambda0.shape)
        if pca is None:
            if (not isotropic_cv) and (len(Lambda0) == 1):
                Lambda0 = np.repeat(Lambda0, len(self.models_))
                xl = np.repeat(xl, len(self.models_))
                xu = np.repeat(xu, len(self.models_))
        else:
            if not len(Lambda0) == 1:
                print("PCA requires scalar ridge penalty, selecting only first component of 'Lambda0'")
                Lambda0 = Lambda0[0]
                xl = [-5,],
                xu = [+5,]

        # CV 
        Lambda = []
        if multistep_cv:
            if Lambda_comps is None:
                Lambda_comps = Lambda0
            else:
                assert len(Lambda_comps) == len(self.models_)

            res = scipy_minimize(
                fun=lambda lb : CV_obj(lb, Lambda_comps),
                x0=Lambda0,
                bounds=tuple(zip(xl, xu)),
                method='L-BFGS-B',
                options={'disp': debug},
            )

            if debug:
                print(f"+ ----------------------------")
                print("Best solution found:")
                print(f"lambda = {10 ** res.x}")
                print(f"F      = {res.fun}")

            Lambda = np.atleast_1d(10 ** (res.x))

        else:
            for s in range(steps):
                res_s = scipy_minimize(
                    fun=lambda lb : CV_obj(lb, s),
                    x0=Lambda0,
                    bounds=tuple(zip(xl, xu)),
                    method='L-BFGS-B',
                    options={'disp': debug},
                )

                if debug:
                    print(f"+ s = {s} --------------------")
                    print("Best solution found:")
                    print(f"lambda = {10 ** res_s.x}")
                    print(f"F      = {res_s.fun}")

                Lambda.append(np.atleast_1d(10 ** (res_s.x)))

        return Lambda

    
    def ridge_lambda_components_cv(self, Y, z, cv_options='-cv:5', Lambda0=1e-2,
                init=None, washout_len=0, debug=True):
        if debug: print("ESNMultiFrequency.ridge_lambda_components_cv()")

        # Flatter data and extract datetime indexes
        Y_, Y_dates, z_, z_dates = self.prep_data(Y, z)

        # Decompose optimization type string
        cv_opts = re.findall("(?<=-)\w+(?=:)?", cv_options)
        cv_nums = re.findall("(?<=:)(\d+(?:\.\d+)?)", cv_options)

        if debug:
            print(". CV Options:")
            print(cv_opts)
            print(cv_nums)

        # States
        init = [None for _ in range(int(self.ar_) + len(z))] if init is None else init
        assert len(init) == len(self.models_), "Length of initialization 'init' must be equal to the number of ESN models"
        
        X = self.multifreq_states(
            Y=Y_, z=z_, Y_dates=Y_dates, z_dates=z_dates,
            init=init, washout_len=washout_len,
        )

        cv_splits = 2
        test_size = None
        if 'cv' in cv_opts:    
            cv_splits = int(cv_nums[cv_opts.index('cv')])
            assert cv_splits >= 0
        if 'test_size' in cv_opts:
            test_size = int(cv_nums[cv_opts.index('test_size')])
            assert test_size >= 0

        if cv_splits > 2:
            tscv = TimeSeriesSplit(
                n_splits=cv_splits, 
                test_size=test_size,
                #max_train_size=20, 
                #gap=0,
            )
        else:
            raise ValueError("Please choose a number of CV splits > 2 (option '-cv:__')")

        
        # Standard one-step-ahead cross-validation
        def CV_obj(p_lambda, j):
            Ys_j = z[j].iloc[1:,].to_numpy()
            Xs_j = X[j].iloc[0:-1,].to_numpy()

            # Penalty
            k = self.states_N_[j]
            Lambda_ = np.eye(k) * (10 ** (p_lambda))

            Obj = 0
            for train_index, test_index in tscv.split(Xs_j):
                Ws_j = ls_ridge(Y=Ys_j[train_index,], X=Xs_j[train_index,], Lambda=Lambda_)

                Ys_fit = np.hstack((np.ones((Xs_j[test_index,].shape[0], 1)), Xs_j[test_index,])) @ Ws_j
                Obj += np.sum((Ys_j[test_index,] - Ys_fit) ** 2) / len(test_index)
            
            return Obj

        #tmp_lambda_ls = np.linspace(-6, 5, 30)
        #tmp_CV_obj = np.array([CV_obj(np.array((l, )), 0) for l in tmp_lambda_ls])
        #plt.plot(tmp_lambda_ls, tmp_CV_obj)
        #plt.grid()
        #plt.show()

        # Initialization and bounds
        Lambda0 = np.atleast_1d(np.array(Lambda0))
        xl = -5 * np.ones(Lambda0.shape),
        xu = +5 * np.ones(Lambda0.shape)
        if len(Lambda0) == 1:
            Lambda0 = np.repeat(Lambda0, len(self.models_))
        assert len(Lambda0) == len(self.models_), "Lambda must have same length as number of models"

        # CV
        Lambda_j = np.full(len(Lambda0), np.nan)
        for j, _ in enumerate(z_):
            res_j = scipy_minimize(
                fun=lambda lb : CV_obj(lb, j),
                x0=Lambda0[j],
                bounds=tuple(zip(xl, xu)),
                method='L-BFGS-B',
                options={'disp': debug},
            )

            if debug:
                print(f"+ j = {j} --------------------")
                print("Best solution found:")
                print(f"lambda = {10 ** res_j.x}")
                print(f"F      = {res_j.fun.round(5)}")

            Lambda_j[j] = 10 ** (res_j.x)

        return Lambda_j
