# newToolbox_ESN
#   Basic ESN functionality
#
# Minor update: July 2024
# ================================================================

from multiprocessing.sharedctypes import Value
import types
from math import floor, ceil, inf 
import pandas as pd
import numpy as np
#from numpy import random
import re
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# ----------------------------------------------------------------
# Preamble

def identity(A):
    return A

def radbas(A):
    return np.exp(-A**2)

def retanh(A):
    return np.tanh(np.maximum(A, 0))

def softplus(A):
    return np.log(1 + np.exp(A))

def esn_data_to_nparray(data):
    v = None
    if (type(data) is pd.DataFrame):
        v = data.to_numpy(copy=True)
    elif (type(data) is pd.Series):
        v = data.to_numpy(copy=True)[:,None]
    elif type(data) is np.ndarray:
        v = np.copy(data)
    if (not v is None) and (v.ndim == 1):
        # Mutate to column vector
        v = np.atleast_2d(v)

    #print(v.shape)

    return v

def vech(V):
    assert type(V) is np.ndarray
    N1, N2 = np.atleast_2d(V).shape
    assert N1 == N2
    v = np.zeros(N1*(N1+1)//2)
    j = 0
    for n in range(N1):
        v[j:(j+N1-n)] = V[n:,n]
        j += N1-n
    return v

def matrh(v, N):
    assert type(v) is np.ndarray
    M = len(v)
    assert N*(N+1)//2 == M
    V = np.zeros((N, N))
    j = 0
    for n in range(N):
        V[n:,n] = v[j:(j+N-n)]
        j += N-n
    return V

# ----------------------------------------------------------------
# State map ingredients

# ----------------------------------------------------------------
# Fitting Methods

def ls_ridge(Y, X, Lambda=0):
    Tx, Kx = X.shape
    Ty, _  = Y.shape
    assert Tx == Ty, "Shapes of X and Y non compatible"

    #print(X.shape)
    #print(Y.shape)

    try:
        Lambda = np.squeeze(np.asarray(Lambda))
    except:
        raise ValueError("Lambda must be a scalar or a 2D numpy array")
        
    # Regression matrices
    if Lambda.shape == ():
        # Lambda is a scalar 
        if Lambda < 1e-12:
            V = np.hstack((np.ones((Tx, 1)), X))
            res = np.linalg.lstsq(V, Y, rcond=None)
            W = res[0]
        else:
            W = np.linalg.solve(((X.T @ X / Tx) + Lambda * np.eye(Kx)), (X.T @ Y / Ty))
            #W = np.linalg.solve(((X.T @ X) + Lambda * np.eye(Kx)), (X.T @ Y))
            a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
            W = np.vstack((a, W))
    #elif 
    elif Lambda.shape == (Kx,Kx):
        W = np.linalg.solve(((X.T @ X / Tx) + Lambda), (X.T @ Y / Ty))
        #W = np.linalg.solve(((X.T @ X) + Lambda), (X.T @ Y))
        a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
        W = np.vstack((a, W))
    else:
        raise ValueError(f"Lambda is not scalar or 2D matrix with Gram shape ({Kx},{Kx}), found shape {Lambda.shape}")
    
    return W

def jack_ridge(Y, X, Lambda):
    Tx, _ = X.shape
    Ty, _  = Y.shape
    assert Tx == Ty, "Shapes of X and Y non compatible"

    if np.isscalar(Lambda):
        Lambda = Lambda * np.eye(X.shape[1])
    Gamma0 = ((X.T @ X / Tx) + Lambda)
    W = np.linalg.solve((Gamma0 @ Gamma0), (X.T @ Y / Ty))
    a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
    W = np.vstack((a, W))
    return W

def r_ls(Y, X, W0=None, P0=None):
    Tx, Kx = X.shape
    Ty, Ky = Y.shape
    assert Tx == Ty, "Shapes of X and Y non compatible"

    V = np.hstack((np.ones((Tx, 1)), X))

    if not P0 is None:
        assert P0.shape == (Kx+1, Kx+1)
        P = np.copy(P0)
    else:
        P = np.linalg.inv(V[[0],] @ V[[0],].T + np.eye(Kx+1))
        #P = np.linalg.inv(X.T @ X)

    if not W0 is None:
        assert W0.shape == (Kx+1, Ky)
        W = [np.copy(W0),]
    else:
        W = [np.zeros((Kx+1, Ky)),]
    Wt = np.copy(W[0])

    Yx = np.zeros(Y.shape)
    # Recursive least-squares updates
    for t in range(1, Ty):
        H = V[[t-1],]
        # Update
        K = P @ H.T @ np.linalg.pinv(1 + H @ P @ H.T)
        P = (np.eye(1+Kx) - K @ H) @ P
        
        Wt += (P @ H.T) @ (Y[t-1,] - H @ Wt)
        W.append(Wt)

        Yx[t,] = V[[t],] @ Wt

    return W, Yx  

# ----------------------------------------------------------------
# ESN Class
#

class ESN:
    def __init__(self, N=None, A=None, C=None, rho=0, zeta=None, gamma=1, leak_rate=0,
                    params=None, activation=identity):
        self.params_ = params
        self.activation_ = activation

        # Pre-allocations
        self.X_ = None
        #self.W_ = None

        # Parameter checks and allocation
        # number of neurons
        self.N_ = N #A.shape[0] if (not type(N) is int) else N
        # reservoir (connectivity) matrix 
        self.A_ = A
        # input mask        
        self.C_ = C
        # input scaling
        self.gamma_ = gamma
        # reservoir (connectivity) matrix spectral radius
        self.rho_ = rho
        # leak rate
        self.leak_rate_ = leak_rate * np.ones((self.N_, 1)) if (not type(leak_rate) is np.ndarray) else leak_rate
        # input shift
        self.zeta_ = zeta if (not zeta is None) else np.zeros((self.N_, 1))
        
        # If 'params' is set, overwrite ESN parameters:
        # useful to programmatically generate ESNs from tuple of parameters
        if not self.params_ is None:
            #A, C, gamma, zeta, rho, leak_rate = (None, None, 
            #                                     1, None, None, 0)
            if len(self.params_) == 3:
                A, C, rho = self.params_
            elif len(self.params_) == 4:
                A, C, rho, zeta = self.params_
            elif len(self.params_) == 5:
                A, C, rho, zeta, gamma = self.params_
            elif len(self.params_) == 6:
                A, C, rho, zeta, gamma, leak_rate = self.params_
            else:
                raise ValueError("Parameter tuple 'self.params_' has unknown content")

            # Assign
            self.N_ = N
            self.A_ = np.copy(A)
            self.C_ = np.copy(C)
            self.zeta_ = np.copy(zeta)
            self.gamma_ = np.copy(gamma)
            self.rho_ = np.copy(rho)
            self.leak_rate_ = np.copy(leak_rate)

        # Parameter checks
        N0, N1 = self.A_.shape
        assert N0 == N1, "A is not square"
        assert N0 == self.N_, "A does not have the size N"
        assert self.C_.shape[0] ==  N0, "A and C are not compatible"  
        assert self.zeta_.shape[0] == N0, "zeta is not shape-compatible with input"
        assert self.zeta_.shape[1] == 1, "zeta is not a vector"
        assert self.leak_rate_.shape[0] == N0, "Leak rate in not shape-compatible with states"
            
        assert np.isscalar(gamma), "gamma is not a scalar"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN State Gathering Functions
    # assuming indexing from 0
    #  feed z_0, z_1, ..., z_{T}

    def generate_states(self, z, A, C, rho, zeta, gamma, leak_rate,
                            init=None, collect='forecast', washout_len=0):
        # Flatten data
        z = np.atleast_2d(np.squeeze(z))
        #if z is None:
        #    raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray.")

        # Check dimensions
        if C.shape[1] != z.shape[0]:
            z = z.T
        Kz, T = z.shape
        assert Kz == C.shape[1], "C is not shape-compatible with input"

        # Gather states
        N = A.shape[0] 
        
        X = np.zeros((N, T))
        if init is None:
            init = np.zeros((N, 1))
        else:
            init = np.reshape(init, (N, 1))
        
        X[:, [0]] = (init * leak_rate + (1 - leak_rate) * self.activation_((A * rho) @ init
                                                                            + gamma * C @ z[:, [0]]
                                                                            + zeta))             
    
        for t in range(1, T):
            X[:, [t]] = (X[:, [t-1]] * leak_rate
                         + (1 - leak_rate) * self.activation_((A * rho) @ X[:, [t-1]]
                                                                + gamma * C @ z[:, [t]]
                                                                + zeta))
            
        X = X[:,washout_len:].T
        
        if collect == 'listen':
            self.X_ = X
        
        return X

    def base_generate_states(self, z, init=None, collect='forecast', washout_len=0):
        return self.generate_states(
            z=z, A=self.A_, C=self.C_, rho=self.rho_, zeta=self.zeta_, 
                    gamma=self.gamma_, leak_rate=self.leak_rate_, 
                    init=init, washout_len=washout_len, collect=collect
        )

    def generate_autostates(self, T, W, A, C, rho, zeta, gamma, leak_rate, init):
        # Check T
        assert (type(T) is int) and (T > 0), "Number of autonomous states to generate 'T' must be a positive integer."
        
        # Gather autonomous-run states
        N = A.shape[0] 
        M = W.shape[1]
        
        y = np.zeros((M, T))
        X = np.zeros((N, T))
        if init is None:
            init = np.zeros((N, 1))
        else:
            init = np.reshape(init, (N, 1))

        # Initial state 
        X[:, [0]] = init  
        y[:, [0]] = W.T @ np.vstack((1, init))
    
        for t in range(1, T):
            X[:, [t]] = (
                X[:, [t-1]] * leak_rate + (1 - leak_rate) * 
                    self.activation_((A * rho) @ X[:, [t-1]] + gamma * C @ y[:, [t-1]] + zeta)
            )
            y[:, [t]] = W.T @ np.vstack((1, X[:,[t]]))

        return X.T, y.T

    def base_generate_autostates(self, T, W, init):
        return self.generate_autostates(
            T=T, W=W, A=self.A_, C=self.C_, rho=self.rho_, zeta=self.zeta_,
                gamma=self.gamma_, leak_rate=self.leak_rate_,
                init=init
        )

    # Convenience fitting function for univariate time series
    def fit_ar(self, Y, method='ridge', Lambda=0, init=None, washout_len=0):
        # Flatten data
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")

        Y0 = Y[:-1,]
        Y1 = Y[1:,]

        #print(Y)
        #print(Y0)
        #print(Y1)

        return self.fit(Y=Y1, z=Y0, method=method, Lambda=Lambda, init=init, washout_len=washout_len)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Forecasting
    # important that zf starts with t out of which is made a forecasting step
    # the output is step shifted w.r.t. zf
    def forecast(self, zf, fit, steps=1, method=None, init='cont'):
        # Flatten data
        zf_ = esn_data_to_nparray(zf)
        if zf_ is None:
            raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")

        # Gather forecasting states
        if init is None:
            print("[ Forecasting states initialization is 'None': fallback to zero init ]\n")
            init_ = None
        # states contain X_0,...,X_{T}
        elif init == 'cont':
            init_ = fit['X'][-1,:]

        else:
            init_ = np.atleast_1d(init)
        
        Xf_ = self.base_generate_states(z=zf_, init=init_, collect='forecast')
        #Txf_, Nx, = Xf_.shape
        
        #assert Nx == self.N_, "dimension of generated states is incorrect"
        
        # Forecast
        Vf_ = None
        if (method == 'none') or (method is None):
            if steps == 1:
                Vf_ = np.hstack((np.ones((Xf_.shape[0]-1, 1)), Xf_[:-1,]))
            
        else:
            raise ValueError("Forecasting method not defined")

        Forecast = Vf_ @ fit['W']
        Yf_      = zf[1:,]

        forecast_out = {
            'Forecast':     Forecast,
            'Yf':           Yf_,
            'Vf':           Vf_,
            'Xf':           Xf_,
            'steps':        steps,
            'method':       method,
            'init':         init,
        }

        return forecast_out

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting

    def plot_rho_lambda_loss(self, Y, z, vrho, vlambda, train_test=False, init=None, washout_len=0, figsize=(7, 6), surf3d=False):
        # Flatten data
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")
        z_ = esn_data_to_nparray(z)
        if z_ is None:
            raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")

        # Check sample size
        assert Y_.shape[0] ==  z_.shape[0], "Data inputs Y and z have incompatile sample sizes"

        Nr = len(vrho)
        Nl = len(vlambda)

        # Move to log10 scale
        Xr, Yl = np.meshgrid(vrho, np.power(10, vlambda))
        
        def RSS_loss_surf(v):
            # States
            X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=v[0], zeta=self.zeta_, 
                                        gamma=self.gamma_, leak_rate=self.leak_rate_, 
                                        init=init, washout_len=washout_len, collect="listen")

            if train_test:
                train_split = floor(X_.shape[0] * 0.8)
                # Fit
                W         = ls_ridge(Y=Y_[0:train_split,], X=X_[0:train_split,], Lambda=v[1])
                #V_fit     = np.hstack((np.ones((X_[train_split:,].shape[0], 1)), X_[train_split:,]))
                Y_fit     = np.hstack((np.ones((X_[train_split:,].shape[0], 1)), X_[train_split:,])) @ W
                #Residuals = Y_ - Y_fit
                RSS       = np.sum((Y_[train_split:,] - Y_fit) ** 2) / (X_.shape[0] - train_split)
            else:
                # Fit
                W         = ls_ridge(Y=Y_, X=X_, Lambda=v[1])
                #V_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_))
                Y_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W
                #Residuals = Y_ - Y_fit
                RSS       = np.sum((Y_ - Y_fit) ** 2) / X_.shape[0]
            
            return RSS
        
        RSS_loss_surf = Parallel(n_jobs=-1, verbose=False, prefer="processes")(
            delayed(RSS_loss_surf)(v) for v in zip(np.matrix.flatten(Xr), np.matrix.flatten(Yl))
        )

        # Reshape
        RSS_loss_surf = np.reshape(RSS_loss_surf, (Nr, Nl), order='F').T

        #RSS_loss_surf = np.full((Nr, Nl), np.nan)
        #for i, rho_i in enumerate(tqdm(vrho)):
        #    for j, lambda_i in enumerate(vlambda):
        #        # States
        #        X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=rho_i, zeta=self.zeta_, 
        #                                    gamma=self.gamma_, leak_rate=self.leak_rate_, 
        #                                    init=init, washout_len=washout_len, collect="listen")
        #        # Fit
        #        W = ls_ridge(Y=Y_, X=X_, Lambda=lambda_i)
        #        #V_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_))
        #        Y_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W
        #        #Residuals = Y_ - Y_fit
        #        RSS       = np.sum((Y_ - Y_fit) ** 2)
        #        #
        #        RSS_loss_surf[j, i] = RSS

        # Plot
        if surf3d:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
            scm1 = ax.plot_surface(Xr, np.log10(Yl), RSS_loss_surf, cmap=plt.cm.nipy_spectral, linewidth=0, antialiased=False)
            plt.colorbar(scm1)
        else:
            plt.figure(figsize=figsize)
            pcm1 = plt.pcolormesh(Xr, np.log10(Yl), RSS_loss_surf, cmap=plt.cm.nipy_spectral, shading='auto')
            plt.colorbar(pcm1)
        plt.xlabel('rho')
        plt.ylabel('log10(lambda)')
        plt.show()

        return RSS_loss_surf

    def plot_rho_gamma_loss(self, Y, z, vrho, vgamma, Lambda=0, log_xy=None, train_test=False, init=None, washout_len=0, figsize=(7, 6), surf3d=False):
        # Flatten data
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")
        z_ = esn_data_to_nparray(z)
        if z_ is None:
            raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")

        # Check sample size
        assert Y_.shape[0] ==  z_.shape[0], "Data inputs Y and z have incompatile sample sizes"

        Nr = len(vrho)
        Ng = len(vgamma)

        # Add log10 scale (if needed)
        vrho = np.power(10, vrho) if log_xy in ('x', 'xy') else vrho
        vgamma = np.power(10, vgamma) if log_xy in ('y', 'xy') else vgamma

        # Compute loss 
        Xr, Yg = np.meshgrid(vrho, vgamma)
        
        def RSS_loss_surf(v):
            # States
            X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=v[0], zeta=self.zeta_, 
                                        gamma=v[1], leak_rate=self.leak_rate_, 
                                        init=init, washout_len=washout_len, collect="listen")

            if train_test:
                train_split = floor(X_.shape[0] * 0.8)
                # Fit
                W         = ls_ridge(Y=Y_[0:train_split,], X=X_[0:train_split,], Lambda=Lambda)
                #V_fit     = np.hstack((np.ones((X_[train_split:,].shape[0], 1)), X_[train_split:,]))
                Y_fit     = np.hstack((np.ones((X_[train_split:,].shape[0], 1)), X_[train_split:,])) @ W
                #Residuals = Y_ - Y_fit
                RSS       = np.sum((Y_[train_split:,] - Y_fit) ** 2) / (X_.shape[0] - train_split)
            else:
                # Fit
                W         = ls_ridge(Y=Y_, X=X_, Lambda=v[1])
                #V_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_))
                Y_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W
                #Residuals = Y_ - Y_fit
                RSS       = np.sum((Y_ - Y_fit) ** 2) / X_.shape[0]
            
            return RSS
        
        RSS_loss_surf = Parallel(n_jobs=-1, verbose=False, prefer="processes")(
            delayed(RSS_loss_surf)(v) for v in zip(np.matrix.flatten(Xr), np.matrix.flatten(Yg))
        )

        # Reshape
        RSS_loss_surf = np.reshape(RSS_loss_surf, (Nr, Ng), order='F').T

        # Remove log10 scale (if needed)
        Xr = np.log10(Xr) if log_xy in ('x', 'xy') else Xr
        Yg = np.log10(Yg) if log_xy in ('y', 'xy') else Yg

        # Plot
        if surf3d:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
            scm1 = ax.plot_surface(Xr, Yg, RSS_loss_surf, cmap=plt.cm.nipy_spectral, linewidth=0, antialiased=False)
            plt.colorbar(scm1)
        else:
            plt.figure(figsize=figsize)
            pcm1 = plt.pcolormesh(Xr, Yg, RSS_loss_surf, cmap=plt.cm.nipy_spectral, shading='auto')
            plt.colorbar(pcm1)
        plt.xlabel('rho')
        plt.ylabel('gamma')
        plt.show()

        return RSS_loss_surf


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helpers

def plotFitted(fit_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    t = type(fit_out['Y'])
    h = fit_out['Y'].shape[1]
    if h > 1:
        gs = fig.add_gridspec(h, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        if (t is pd.DataFrame) or (t is pd.Series):
            for i, c in enumerate(fit_out['Y'].columns):
                axs[i].plot(fit_out['Y'][c], c='k', alpha=0.3, label=str(c))
                axs[i].plot(fit_out['Y_fit'][c], c='C'+str(i), label="Fitted")
                axs[i].grid()
                axs[i].legend()
                axs[i].label_outer()
        else: 
            for i in range(h):
                axs[i].plot(fit_out['Y'][:,i], c='k', alpha=0.3, label=str(i))
                axs[i].plot(fit_out['Y_fit'][:,i], c='C'+str(i), label="Fitted")
                axs[i].grid()
                axs[i].legend()
                axs[i].label_outer()
        axs[0].set_title("ESN - Fitted Values")
    else:
        plt.plot(fit_out['Y'], label="Data", alpha=0.3)
        plt.plot(fit_out['Y_fit'], label="Fitted")
        plt.grid()
        ax.legend()
        ax.set_title("ESN - Fitted Values")
    #return fig

def plotResiduals(fit_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(fit_out['Residuals'], label="Residuals")
    plt.grid()
    ax.legend()
    ax.set_title("ESN - Fit Residuals")
    #return fig

def plotStates(fit_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    N = fit_out['X'].shape[1]
    plt.plot(fit_out['X'][:,:min(N,100)], color='k', alpha=0.15)
    plt.grid()
    ax.set_title("ESN - Collected States [max 100]")

def plotW(fit_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(fit_out['W'], c="k", marker=".")
    plt.grid()
    ax.set_title("ESN - Estimated Weigths")
    #return fig

def plotOptimFitted(optim_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    t = type(optim_out['Y'])
    h = optim_out['Y'].shape[1]
    r = np.arange(-1, len(optim_out['Y'])-1)
    if h > 1:
        gs = fig.add_gridspec(h, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        if (t is pd.DataFrame) or (t is pd.Series):
            for i, c in enumerate(optim_out['Y'].columns):
                axs[i].plot(r, optim_out['Y'][c], c='k', alpha=0.3, label=str(c))
                for j in range(len(optim_out['Y_fit_opt'])):
                    l1 = optim_out['length_train'][j]
                    l2 = optim_out['Y_fit_opt'][j].shape[0]
                    if l2 > 1:
                        axs[i].plot(np.arange(l1, l1+l2), optim_out['Y_fit_opt'][j][c], c='C'+str(i))
                    else:
                        axs[i].scatter(l1+l2, optim_out['Y_fit_opt'][j][c], c='C'+str(i))
                axs[i].grid()
                axs[i].legend()
                axs[i].label_outer()
        else: 
            for i in range(h):
                axs[i].plot(r, optim_out['Y'][:,i], c='k', alpha=0.3, label=str(i))
                for j in range(len(optim_out['Y_fit_opt'])):
                    l1 = optim_out['length_train'][j]
                    l2 = optim_out['Y_fit_opt'][j].shape[0]
                    if l2 > 1:
                        axs[i].plot(np.arange(l1, l1+l2), optim_out['Y_fit_opt'][j][:,i], c='C'+str(i))
                    else:
                        axs[i].scatter(l1+l2, optim_out['Y_fit_opt'][j][:,i], c='C'+str(i))
                axs[i].grid()
                axs[i].legend()
                axs[i].label_outer()
        axs[0].set_title("ESN - Fitted Values")
    else:
        plt.plot(r, optim_out['Y'], label="Data", alpha=0.3)
        for j in range(len(optim_out['Y_fit_opt'])):
            l1 = optim_out['length_train'][j]
            l2 = optim_out['Y_fit_opt'][j].shape[0]
            if l2 > 1:
                plt.plot(np.arange(l1, l1+l2), optim_out['Y_fit_opt'][j])
            else:
                plt.scatter(l1, optim_out['Y_fit_opt'][j], marker='.')
        plt.grid()
        ax.legend()
        ax.set_title("ESN - Optimization - Fitted Values")
    #return fig

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# State Matrix Generator
    
def stateMatrixGenerator(shape, dist='normal', sparsity=None, normalize=None, options=None, seed=None):
    # stateMatrixGenerator
    #   Function to generate state matrices (e.g. state matrix A, input mask C)
    #   according to commonly used entry-wise or matrix-wise distributions.
    #
    #       shape       tuple, dimensions of the matrix to generate
    #       dist        type of matrix to generate
    #       sparsity    degree of sparsity (~ proportion of 0 elements)
    #                   of the generated matrix. Ignored if 'type' does
    #                   not have a 'sparse_' prefix
    #       normalize   normalization to apply to the matrix:
    #                       'eig'       maximum absolute eigenvalue
    #                       'sv'        maximum singular value
    #                       'norm2'     spectral norm
    #                       'normS'     infinity (sup) norm         
    #  

    # Set seed
    if not seed is None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(12345)

    if (shape is None) or (not type(shape) is tuple):
        raise ValueError("No shape selected")
    if (len(shape) > 2):
        raise ValueError("Shape tuple is larger than 2d")
    if dist is None:
        dist = 'normal'
    if sparsity is None:
        if re.findall('^sparse', dist):
            raise ValueError('Sparse distributions require choosing sparsity degree')
        sparsity = 1.
    else:
        if (sparsity < 0) or (sparsity >= 1):
            raise ValueError("Chosen sparsity degree is not within [0,1)")
    
    # Generate
    H = np.empty(shape)
    if dist == 'normal':
        H = rng.standard_normal(size=shape)
    elif dist == 'uniform':
        H = rng.uniform(low=-1, high=1, size=shape)
    elif dist == 'sparse_normal':
        B = rng.binomial(n=1, p=sparsity, size=shape)
        H = B * rng.standard_normal(size=shape)
    elif dist == 'sparse_uniform':
        B = rng.binomial(n=1, p=sparsity, size=shape)
        H = B * rng.uniform(low=-1, high=1, size=shape)
    elif dist == 'orthogonal':
        H = np.linalg.svd(rng.standard_normal(size=shape))[0]
        # Ignore normalization
        normalize = None
    elif dist  == 'takens':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            H = np.eye((N1-1))
            H = np.hstack((H, np.zeros((N1-1, 1))))
            H = np.vstack((np.zeros((1, N1)), H))
            H = np.kron(np.eye(M), H)
        elif N1 > N2:
            H = np.hstack((1, np.zeros(N1-1)))
            H = np.atleast_2d(H).T
            id_matrix = np.eye(M)
            H = np.kron(id_matrix, H)
        else:
            raise ValueError("Shape not comformable to takens")
    elif dist  == 'takens_exp':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            for j in range(1,M + 1):
                H = np.eye((N1-1))
                np.fill_diagonal(H, np.exp(-np.random.uniform(low=1, high=M, size=(1,1))*np.array(range(1,N1))))
                H = np.hstack((H, np.zeros((N1-1, 1))))
                H = np.vstack((np.zeros((1, N1)), H))
                if j == 1:
                    H_res = H
                else:
                    H_res = block_diag(H_res, H)
            H = H_res
        elif N1 > N2:
            H = np.hstack((1, np.zeros(N1-1)))
            H = np.atleast_2d(H).T
            id_matrix = np.eye(M)
            np.fill_diagonal(id_matrix, np.random.uniform(low=0, high=1, size=(1,M)))
            H = np.kron(id_matrix, H)
        else:
            raise ValueError("Shape not comformable to takens")
    elif dist == 'takens_augment':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            H = np.zeros((M*N1, M*N1))
            for m in range(M):
                def underdiag_f(j):
                    #return np.random.uniform(low=0, high=0.5, size=(j))
                    #return (M-j)/(M+1) * np.ones((j))
                    #return np.exp(-(N1-1-j)) * np.ones((j))
                    return np.exp(-np.arange(N1-1, N1-1-j, -1)**.5)
                    #return np.exp(-np.arange(j)/N1)
                # Progressive fill under-diagonals
                H_m = np.atleast_2d(underdiag_f(1))
                for j in range(1, N1):
                    Q_j = np.zeros((j, j))
                    Q_j[1:,:-1] = H_m
                    np.fill_diagonal(Q_j, underdiag_f(j))
                    H_m = Q_j
                # Largest under-diagonal is just 1s (normalization)
                # Q_j = np.zeros((N1-1, N1-1))
                # Q_j[1:,:-1] = H_m
                # np.fill_diagonal(Q_j, np.ones(N1-1))
                H_m = Q_j
                H_m = np.hstack((H_m, np.zeros((N1-1, 1))))
                H_m = np.vstack((np.zeros((1, N1)), H_m))
                H[m*N1:(m+1)*N1,m*N1:(m+1)*N1] = H_m
        else:
            raise ValueError("Shape not comformable to takens")
    else:
        raise ValueError("Unknown matrix distribution/type")

    # Normalize
    if not normalize is None:
        if normalize == 'eig':
            H /= np.max(np.abs(np.linalg.eigvals(H)))
        elif normalize == 'sv':
            H /= np.max(np.linalg.svd(H)[1])
        elif normalize == 'norm2':
            H /= np.linalg.norm(H, ord=2)
        elif normalize == 'normS':
            H /= np.linalg.norm(H, ord=inf)
        else:
            raise ValueError("Unknown normalization")

    return H