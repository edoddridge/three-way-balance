import os.path as p

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

self_path = p.dirname(p.abspath(__file__))
root_path = p.dirname(self_path)

import sys
sys.path.append(p.join(root_path, 'test'))
sys.path.append(p.join(root_path, 'reproductions/BG_2016'))

import aronnax as aro
import aronnax.driver as drv
from aronnax.utils import working_directory


def BG_wetmask(X, Y):
    """The wet mask."""

    # water everywhere, doubly periodic
    wetmask = np.ones(X.shape, dtype=np.float64)

    return wetmask



def find_last_checkpoint():
    
    import glob

    dhdt_checkpoints = sorted(glob.glob('checkpoints/dhdt.*'))

    try:
        niter0 = int(dhdt_checkpoints[-1][-10:])
    except:
        niter0 = 0

    return niter0


def run_ice_drag(nTimeSteps, dt, kappa, wind_mag, wind_stoch, wind_period,
                    dir=None, iniH=400.):

    #assert layers == 1
    nx = 320
    ny = 320
    layers = 1
    xlen = 1280e3
    ylen = 1280e3
    dx = xlen / nx
    dy = ylen / ny

    grid = aro.Grid(nx, ny, layers, dx, dy)

    if wind_stoch:
        def wind_time_series(nTimeSteps, dt):
            # np.random.seed(42)
            wind_ts = (np.random.rand(nTimeSteps)*wind_mag) + (wind_mag/2.)
            return wind_ts
    elif wind_period!=0:
        # sinusoidally varying wind
        def wind_time_series(nTimeSteps, dt):
            time_vec = np.arange(nTimeSteps, dtype=np.float)*dt
            wind_ts = wind_mag*(1. + 0.5*np.sin(2.*np.pi*time_vec/wind_period))
            return wind_ts
    else:
        wind_time_series = wind_mag

    with working_directory(p.join(self_path, 
        "{0}".format(dir))):
        niter0 = find_last_checkpoint()

        # because of the frustrating restriction about setting things to
        # 0 or 1, we need if statements here.
        if niter0 == 0:
            if kappa == 0:
                drv.simulate(
                        initHfile=[iniH],
                        wind_mag_time_series_file=[wind_time_series],
                        wetMaskFile=[BG_wetmask],
                        nx=nx, ny=ny, dx=dx, dy=dy, 
                        exe='aronnax_core', 
                        dt=dt, nTimeSteps=nTimeSteps)
            else:
                drv.simulate(
                        initHfile=[iniH],
                        wind_mag_time_series_file=[wind_time_series],
                        wetMaskFile=[BG_wetmask],
                        kh=kappa,
                        nx=nx, ny=ny, dx=dx, dy=dy, 
                        exe='aronnax_core', 
                        dt=dt, nTimeSteps=nTimeSteps)

        else:
            if kappa == 0:
                drv.simulate(
                        initHfile=[iniH],
                        wind_mag_time_series_file=[wind_time_series],
                        wetMaskFile=[BG_wetmask],
                        niter0=niter0,
                        nx=nx, ny=ny, dx=dx, dy=dy, 
                        exe='aronnax_core', 
                        dt=dt, nTimeSteps=nTimeSteps)
            else:
                drv.simulate(
                        initHfile=[iniH],
                        wind_mag_time_series_file=[wind_time_series],
                        wetMaskFile=[BG_wetmask],
                        kh=kappa,
                        niter0=niter0,
                        nx=nx, ny=ny, dx=dx, dy=dy, 
                        exe='aronnax_core', 
                        dt=dt, nTimeSteps=nTimeSteps)

# ~7 hours per year on Ed's laptop

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Run BG ice drag experiment')

    parser.add_argument('-nTimeSteps', type=int, default=6220800, 
                        help='number of time steps to run')

    parser.add_argument('-dt', type=float, default=100, 
                        help='time step in seconds')
    
    parser.add_argument('-kappa', type=float, default=10, 
                        help='eddy diffusivity')

    parser.add_argument('-wind_mag', type=float, default=0.08,
                        help='ice velocity in cm/s')

    parser.add_argument('-wind_stoch', type=bool, default=False,
                        help='stochastically variable momentum forcing?')

    parser.add_argument('-wind_period', type=float, default=0,
                        help='period for sinusoidal momentuum forcing')
    
    parser.add_argument('-dir', type=str, default='default', 
                        help='folder to run simulation in')
    
    parser.add_argument('-iniH', type=float, default=400.,
                        help='initial thickness of active layer')

    args = parser.parse_args()
    run_ice_drag(**vars(args))
    # 31104000 for 100 year
