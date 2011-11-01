import numpy as np

datadir='~/fbmdata/'

def fbmmatrix(start=0,end=30,ntimes=7560,hurst=0.5):
    timescol=np.matrix(np.linspace(start,end,ntimes)[1:])
    # drop the zero element, picked up this trick from R fArma
    # matrix is not SPD otherwise
    timesrow=timescol.T
    twohurst=2.*hurst
    gamma=0.5*(np.power(timesrow,twohurst) + np.power(timescol,twohurst)
            - np.power(np.abs(timesrow - timescol),twohurst))
    return np.linalg.cholesky(gamma)
