import numpy as np
from globalvars import *

def fbmmatrix(start=0,end=30,ntimes=7560,hurst=0.5):
    timescol=np.matrix(np.linspace(start,end,ntimes)[1:])
    # drop the zero element, picked up this trick from R fArma
    # matrix is not SPD otherwise
    timesrow=timescol.T
    twohurst=2.*hurst
    gamma=0.5*(np.power(timesrow,twohurst) + np.power(timescol,twohurst)
            - np.power(np.abs(timesrow - timescol),twohurst))
    return np.linalg.cholesky(gamma)



for h in covmatrange:
    covmat = fbmmatrix(hurst=h)
    fn=covmatprefix + str(h)
    np.save(file=datadir+fn, arr=covmat)
    del(covmat)
    print('finished covmat for hurst='+str(h))

#print('finished covmat generation')

for i in mcrange:
    montecarlo=0.3*np.sqrt(1./252.)*np.random.randn(7560,10000)
    fn=mcprefix+str(i)
    np.save(file=datadir+fn, arr=montecarlo)
    del(montecarlo)
    print('finished montecarlo run number '+str(i))

exit()
