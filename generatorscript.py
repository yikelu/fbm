import numpy as np

def fbmmatrix(start=0,end=30,ntimes=7560,hurst=0.5):
	times=np.linspace(start,end,ntimes)
	twohurst=2.*hurst
	Rts=lambda t, s, twoH: 0.5*(s**(twoH) + t**(twoH) - np.abs(t - s)**(twoH))
	# lambda versus function definition doesn't change speed
	# Rts executes 7.39 us per time
	# This implies gamma runs over 422s for ntimes=7560
	gamma=np.zeros([ntimes,ntimes])
	for i in range(ntimes):
		for j in range(ntimes):
			gamma[i,j]=Rts(times[i],times[j],twohurst)
	return gamma
#	sigma=np.linalg.cholesky(gamma)

def fbmmatrix2(start=0,end=30,ntimes=7560,hurst=0.5):
	timescol=np.matrix(np.linspace(start,end,ntimes)[1:])
	# drop the zero element, picked up this trick from R fArma
	# matrix is not SPD otherwise
	timesrow=timescol.T
	twohurst=2.*hurst
	gamma=0.5*(np.power(timesrow,twohurst) + np.power(timescol,twohurst)
			- np.power(np.abs(timesrow - timescol),twohurst))
	# this only takes 6.13 seconds, clever matrix manips
	return gamma

# for size 1000, np.linalg.cholesky(gamma)=>243ms
# implies for 7560, it should take 105s per matrix.
# timeit result is 121s per loop
# this translates into 3.36 hours of compute time on my laptop
# for a resolution of 0.01 in Hurst exponent

# sigma * dB matrix time for 7560 is 83.5ms per loop from timeit
# from Numerical Methods, approximation error of Monte Carlo is
# given by O(max(delta t, 1/sqrt(n))) where delta t is the increment
# size and n is the number of sample paths.

# per my calculations, this means I need around 60k sample paths for
# meaningful convergence. Given this, we can choose 100k for simplicity
# and this translates into calculation time of 230 hours (2.3 hours per
# value of H)  OUCH this is not good.

# From this, it appears we should either step up to Amazon cluster computer
# or scale down the ambitiousness of this project.
# For one, we can definitely scale down by a factor of 10, for it is
# definitely overkill to use H to 0.01 resolution as no estimator can
# reasonably assure that level of accuracy.
# Therefore we change it to 0.1 resolution in H, cutting compute time down
# to 23 hours.
# In order to cut down on memory usage, we can similarly cut down on the
# number of time steps. Cutting it in half (15 years) gives:

# cholesky takes 15.5s per decomp
# matrix multiply takes 21.1 ms per loop.
# Cholesky is now a non factor - it will only take 155 seconds to
# get all the matrices.
# the matrix multiply will now only take 6 hours!

# Still we should probably investigate cluster compute at this point.
# Next step will be to time things on the micro instance on Amazon.

