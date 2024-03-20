# Common functions
import numpy as np
from scipy.signal import savgol_filter

def parse_float(s):
  try:
    return(float(s))
  except:
    return np.nan

def moving_average(x, w):
  return np.convolve(x, np.ones(w), "same") / w

def impute_missing_values(xs, ys):
  i = 0
  while i < len(xs):
    if np.isnan(ys[i]):
      i0 = i-1; i1 = i; i2 = i
      while np.isnan(ys[i]):
        i2 = i
        i += 1
      i3 = i2 + 1
      deriv = (ys[i3]-ys[i0])/(xs[i3]-xs[i0])
      for j in range(i1, i2+1):
        dx = xs[j] - xs[i0]
        ys[j] = ys[i0] + dx*deriv
    i += 1

def smoothen(values):
  # return moving_average(values, 30)
  return savgol_filter(values, 5, 1)