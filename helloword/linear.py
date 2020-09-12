import matplotlib.pyplot as plt
from scipy import stats

x = [1,2,3,4,5,6,7,8,9]
y = [211,586,539,270,217,413,546,351,338]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()