import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

n = 100
beta_0 = 5
beta_1 = 2

np.random.seed(1)
x = 10 * ss.uniform.rvs(size = 100)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc=0, scale=1, size=n)

# Practice plotting
"""
plt.figure()
plt.plot(x,y,"o",ms=5)

xx = np.array([0,10])
plt.plot(xx, beta_0 + beta_1 * xx)
plt.xlabel("X")
plt.ylabel("Y")

def estimate_y(x, b_0, b_1):
    return (b_0 + b_1 * x)

def compute_rss(y_estimate, y):
    return sum(np.power(y-y_estimate, 2))
"""

#Finding best slope (writing a regression function... basically)
"""
rss = []
slopes = np.arange(-10,15,0.0001)

for slope in slopes:
    rss.append(np.sum((y - beta_0 - slope * x)**2))
    
ind_min = np.argmin(rss)
print(ind_min)

print("Estimate for the slope", slopes[ind_min])

plt.figure()
plt.plot(slopes, rss)
plt.xlabel("Slope")
plt.ylabel("RSS")
"""

# Regression
"""
X = sm.add_constant(x)
mod = sm.OLS(y,X)
est = mod.fit()
print(est.summary())
"""

# Multi Variable Regression
"""
n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1

x_1 = 10*ss.uniform.rvs(size=n)
x_2 = 10*ss.uniform.rvs(size=n)

y = beta_0 + beta_1*x_1 + beta_2*x_2 + ss.norm.rvs(loc=0, scale=1, size=n)

X = np.stack([x_1, x_2], axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:,0], X[:,1], y , c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")


lm = LinearRegression(fit_intercept=True)
lm.fit(X, y)

print("Intercept: ", lm.intercept_)
print("Beta 1: ", lm.coef_[0])
print("Beta 2: ", lm.coef_[1])

#plug in X_0 which includes a point x1 and x2
X_0 = np.array([2,4])

predict_X0 = lm.predict(X_0.reshape(1,-1))
print(predict_X0)

rSquared = lm.score(X, y)
print(rSquared)
"""

# Basic Splitting Data
"""
#Training Data
x_1 = 10*ss.uniform.rvs(size=n)
x_2 = 10*ss.uniform.rvs(size=n)

X = np.stack([x_1, x_2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, y_train)

print(lm.score(X_test, y_test))

"""

# Cllassifying Data

h = 1
sd = 1 
n = 1000

def generateData (h, sd1, sd2, n):
    x1 = ss.norm.rvs(-h, sd1, n)
    y1 = ss.norm.rvs(0, sd1, n)
    x2 = ss.norm.rvs(h, sd2, n)
    y2 = ss.norm.rvs(0, sd2, n)
    return (x1, y1, x2, y2)

sample1 = generateData(h,sd, sd, n)

def plotData(data_tuple):
    plt.figure()
    plt.plot(data_tuple[0], data_tuple[1], "o", ms=2)
    plt.plot(data_tuple[2], data_tuple[3], "o", ms=2)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_1$")
    
#plotData(sample1)

# Classification Type Regression (Logistic Regression)
def vstackData(sample):
    x1 = np.vstack((sample[0],sample[1])).T
    x2 = np.vstack((sample[2], sample[3])).T
    return (np.vstack((x1,x2)))
    
def hstackData(n):
    return (np.hstack((np.repeat(1,n), np.repeat(2,n))))

Xtrain, Xtest, Ytrain, Ytest = train_test_split(vstackData(sample1), hstackData(n), train_size=0.5, random_state=1)
clf = LogisticRegression()
clf.fit(Xtrain, Ytrain)
print(clf.score(Xtest, Ytest))
print(clf.predict(np.array([-2,0]).reshape(1,-1)))