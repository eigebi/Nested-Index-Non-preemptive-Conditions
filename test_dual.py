import numpy as np
from scipy.optimize import linprog
import cvxpy as cp
import time

idx = np.array([[1,4],[3,5],[4,2]],dtype = np.float32)
idx = np.array(idx).transpose([1,0]).reshape(-1)
x0 = np.zeros(6)
x = cp.Variable(6)
A_ub = np.concatenate((np.kron(np.ones(2),np.identity(3)),np.kron(np.identity(2),np.ones(3))),0)
B_ub = np.concatenate((np.array([1 for _ in range(3)]),[1,]),0)
A_ub = np.concatenate((np.diag(-np.ones(6)),A_ub),0)
B_ub = np.concatenate((np.zeros(6),B_ub),0)
prob = cp.Problem(cp.Minimize(-idx.T@x),
                  [A_ub@x <= B_ub])
s = time.time()
prob.solve()
e = time.time()
print(x.value)
print(prob.constraints[0].dual_value)
print(e-s)
