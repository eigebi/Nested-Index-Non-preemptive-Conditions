#maximization of the nested index: linear programming
import numpy as np
#from scipy.optimize import linprog
import cvxpy as cp
from scipy.optimize import linprog
import time

class MIP():
    def __init__(self,num_n,num_m,available_m) -> None:
        self.n = num_n
        self.m = num_m
        self.available_m = available_m
        self.A_ub = np.concatenate((np.kron(np.ones(self.m),np.identity(self.n)),np.kron(np.identity(self.m),np.ones(self.n))),0)
        self.A_ub = np.concatenate((np.diag(-np.ones(self.n*self.m)),self.A_ub),0)
        self.B_ub = np.concatenate((np.array([1 for _ in range(self.n)]),self.available_m),0)
        self.B_ub = np.concatenate((np.zeros(self.n*self.m),self.B_ub),0)
        #self.x_box = [(0,1) for _ in range(self.n*self.m)]
        #self.dual_box = [(0,10000) for _ in range(self.m+self.n)]
        

    def cal_arm(self, index, available_m=None):
        if available_m is not None:
            self.available_m = available_m
        idx = np.array(index).transpose([1,0]).reshape(-1)
        x = cp.Variable(self.n*self.m)
        prob = cp.Problem(cp.Minimize(-idx.T@x),
                          [self.A_ub@x <= self.B_ub])
        prob.solve()
        res = linprog(-idx,self.A_ub,self.B_ub)
        dual = linprog(self.B_ub.T,-self.A_ub.T,-idx.T)
        return [[np.array(np.isclose(x.value,1),dtype=np.int8),prob.constraints[0].dual_value[-self.m:]],[res.x,dual.x[-self.m:]]]
    

if __name__ == "__main__":
    opt = MIP(3,2,[2,2])
    print(opt.A_ub)
    print(opt.B_ub)
    s = time.time()
    print(opt.cal_arm([[1,4],[3,5],[1,2]]))
    e = time.time()
    print("time: {:.5f}".format(e-s))