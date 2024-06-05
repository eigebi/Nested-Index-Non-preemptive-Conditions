from env import MEC_Model
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
from compute_action import IntraIndex, MIP

N = 50
K = np.array([20,2])
r_max = 21
prob1=[0.8,0.7,0.6,0.5,0.3,0.1]
prob2=[0.9,0.9,0.8,0.9,0.7,0.5]
t_=[2,4,8,16,32,64]
n_=[5,10,5,5,10,15]


prob_n1=[]
prob_n2=[]
t=[]
for i,pp1,pp2,tt in zip(n_,prob1,prob2,t_):
    prob_n1 += [pp1 for _ in range(i)]
    prob_n2 += [pp2 for _ in range(i)]
    t += [tt for _ in range(i)]
p1 = np.array(prob_n1).reshape(1,-1)
p2 = np.array(prob_n2).reshape(1,-1)
p = np.concatenate((p1,p2),0).transpose(1,0)
pp = np.array([prob1,prob2]).transpose(1,0)

if __name__=='__main__': 
    for r in range(1,2):
        fin = []
        tau = [t for _ in range(r)]
        tau = np.array(tau).reshape(-1)
        prob_n = [p for _ in range(r)]
        prob_n = np.array(prob_n).reshape(N*r,2)
        env = MEC_Model(N*r, tau, prob_n, preemptive=False, max_t=K*r)
        status, max_t = env.reset()
        II = IntraIndex(prob_n)
        #v = np.ones(2)
        v = np.array([1100., 5.])
        #dual ascent to derive the optiaml server cost and the lower bound for the sub-problems
        beta = 1
        v_list = []
        for i in range(1000):
            delta_ = II.opt_dual_value(v,pp,t_,n_)
            #v = np.maximum(v+ beta*(delta_-K),0)
            v = v+ beta*(delta_-K)
            v_list.append(v)
            if i%10==0:
                print("round",i)
                print('delta',delta_)

        v_list = np.array(v_list)
        plt.plot(v_list)
        plt.show()
        
        