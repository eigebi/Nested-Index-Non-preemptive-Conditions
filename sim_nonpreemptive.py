from env import MEC_Model
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
from compute_action import IntraIndex, MIP


# r_max: the maximum scale of the system for simulation
N = 50
K = np.array([3,2])
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

steps = 10000

#from bandit import index_func,threshold_tab, thres_L, expect_u

compare = []



if __name__=='__main__': 
    
    final_lambda = []

    for r in range(1,2):
        fin = []
        tau = [t for _ in range(r)]
        tau = np.array(tau).reshape(-1)
        prob_n = [p for _ in range(r)]
        prob_n = np.array(prob_n).reshape(N*r,2)
        env = MEC_Model(N*r, tau, prob_n, preemptive=False, max_t=K*r)
        status, max_t = env.reset()
        result = []
        temp = np.zeros([N*r], dtype=np.float32)
        for i in range(steps):
            # heuristic manners, greedily choose server by age
            part = status[:,0]+ (status[:,0]-status[:,1])/np.mean(prob_n,1)
            #id_t = np.argpartition(status[:,0],-K*r)[-K*r:]
            id_t = np.argpartition(part,-np.sum(K)*r)[-np.sum(K)*r:]
            id_t = [id_t[:K[0]*r],id_t[K[0]*r:]]
            env.step(id_t)
            temp += status[:,0]
            result.append(np.sum(temp)/(i+1))
        result = np.array(result)/(N*r)
        fin.append(result[-1])
        plt.plot(result)
        plt.show()


        # intra-index policy
        II = IntraIndex(prob_n)
        v = np.ones(2)
        status, max_t = env.reset()
        beta = 0.1
        temp = np.zeros([N*r], dtype=np.float32)
        result_id = []
        for i in range(2000):
            if not env.max_t.any():
                #if no arm is available, act the last action,and save result
                env.step(id_t)
                temp += status[:,0]
                result_id.append(np.sum(temp)/(i+1))
                continue
            id = II.compute(status[:,0],v)
            id = np.maximum(id[:,1:],0)
            opt = MIP(N*r,2,env.max_t)
            temp_,dual = opt.cal_arm(id)
            temp_ = temp_.reshape(2,-1)
            id_t = [np.where(temp_[0]==1)[0],np.where(temp_[1]==1)[0]]
            env.step(id_t)
            temp += status[:,0]
            v = v+beta*(dual-v)
            print(v)
            result_id.append(np.sum(temp)/(i+1))
        result_id = np.array(result_id)/(N*r)
        plt.plot(result_id)
        plt.show()


        
