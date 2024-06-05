#this is the evironment for the non-preemptive schemes

import copy
import numpy as np


seed = 10088
np.random.seed(seed)

#step(id_t) returns the next state (and available num_a) for all arms
#id_t is a list, when preemptive=True, id_t is 2-dimensional: represent the action and type

# in the non-preemptive condition, the prob_n should be [n x m]


class MEC_Model:
    def __init__(self, num_t, min_time, prob_n, preemptive=False, max_t=[1,1]):

        # num_t: the number of users, n
        # min_time: tau_min, minimum computation time, list
        # prob_n: the transition probability ,list n x 2
        # max_t: available arm, [m1, m2]
        # status: record the generated age and the current age of each arm, mat
        # preemptive: flag whether it is preemptive or non-preemptive

        self.num_t = num_t
        self.min_time = np.array(min_time, dtype=np.int8)
        self.prob_n = np.array(prob_n, dtype=np.float32)
       
        self.max_t = copy.deepcopy(max_t)
        self.max_t_init = copy.deepcopy(max_t)
        self.status = np.zeros([self.num_t, 2],dtype=np.int32)
        # the id of the server assigned with users
        self.current_arm = np.zeros(self.num_t,dtype=np.int16)

        self.preemptive = preemptive

        #evolve: plus which to evolve all the arm to the next time slot
        self.evolve = np.ones_like(self.status)
        
    def transitable(self, t):
        # judge whether the computation status is transitable, compared with tau_min
        # index 0 is the current age, index 1 is the age at the generation time
        # to optimize, can return the status as a whole in mat

        if self.status[t,0] - self.status[t, 1] < self.min_time[t]:
            return False
        else:
            return True
           
    def reset(self):
        self.status = np.zeros([self.num_t, 2], dtype=np.int32)
        self.current_arm = -np.ones(self.num_t,dtype=np.int16)
        self.max_t = copy.deepcopy(self.max_t_init)
        return self.status, self.max_t
        
    
    # there maybe more efficient ways for step update
    def step(self, id_t):
        #id_t: the id derived from the compute_action(). it should be a list contains m lists. Can Not Be np.array()!
        # shape like [[1,4,6],[2,3]]
        # every age plus one in advance
        self.status += self.evolve
        if not self.preemptive:
            #non-preemptive condition
            id_comp = np.where(self.status[:,0]!=self.status[:,1])[0]

            #consider users already in computation. arms can only be released
            for t in id_comp:
                if self.transitable(t):
                    is_finished = bool(np.random.choice(2, p=[1-self.prob_n[t][self.current_arm[t]], self.prob_n[t][self.current_arm[t]]]))
                    if not is_finished:
                        self.status[t,1] -= 1
                    else:
                        temp = self.status[t,0] - self.status[t,1]
                        self.status[t,0] = temp + 1
                        self.status[t,1] = temp + 1
                        #release the arm
                        self.max_t[self.current_arm[t]] += 1
                        self.current_arm[t] = -1

                else:
                    self.status[t,1] -= 1


            # assign the action. since the non-preemptive condition, arms can only be assigned
            for m, t_s in enumerate(id_t):
                #id_t: n x 2
                for t in t_s:
                    if self.max_t[m] == 0:
                        break
                    if t not in id_comp:
                        
                        # if not in computation, it means to generate a new task and offload 
                        self.status[t,1] -= 1
                        #assign the arm
                        self.max_t[m] -= 1
                        self.current_arm[t] = m
                       

        else:
            id_comp = np.where(self.status[:,0]!=self.status[:,1])[0]
            for t in id_comp:
                if t not in id_t[0] or t not in id_t[1]:
                #there are only two type of servers
                #reset arms that are not computing
                    #drop current task
                    self.status[t,1] = self.status[t,0]
                    #release the arm
                    self.max_t[self.current_arm[t]] += 1
                    self.current_arm[t]=-1
 
            for m,t_s in enumerate(id_t):
                for t in t_s:
                    if self.transitable(t):
                        is_finished = bool(np.random.choice(2, p=[1-self.prob_n[t][m], self.prob_n[t][m]]))
                        if not is_finished:
                            self.status[t,1] -= 1
                        else:
                            temp = self.status[t,0] - self.status[t,1]
                            self.status[t,0] = temp + 1
                            self.status[t,1] = temp + 1
                            #release the arm
                            self.max_t[self.current_arm[t]] += 1
                            self.current_arm[t] = -1

                    else:
                        self.status[t,1] -= 1
                        #assign arms
                        if self.current_arm[t]==0:
                            self.current_arm[t]==m
                            self.max_t[m] -= 1
        return self.current_arm

if __name__=="__main__":
    env = MEC_Model(3,[2,4,8],[[0.1,0.5],[0.3,0.7],[0.2,0.6]])
    env.reset()
    env.step([[0,2],[1]])
    env.step([[0,2],[1]])