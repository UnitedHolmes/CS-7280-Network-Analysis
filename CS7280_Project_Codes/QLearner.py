"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr =  radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.Q = np.zeros([num_states, num_actions])
        self.experience = []

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        current_random = rand.random()
        if current_random < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[self.s,:])
        self.a = action
        if self.verbose: print( "s =", s,"a =",action)
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        # Imrpove Q table
        argmaxQ = np.argmax(self.Q[s_prime,:])
        self.Q[self.s,self.a] = (1-self.alpha)*self.Q[self.s,self.a] + self.alpha*(r+self.gamma*self.Q[s_prime,argmaxQ])
        
        # Dyna-Q
        if self.dyna > 0:
            if self.experience == []:
                self.experience = np.array([self.s, self.a, s_prime, r])
                flag_first = True
            else:
                current_tuple = np.array([self.s, self.a, s_prime, r])
                self.experience = np.vstack((self.experience,current_tuple))
                flag_first = False
            if flag_first == False:                
                for k in range(0,self.dyna):
                    row_index = rand.randint(0,self.experience.shape[0]-1)
                    ss = self.experience[row_index,0]
                    aa = self.experience[row_index,1]
                    ss_prime = self.experience[row_index,2]
                    rr = self.experience[row_index,3]
                    argmaxQ = np.argmax(self.Q[ss_prime,:])
                    self.Q[ss,aa] = (1-self.alpha)*self.Q[ss,aa] + self.alpha*(rr+self.gamma*self.Q[ss_prime,argmaxQ])
            
        # update state
        self.s = s_prime
                
        # choose best action a, with random action implemented
        current_random = rand.random()
        if current_random < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[self.s,:])
        self.a = action
        # Update rar
        self.rar = self.rar * self.radr
                
        if self.verbose: print( "s =", s_prime,"a =",action,"r =",r)
        return action
    
    def author(self):
        return 'yzhu94'

if __name__=="__main__":
    print( "Remember Q from Star Trek? Well, this isn't him")
