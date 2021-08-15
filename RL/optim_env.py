import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
import matplotlib.pyplot as plt

class OptimEnv:

    def __init__(self) -> None: # ax^2 + by^2 + cxy + dx + ey + f
        self.a = np.random.rand() * 10.0
        self.b = np.random.rand() * 10.0
        self.c = (np.random.rand()*2 - 1)*np.sqrt(self.a*self.b)*2  # to make the Hessian positive
        self.d = (np.random.rand()*2 - 1)*10
        self.e = (np.random.rand()*2 - 1)*10
        self.f = (np.random.rand()*2 - 1)*10
        self.x = (np.random.rand()*2 - 1)*20
        self.y = (np.random.rand()*2 - 1)*20
        self.start_x = self.x
        self.start_y = self.y
        self.global_min_x =  (2*self.b*self.d - self.c*self.e)/(self.c**2 - 4*self.a*self.b)
        self.global_min_y =  (2*self.a*self.e - self.c*self.d)/(self.c**2 - 4*self.a*self.b)
        self.count = 0
        plt.scatter(self.global_min_x, self.global_min_y, color='g')
        plt.scatter(self.x, self.y, color='k')
        self.prev_x = self.x
        self.prev_y = self.y

    def func_val(self):
        x = self.x
        y = self.y
        return (self.a *(x**2) + self.b*(y**2) + self.c*(x*y) + self.d*x + self.e*y + self.f)
    
    def func_val_with_coord(self, x, y):
        return (self.a *(x**2) + self.b*(y**2) + self.c*(x*y) + self.d*x + self.e*y + self.f)
    
    def step(self, action):
        """
            action = [del_x, del_y]
            state = [x,y,a,b,c,d,e,f]
        """
        self.count += 1
        del_x, del_y = action
        d1 = (( np.sqrt((self.x-self.global_min_x)**2 + (self.y-self.global_min_y)**2)))

        s = np.array([self.x, self.y], dtype=np.float32)
        g = np.array([self.global_min_x, self.global_min_y], dtype=np.float32)
        # print(self.x)
        self.x += del_x
        self.y += del_y
        plt.scatter(self.x, self.y, color='r')
        plt.plot([self.prev_x, self.x], [self.prev_y, self.y], 'r')

        s1 = np.array([self.x, self.y], dtype=np.float32)

        self.prev_x = self.x
        self.prev_y = self.y
        next_state = np.array([self.x, self.y, self.a, self.b, self.c, self.d, self.e, self.f])
        

        d = (( np.sqrt((self.x-self.global_min_x)**2 + (self.y-self.global_min_y)**2)))
        d_bar = abs(self.func_val() - self.func_val_with_coord(self.global_min_x, self.global_min_y)) + d
        # if(d):
        #     reward = 1.0/d
        # else :
        #     reward = 10000

        # reward = -1 * self.func_val()

        t1 = s1-s
        t2 = g-s
        reward = np.dot(t1.T, t2)
        reward = reward/d1
        reward += (1/d)

        if(d <= 1):
            reward += 1/d

        # else :
        #     reward=-1

        done = True if(self.count == 200) else False
        if(d < 0.1):
            print("REACHED GLOBAL MINIMA")
            reward += 1000 
            done = True
            # plt.pause(0.001)
        info_dict = {}  # change it later
        return [next_state, reward, done, info_dict]

    def render(self):
        plt.pause(0.001)
        # print(
        #     f"Current Coordinates : ({round(self.x, 5)} , {round(self.y,5)})\nGlobal Minima : ({round(self.global_min_x,5)} , {round(self.global_min_y,5)})\nCurrent Cost : {self.func_val()}\nGlobal Optimum Cost : {self.func_val_with_coord(self.global_min_x, self.global_min_y)}"
        #     )
        # print(f'start point = {self.start_x} , {self.start_y}')
        # print(f'{round(self.x, 5)} , {round(self.y,5)}')
    def reset(self):
        plt.clf()
        self.__init__()
        return np.array([self.x, self.y, self.a, self.b, self.c, self.d, self.e, self.f])