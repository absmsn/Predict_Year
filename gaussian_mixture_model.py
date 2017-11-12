"""
the procedure of em algorithm 
given the Q function: Q(θ|θi)=Σ(Z)(logP(Y,Z|θ)P(Z|Θi,Y))
Y represents the observation data, theta i represents the theta value in the 
last recursion step and Z represents the latent variable.
The aim is to maximum the value of Q and deem it as the theta(i+1) and proceed 
the process until theta(i+1)-theta(i) << some little value(convenge to a specific value)

so the main idea is to maximum the 

GMM(guassian mixture model):
the structure of parameter theta: 
miu: the mean of every gaussian distribution
sigma: the variance of every gaussian distribution
alpha: the power of every gaussian distribution

the precess of getting Gmm parameter:
initialize the parameter(miu,sigma,alpha)
calculating the responsivity of model to every data
beta(j,k)=alpha(k)N(Y(i)|theta(k))/SUM(alpha(k)N(y(i)|theta(i))
@ j the num of data
@ k the num of model
"""
import numpy as np
import scipy
from itertools import repeat

data_set = []


def initialize_parameter(dis_num):
    miu = np.array()
    sigma = np.array()
    alpha = list(repeat(1/dis_num, dis_num))
    return miu, sigma, alpha


def recusrion(data, dis_num):
    miu, sigma, alpha = initialize_parameter(5)
    data_len = len(data)
    res_matrix = np.ndarray((data_len, dis_num))
    norm_dis = [scipy.stats(miu[i], sigma[i]) for i in range(dis_num)]
    for i in range(data_len):
        denominator = []
        for j in range(dis_num):
            member = alpha[j] * norm_dis[j].pdf(data[i])
            denominator.append(member)
        den_num = sum(denominator)
        for j in range(dis_num):
            res_matrix[i][j] = denominator[j] / den_num
    for i in range(dis_num):
        res_sum = sum(res_matrix[:, i])
        miu[i] = np.array(data).dot(res_matrix[:, i]) / res_sum
        member = sum(np.array(res_matrix[:, i].dot((np.array(data)-miu[i])*2)))
        sigma[i] = member / res_sum
        alpha[i] = res_sum/data_len
