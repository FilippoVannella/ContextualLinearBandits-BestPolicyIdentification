import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from utils import *

np.random.seed(5031240) # numpy random seed
N_run = 100 # Number of runs
d = 3 # Dimensions
K = 4 # Arms
C = 5 # Contexts
delta = 0.01 # Confidence
eps = 0 # Accuracy
sigma = 1 # Standard deviation
theta = np.ones(d)
Phi = np.random.normal(size = (C, K, d)) # Features
u = 1
g = np.max([np.linalg.norm(Phi[x, a], ord = 2)**2 for x in range(C) for a in range(K)])
a_star = np.argmax(Phi@theta, axis = 1)
p_X = np.random.rand(C) # context probability
p_X = p_X/sum(p_X)
alpha = np.random.rand(C, K) # Arms proportions (random)
for x in range(C):
    alpha[x] = alpha[x]/np.sum(alpha[x]) # Normalization

# Compute lower bound (passive, independent, active)
dx_vec = []
for x in range(C):
    dx_vec.append(np.linalg.matrix_rank(compute_A_ind(Phi, x, K, p_X, alpha))) 

T_star_passive = compute_T_star_passive(Phi, theta, a_star, p_X, alpha, eps, P = None)
T_star_ind = compute_T_star_ind(Phi, theta, a_star, p_X, alpha, eps, P = None)
T_star_active, alpha_star = optimize(C, K, Phi, p_X, a_star, theta)
_, T_star_ind_act = compute_T_star_active_ind(Phi, p_X, theta, a_star, eps)
print("Lower Bound Passive:", T_star_passive*np.log(1/delta))
print("Lower Bound Independent:", T_star_ind*np.log(1/delta))
print("Lower Bound Active:",T_star_active*np.log(1/delta))
print("Lower Bound Active Independent:",T_star_ind_act*np.log(1/delta))

# print("Optimal allocation", alpha_star)

# Run PCL-TaS (random)
print("Running PCL-TaS (random)...")
l_min, theta_hat, A_t, beta_vec, Z_vec, sample_complexity_vec_passive, best_arm_vec, eps_arm_vec, err_vec = PCL_TaS(N_run, theta, Phi, K, C, d, g, u, eps, delta, sigma, alpha, p_X, a_star, print_flag = False)
print("Sample complexity (avg +- std):", np.mean(sample_complexity_vec_passive), "+-", np.std(sample_complexity_vec_passive))
print("Error (avg +- std):", np.mean(err_vec), "+-", np.std(err_vec))
Z_passive, error_Z_passive = tolerant_mean(Z_vec)
beta_passive, error_beta_passive = tolerant_mean(beta_vec)

plt.figure()
plt.plot(np.arange(len(Z_passive))+1, Z_passive)
plt.fill_between(np.arange(len(Z_passive))+1, Z_passive.data - error_Z_passive,  Z_passive.data + error_Z_passive, color='C0', alpha=0.5)
#plt.plot(np.arange(len(beta_passive)) + 1, beta_passive,"r")
#plt.fill_between(np.arange(len(beta_passive)) + 1, beta_passive.data - error_beta_passive, beta_passive.data + error_beta_passive, color = "r", alpha=0.5)

# PI-TaS
print("Running PI-TaS (random)...")
l_min, theta_hat,A_t,beta_vec, Z_vec, sample_complexity_vec_off, best_arm_vec, eps_arm_vec, err_vec = run_passive_ind(N_run, theta, Phi, K, C, d, g, u, eps, delta, sigma, alpha, p_X, a_star, print_flag = False)
p_err_mean = np.mean(err_vec)
p_err_std = np.std(err_vec)
print("Error mean (avg +- std):", p_err_mean, "+-", p_err_std)
print("sample complexity (avg +- std):", np.mean(sample_complexity_vec_off), "+-", np.std(sample_complexity_vec_off))

Z_vec_list = [[] for _ in range(C)]
for n in range(N_run):
    for c in range(C):
        Z_arr = np.array(Z_vec[n])
        Z_vec_list[c].append(list(Z_arr[:, c]))
        
for c in range(C):
    y, error = tolerant_mean(Z_vec_list[c])
    plt.plot(np.arange(len(y))+1, y)
    yerr0 = y.data - error
    yerr1 = y.data + error
    plt.fill_between(np.arange(len(y))+1, yerr0, yerr1, alpha = 0.5)

"""beta_vec_list = [[] for _ in range(C)]
for n in range(N_run):
    for c in range(C):
        beta_arr = np.array(beta_vec[n])
        beta_vec_list[c].append(list(beta_arr[:, c]))

for c in range(C):
    y, error = tolerant_mean(beta_vec_list[c])
    plt.plot(np.arange(len(y))+1, y)
    yerr0 = y.data - error
    yerr1 = y.data + error
    plt.fill_between(np.arange(len(y))+1, yerr0, yerr1, alpha = 0.5)"""

A0 = []
dx_vec = []
for x in range(C):
    dx_vec.append(np.linalg.matrix_rank(compute_A_ind(Phi, x, K, p_X, alpha)))
    A0.append(find_minimal_size_basis([list(Phi[x, a]) for a in range(K)])) 
# Run CL-TaS
print("Running CL-TaS...")
alpha_vec, forced_expl_steps, tracking_steps, N_vec, l_min, theta_hat, A_t, beta_vec, Z_vec, sample_complexity_vec_active, best_arm_vec, eps_arm_vec, err_vec = run_active(A0, N_run, theta, Phi, K, C, d, dx_vec, g, u, eps, delta, sigma, alpha, p_X, a_star, print_flag = False)
print("Error mean (avg +- std):", np.mean(err_vec), "+-", np.std(err_vec))
print("sample complexity (avg +- std):", np.mean(sample_complexity_vec_active), "+-", np.std(sample_complexity_vec_active))
"""Z_act, error_Z_act = tolerant_mean(Z_vec)
beta_act, error_beta_act = tolerant_mean(beta_vec)plt.figure()
plt.plot(np.arange(len(Z_act)) + 1, Z_act)
plt.fill_between(np.arange(len(Z_act)) + 1, Z_act.data - error_Z_act,  Z_act.data + error_Z_act, color='C0', alpha=0.5)
plt.plot(np.arange(len(beta_act)) + 1, beta_act, "r")
plt.fill_between(np.arange(len(beta_act)) + 1, beta_act.data - error_beta_act, beta_act.data + error_beta_act, color = "r", alpha=0.5)
plt.show()"""
plt.xlim([0,len(Z_passive)])
plt.show()