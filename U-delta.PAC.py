import numpy as np
import matplotlib.pyplot as plt
from utils import *
import tikzplotlib
np.random.seed(0) # numpy random seed
N_run = 10 # Number of runs
d = 3 # Dimension
K = 2 # Arms
C = 3 # Contexts
delta = 0.01 # Confidence
sigma = 1 # Standard deviation
gap = .5 # Arms gap
theta = np.array([1, 1, 1])
Phi = np.array([[[1, 0, 0], [0, 1 - gap, 0]], 
                [[0, 1, 0], [1 - gap, 0, 0]], 
                [[1, 0, 0], [0, 0, 1 - gap]]])
a_star = np.argmax(Phi@theta, axis = 1) # Optimal policy
g = np.max([np.linalg.norm(Phi[x, a], ord = 2)**2 for x in range(C) for a in range(K)])
u = .1
p_X = np.ones(C)/C # Context distribution (uniform random)
alpha = np.ones((C, K))/K # Arms proportions (uniform random)
A0 = []
dx_vec = []
for x in range(C):
    dx_vec.append(np.linalg.matrix_rank(compute_A_ind(Phi, x, p_X, alpha)))
    A0.append(find_minimal_size_basis([list(Phi[x, a]) for a in range(K)])) 

# Compute lower bounds (passive and active) 
T_star_passive = compute_T_star_passive(Phi, theta, a_star, p_X, alpha)
T_star_active, alpha_star = solve_T_star_active(Phi, p_X, a_star, theta)
print("Lower Bound Passive (random):", T_star_passive*np.log(1/delta),"\nLower Bound Active:",T_star_active*np.log(1/delta))
# print("Optimal allocation", alpha_star)

# Run PCL-TaS (random)
print("Running PCL-TaS (random)...")
l_min, theta_hat, A_t, beta_vec, Z_vec, sample_complexity_vec_passive, best_arm_vec, err_vec = PCL_TaS(N_run, theta, Phi, g, u, delta, sigma, alpha, p_X, a_star, print_flag = False)
print("Sample complexity (avg +- std):", np.mean(sample_complexity_vec_passive), "+-", np.std(sample_complexity_vec_passive))
print("Error (avg +- std):", np.mean(err_vec), "+-", np.std(err_vec))
"""Z_passive, error_Z_passive = tolerant_mean(Z_vec)
beta_passive, error_beta_passive = tolerant_mean(beta_vec)
plt.figure()
plt.plot(np.arange(len(Z_passive))+1, Z_passive)
plt.fill_between(np.arange(len(Z_passive))+1, Z_passive.data - error_Z_passive,  Z_passive.data + error_Z_passive, color='C0', alpha=0.5)
plt.plot(np.arange(len(beta_passive)) + 1, beta_passive,"r")
plt.fill_between(np.arange(len(beta_passive)) + 1, beta_passive.data - error_beta_passive, beta_passive.data + error_beta_passive, color = "r", alpha=0.5)
plt.show()"""

# Run CL-TaS
print("Running CL-TaS...")
alpha_vec, forced_expl_steps_active, tracking_steps_active, N_vec_active, l_min_active, theta_hat_active, A_t, beta_vec_active, Z_vec_active, sample_complexity_vec_active, best_arm_vec_active, err_vec_active = CL_TaS(A0, N_run, theta, Phi, dx_vec, g, u, delta, sigma, alpha, p_X, a_star, print_flag = False)
print("Error mean (avg +- std):", np.mean(err_vec_active), "+-", np.std(err_vec_active))
print("Sample complexity (avg +- std):", np.mean(sample_complexity_vec_active), "+-", np.std(sample_complexity_vec_active))
Z_act, error_Z_act = tolerant_mean(Z_vec_active)
beta_act, error_beta_act = tolerant_mean(beta_vec_active)
"""plt.figure()
plt.plot(np.arange(len(Z_act)) + 1, Z_act)
plt.fill_between(np.arange(len(Z_act)) + 1, Z_act.data - error_Z_act,  Z_act.data + error_Z_act, color='C0', alpha=0.5)
plt.plot(np.arange(len(beta_act)) + 1, beta_act, "r")
plt.fill_between(np.arange(len(beta_act))+1, beta_act.data - error_beta_act, beta_act.data + error_beta_act, color = "r", alpha=0.5)
plt.show()"""

# Deterministic sampling rule
alpha = np.zeros((C, K)) # Arms proportions (random)
for x in range(C):
    alpha[x][0] = 1
P = np.array([[1,0,0], [0,1,0]])
r = 2

# Compute Lower Bound
T_star_det = compute_T_star_passive(Phi, theta, a_star, p_X, alpha, P)
print("Lower Bound Passive (deterministic):", T_star_det*np.log(1/delta))

# Run PCL-TaS
print("Running PCL-TaS (deterministic)...")
l_min, theta_hat, A_t, beta_vec, Z_vec, sample_complexity_vec_passive_det, best_arm_vec, err_vec = PCL_TaS(N_run, theta, Phi, g, u, delta, sigma, alpha, p_X, a_star, print_flag = False, P = P, r = r)
print("Sample complexity (avg +- std):", np.mean(sample_complexity_vec_passive_det), "+-",np.std(sample_complexity_vec_passive_det))
print("Error mean (avg +- std):", np.mean(err_vec), "+-", np.std(err_vec))
Z_pass_det, error_Z_pass_det = tolerant_mean(Z_vec)
beta_tot, error_beta_tot = tolerant_mean(beta_vec)
"""plt.figure()
plt.plot(np.arange(len(Z_pass_det)) + 1, Z_pass_det)
plt.fill_between(np.arange(len(Z_pass_det))+1, Z_pass_det.data - error_Z_pass_det,  Z_pass_det.data + error_Z_pass_det, color='C0', alpha=0.5)
plt.plot(np.arange(len(beta_tot)) + 1, beta_tot,"r")
plt.fill_between(np.arange(len(beta_tot))+1, beta_tot.data - error_beta_tot, beta_tot.data + error_beta_tot, color = "r", alpha=0.5)
plt.show()
"""

# Box plots
xs = ["PCL-TaS \n(random)", "CL-TaS", "PCL-TaS \n(deterministic)"]
vals = [sample_complexity_vec_passive, sample_complexity_vec_active, sample_complexity_vec_passive_det]
plt.figure()
#plt.title("$\Delta_{\min} = $" + str(gap))
plt.boxplot(vals, labels = xs, showfliers=False)
plt.scatter(np.random.normal(2, 0.02, len(vals[1])), vals[1],alpha = 0.25, c = "#89C417")
plt.scatter(np.random.normal(1, 0.02, len(vals[0])), vals[0],alpha = 0.25, c = "#E32219")
plt.scatter(np.random.normal(3, 0.02, len(vals[2])) ,vals[2],alpha = 0.25, c = "y")
plt.ylabel("Sample complexity")
plt.xticks([1, 2, 3], xs)
plt.axhline(T_star_passive*np.log(1/delta), c = "#E32219", linestyle = "--")
plt.axhline(T_star_active*np.log(1/delta), c = "#89C417", linestyle = "--")
plt.axhline(T_star_det*np.log(1/delta), c = "y", linestyle = "--")
tikzplotlib.save("figures/box_"+ str(gap)+ ".pgf")
plt.show()

"""plt.figure()
plt.plot(np.arange(len(Z_act)) + 1, Z_act)
plt.fill_between(np.arange(len(Z_act)) + 1, Z_act.data - error_Z_act,  Z_act.data + error_Z_act, color='C0', alpha=0.5)
plt.plot(np.arange(len(Z_pass_det)) + 1, Z_pass_det)
plt.fill_between(np.arange(len(Z_pass_det))+1, Z_pass_det.data - error_Z_pass_det,  Z_pass_det.data + error_Z_pass_det, color='C0', alpha=0.5)
plt.legend(["Active", "_", "Passive"])
plt.xlabel("$t$")
plt.ylabel("$Z(t)$")
plt.xlim([0, 300])
tikzplotlib.save("figures/box_"+ str(gap)+ ".pgf")"""