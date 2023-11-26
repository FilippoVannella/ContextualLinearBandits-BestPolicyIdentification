import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.simplefilter("error")

##### Lower Bound
def solve_inf_lb(theta, Phi, alpha, a_star, a):
    mu = cp.Variable(len(theta), pos = True)
    A = 0
    for a_for in range(Phi.shape[0]):
        A += np.outer(Phi[a_for], Phi[a_for])*alpha[a_for]
    obj = cp.quad_form((mu-theta),A)
    constr = [mu@(Phi[a_star] - Phi[a])<= 0]
    prob = cp.Problem(cp.Minimize(obj), constr)
    result = prob.solve()
    return result, mu.value

def compute_A(Phi, p_X, alpha):
    """Compute covariates matrix A(alpha)"""
    A = 0
    for x in range(Phi.shape[0]):
        for a in range(Phi.shape[1]):
            A += np.outer(Phi[x, a], Phi[x, a])*alpha[x, a]*p_X[x]
    return A

def compute_A_ind(Phi, x, p_X, alpha):
    """Compute (independent) covariates matrix for context x"""
    A = 0
    for a in range(Phi.shape[1]):
        A += np.outer(Phi[x, a], Phi[x, a])*alpha[x, a]*p_X[x]
    return A

def compute_T_star_active(Phi, p_X, theta, a_star):
    """Compute lower bound in the active learning setting using CVXPY"""
    alpha = cp.Variable((Phi.shape[0], Phi.shape[1]), pos = True)
    z = cp.Variable()
    A = compute_A(Phi, p_X, alpha)
    constraints = []
    for x in range(Phi.shape[0]):
        constraints.append(cp.sum(alpha[x]) == 1)
        for a in range(Phi.shape[1]):
            if a != a_star[x]:
                constraints.append(z >= 2*cp.matrix_frac(Phi[x, a_star[x]] - Phi[x,a], A)/(theta@(Phi[x, a_star[x]] - Phi[x,a]))**2)
    objective = cp.Minimize(z)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return result, alpha.value

def compute_T_star_passive(Phi, theta, a_star, p_X, alpha, P = None):
    """Compute lower bound in the passive learning setting"""
    if P is None: 
        P = np.eye(len(theta))
    A_inv = np.linalg.pinv(P@compute_A(Phi, p_X, alpha)@P.T)
    lb_list = []
    for x in range(Phi.shape[0]):
        for a in range(Phi.shape[1]):
            if a != a_star[x]:
                lb_list.append((2*(Phi[x, a_star[x]] - Phi[x, a]).T@P.T@A_inv@P@(Phi[x, a_star[x]] - Phi[x, a]))/(theta@P.T@P@(Phi[x, a_star[x]] - Phi[x, a]))**2)
    return np.max(lb_list)

def compute_T_star_ind(Phi, theta, a_star, p_X, alpha, P = None):
    """Compute lower bound in the independent (passive) learning setting"""
    if P is None: 
        P = np.eye(len(theta))
    lb_list = []
    for x in range(Phi.shape[0]):
        A_inv = np.linalg.pinv(P@compute_A_ind(Phi, x, p_X, alpha)@P.T)
        for a in range(Phi.shape[1]):
            if a != a_star[x]:
                lb_list.append((2*(Phi[x, a_star[x]] - Phi[x, a]).T@P.T@A_inv@P@(Phi[x, a_star[x]] - Phi[x, a]))/(theta@P.T@P@(Phi[x, a_star[x]] - Phi[x, a]))**2)
    return np.max(lb_list)

def compute_T_star_ind_x(Phi, x, theta, a_star, p_X, alpha, P = None):
    """Compute lower bound in the independent (passive) learning setting"""
    if P is None: 
        P = np.eye(len(theta))
    lb_list = []
    A_inv = np.linalg.pinv(P@compute_A_ind(Phi, x, p_X, alpha)@P.T)
    for a in range(Phi.shape[1]):
        if a != a_star[x]:
            lb_list.append((2*(Phi[x, a_star[x]] - Phi[x, a]).T@P.T@A_inv@P@(Phi[x, a_star[x]] - Phi[x, a]))/(theta@P.T@P@(Phi[x, a_star[x]] - Phi[x, a]))**2)
    return np.max(lb_list)

def compute_T_star_active_ind(Phi, p_X, theta, a_star):
    """Compute lower bound in the active learning setting using CVXPY"""
    alpha = cp.Variable((Phi.shape[0], Phi.shape[1]),pos = True)
    z = cp.Variable()
    constraints = []
    for x in range(Phi.shape[0]):
        A = compute_A_ind(Phi, x, p_X, alpha)
        constraints.append(cp.sum(alpha[x]) == 1)
        for a in range(Phi.shape[1]):
            if a != a_star[x]:
                constraints.append(z >= 2*cp.matrix_frac(Phi[x, a_star[x]] - Phi[x,a], A)/((theta@(Phi[x, a_star[x]] - Phi[x,a]))**2))
    objective = cp.Minimize(z)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return alpha, result

def solve_T_star_active(Phi, p_X, a_star, theta, x0 = None):
    """Compute lower bound in the active learning setting using Scipy"""
    C = Phi.shape[0]
    K = Phi.shape[1]
    # Define the optimization variables
    # Define the objective function and constraints
    def objective_function(x):
        return x[-1]

    def constraint(x):
        alpha = x[:-1].reshape((C, K))
        A_inv = np.linalg.pinv(compute_A(Phi, p_X, alpha))
        constraints = []
        for c in range(C):
            for a in range(K):
                if a != a_star[c]:
                    constraints.append(-2*((Phi[c, a_star[c]] - Phi[c,a]).T@A_inv@(Phi[c, a_star[c]] - Phi[c,a]))/(theta@(Phi[c, a_star[c]] - Phi[c,a]))**2 + x[-1])
        return constraints

    def eq_constraint(x):
        alpha = x[:-1].reshape((C, K))
        constraints = []
        for c in range(C):
            constraints.append(np.sum(alpha[c]) - 1)
        return constraints
    
    bounds = [(0, 1) for _ in range(C*K)]
    bounds.append((0, None))
    if x0 is None:
        x0 = np.ones(C*K + 1)/K

    result = minimize(objective_function, x0, constraints=[{'type': 'ineq', 'fun': constraint}, {'type': 'eq', 'fun': eq_constraint}], bounds = bounds)

    # Extract the optimal value and optimal solution
    optimal_value_scipy = result.fun
    optimal_solution_scipy = result.x[:-1].reshape((C, K))
    return optimal_value_scipy,optimal_solution_scipy

##### Algorithm
def reset(C, K, d):
    """Reset the parameters at the start of each run"""
    t = 0
    w = 0
    theta_hat = np.zeros(d)
    A_t = np.zeros((d, d))
    A_t_x = np.zeros((C, d, d))
    Z_t = 0
    beta = 0
    N_x = np.zeros(C)
    N_xa = np.zeros((C, K))
    return t, N_x, N_xa, w, theta_hat, A_t, A_t_x, Z_t, beta

def reset_ind(C, d):
    """Reset the parameters at the start of each run (for the independent TaS)"""
    t = 0
    w = np.zeros((C, d))
    theta_hat = np.zeros((C, d))
    A_t = np.zeros((C, d, d))
    Z_t = np.zeros(C)
    beta = np.zeros(C)
    return t, w, theta_hat, A_t, Z_t, beta

def PI_TaS(N_run, theta, Phi, g, u, delta, sigma, alpha, p_X, a_star, print_flag = False):
    C = Phi.shape[0]
    K = Phi.shape[1]
    d = Phi.shape[2]
    """ Run Passive Independent Track-and-Stop"""
    lamda_min_vec, best_arm_vec, sample_complexity_vec, err_vec, Z_off, beta_off = [], [], [], [], [], []
    a_star_hat = - np.ones(C)
    for trial in tqdm(range(N_run)):
        Z_vec, beta_vec = [], []
        if print_flag:
            print("Simulation number:", trial)
        t, w, theta_hat, A_t, Z_t, beta = reset_ind(C, d)
        stop_flag = False
        c = 0
        while not stop_flag:
            lamda_min_vec.append(find_min_eig(A_t))
            # Context-Action-Reward cycle
            c = np.random.choice(C, p = p_X) # sample context
            a = np.random.choice(K, p = alpha[c]) # sample action
            r = np.random.normal(Phi[c, a]@theta, sigma) # sample reward

            # Update theta_hat_t
            A_t[c] += np.outer(Phi[c, a], Phi[c, a])
            if find_min_eig(A_t[c]) > 1e-5:
                w[c] += Phi[c, a]*r
                theta_hat[c] = np.linalg.pinv(A_t[c])@w[c]
            
            # Update beta_t
            beta[c] = sigma**2*(1 + u)*np.log((np.linalg.det(A_t[c]/(u*g) + np.eye(d)))**(1/2)/delta)
            beta_vec.append(beta.copy())
            a_star_hat[c] = np.argmax(Phi[c]@theta_hat[c], axis = 0)

            # Compute Z
            if find_min_eig(A_t[c]) > 1e-5:
                #Z_t[c] = compute_Z_ind(c, theta_hat[c], Phi[c], K, np.linalg.pinv(A_t[c]))
                try:
                    Z_t[c] = t/compute_T_star_ind(Phi, theta_hat[c], a_star, p_X, alpha, P = None)
                except RuntimeWarning:
                     pass
                     #print("Runtime Error!!!")
            Z_vec.append(Z_t.copy())
            t += 1
            # print("t:", t, "Z:", (Z_t), "beta", beta, "lambda_min", find_min_eig(A_t[c]))
            # print("Z_c", Z_t[c])
            stop_flag = ((Z_t[c] > beta[c]) and (find_min_eig(A_t[c]) > g))
        if print_flag:
            print("Sample complexity:",t,"Z", Z_t,"beta", beta)

        sample_complexity_vec.append(t)
        best_arm_vec.append(a_star_hat)
        err_vec.append(not(eq(a_star, a_star_hat)))

        if print_flag:
            print("Cumulative error distribution:", np.mean(err_vec))
            print("Cumulative sample complexity:", np.mean(sample_complexity_vec))
        Z_off.append(Z_vec.copy())
        beta_off.append(beta_vec.copy())
        
        #print("here", t, find_min_eig(A_t),g)
    return lamda_min_vec, theta_hat,A_t, beta_off, Z_off, sample_complexity_vec, best_arm_vec, err_vec

def CL_TaS(A0, N_run, theta, Phi, dx_vec, g, u, delta, sigma, alpha, p_X, a_star, print_flag = False, T_max = 10000):
    C = Phi.shape[0]
    K = Phi.shape[1]
    d = Phi.shape[2]
    """ Run Contextual Linear Track-and-Stop"""
    print_flag = False
    best_arm_vec, alpha_out, sample_complexity_vec, err_vec, Z_off, beta_off, lamda_min_off, forced_expl_steps_vec,tracking_steps_vec = [], [], [], [], [], [], [], [], []
    T_star_hat = 1e6
    x0 = None 
    for trial in tqdm(range(N_run)):
        lamda_min_vec, alpha_vec, Z_vec, beta_vec, N_vec, forced_expl_steps, tracking_steps = [], [], [], [], [], [], []
        if print_flag:
            print("Simulation number:", trial)
        t, N_x, N_xa, w, theta_hat, A_t, A_t_x, Z_t, beta = reset(C, K, d)
        lamda_min_t = 0
        sum_A0 = []
        for c in range(C):
            lhs = 0
            for phi_ca in A0[c]:
                lhs += np.outer(phi_ca, phi_ca)
            sum_A0.append(lhs)

        # Run algorithm
        while ((Z_t < beta) or (lamda_min_t < g)): # Stopping condition
            c = np.random.choice(C, p = p_X) 
            N_x[c] += 1
            # Forced Exploration
            if (find_nth_min_eig(A_t_x[c] - np.sqrt(N_x[c]/dx_vec[c])*sum_A0[c], dx_vec[c]) <= 0):
                forced_expl_steps.append(t)
                l_min_vec_expl = []
                for phi_ca in A0[c]:
                    l_min_vec_expl.append(find_min_eig(A_t + np.outer(phi_ca, phi_ca)))
                a = np.argmax(l_min_vec_expl)
                N_vec.append(N_xa)
            # Tracking
            else:
                if (lamda_min_t > g):
                    tracking_steps.append(t)
                    if t % 1 == 0:
                        try:
                            #if T_star_hat != 1e6:
                            #    x0 = np.ones(K*C+1)
                            #    x0[:-1] = alpha.flatten()
                            #    x0[-1] = T_star_hat
                            sol = compute_T_star_active(Phi, p_X, theta_hat, a_star_hat) # 
                        except RuntimeWarning:
                            print("Runtimewarning", t, theta_hat)
                        except cp.SolverError:
                            print("SolverError!!!", "theta_hat", theta_hat)
                        T_star_hat = sol[0]
                        alpha = sol[1]
                        alpha_vec.append(alpha)
                    a = np.argmax(N_xa[c] - alpha[c])
            N_xa[c, a] += 1
            r = np.random.normal(Phi[c, a]@theta, sigma) # sample reward

            # Update theta_hat_t
            lamda_min_t = find_min_eig(A_t)
            A_t_x[c] += np.outer(Phi[c, a], Phi[c, a])
            A_t += np.outer(Phi[c, a], Phi[c, a])
            A_t_inv = np.linalg.pinv(A_t)
            w += Phi[c, a]*r
            theta_hat = A_t_inv@w

            # Update beta_t
            beta = sigma**2*(1 + u)*np.log((np.linalg.det(A_t/(u*g) + np.eye(d)))**(1/2)/delta)
            beta_vec.append(beta)
            a_star_hat = np.argmax(Phi@theta_hat, axis = 1)

            # Compute Z
            if (find_min_eig(A_t) > g):
                try:
                    Z_t = t/T_star_hat # compute_Z(theta_hat, Phi, A_t_inv)
                except RuntimeWarning:
                    print("runtimewarning at time",t, "T_star_hat",T_star_hat)
                Z_vec.append(Z_t)

            lamda_min_vec.append(lamda_min_t)
            t += 1

        sample_complexity_vec.append(t)
        best_arm_vec.append(a_star_hat)
        err_vec.append(not(eq(a_star, a_star_hat)))
        Z_off.append(Z_vec)
        beta_off.append(beta_vec)
        lamda_min_off.append(lamda_min_vec)
        alpha_out.append(alpha_vec)
        forced_expl_steps_vec.append(forced_expl_steps)
        tracking_steps_vec.append(tracking_steps)
        if print_flag:
            print("Cumulative error distribution:", np.mean(err_vec))
            print("Cumulative sample complexity:", np.mean(sample_complexity_vec))
    return alpha_vec, forced_expl_steps, tracking_steps, N_vec, lamda_min_vec, theta_hat,A_t, beta_off, Z_off, sample_complexity_vec, best_arm_vec, err_vec

def I_TaS(A0, N_run, theta, Phi, dx_vec, g, u, delta, sigma, alpha, p_X, a_star, print_flag = False, T_max = 10000):
    C = Phi.shape[0]
    K = Phi.shape[1]
    d = Phi.shape[2]
    """ Run Contextual Linear Track-and-Stop"""
    print_flag = False
    best_arm_vec, alpha_out, sample_complexity_vec, err_vec, Z_off, beta_off, lamda_min_off, forced_expl_steps_vec,tracking_steps_vec = [], [], [], [], [], [], [], [], []
    T_star_hat = 1e6
    x0 = None
    for trial in tqdm(range(N_run)):
        lamda_min_vec, alpha_vec, Z_vec, beta_vec, N_vec, forced_expl_steps, tracking_steps = [], [], [], [], [], [], []
        if print_flag:
            print("Simulation number:", trial)
        t, N_x, N_xa, w, theta_hat, A_t, A_t_x, Z_t, beta = reset(C, K, d)
        lamda_min_t = 0
        sum_A0 = []
        for c in range(C):
            lhs = 0
            for phi_ca in A0[c]:
                lhs += np.outer(phi_ca, phi_ca)
            sum_A0.append(lhs)
        
        # Run algorithm
        while ((Z_t < beta) or (lamda_min_t < g)): # Stopping condition
            c = np.random.choice(C, p = p_X) 
            N_x[c] += 1
            # Forced Exploration
            if (find_nth_min_eig(A_t_x[c] - np.sqrt(N_x[c]/dx_vec[c])*sum_A0[c], dx_vec[c]) <= 0):
                forced_expl_steps.append(t)
                l_min_vec_expl = []
                for phi_ca in A0[c]:
                    l_min_vec_expl.append(find_min_eig(A_t + np.outer(phi_ca, phi_ca)))
                a = np.argmax(l_min_vec_expl)
                N_vec.append(N_xa)
            # Tracking
            else:
                if (lamda_min_t > g):
                    tracking_steps.append(t)
                    if t % 1 == 0:
                        try:
                            #if T_star_hat != 1e6:
                            #    x0 = np.ones(K*C+1)
                            #    x0[:-1] = alpha.flatten()
                            #    x0[-1] = T_star_hat
                            sol = compute_T_star_active(Phi, p_X, theta_hat, a_star_hat) # 
                        except RuntimeWarning:
                            print("Runtimewarning", t, theta_hat)
                        except cp.SolverError:
                            print("SolverError!!!", "theta_hat", theta_hat)
                        T_star_hat = sol[0]
                        alpha = sol[1]
                        alpha_vec.append(alpha)
                    a = np.argmax(N_xa[c] - alpha[c])
            N_xa[c, a] += 1
            r = np.random.normal(Phi[c, a]@theta, sigma) # sample reward

            # Update theta_hat_t
            lamda_min_t = find_min_eig(A_t)
            A_t_x[c] += np.outer(Phi[c, a], Phi[c, a])
            A_t += np.outer(Phi[c, a], Phi[c, a])
            A_t_inv = np.linalg.pinv(A_t)
            w += Phi[c, a]*r
            theta_hat = A_t_inv@w

            # Update beta_t
            beta = sigma**2*(1 + u)*np.log((np.linalg.det(A_t/(u*g) + np.eye(d)))**(1/2)/delta)
            beta_vec.append(beta)
            a_star_hat = np.argmax(Phi@theta_hat, axis = 1)

            # Compute Z
            if (find_min_eig(A_t) > g):
                try:
                    Z_t = t/T_star_hat # compute_Z(theta_hat, Phi, A_t_inv)
                except RuntimeWarning:
                    print("runtimewarning at time",t, "T_star_hat",T_star_hat)
                Z_vec.append(Z_t)

            lamda_min_vec.append(lamda_min_t)
            t += 1

        sample_complexity_vec.append(t)
        best_arm_vec.append(a_star_hat)
        err_vec.append(not(eq(a_star, a_star_hat)))
        Z_off.append(Z_vec)
        beta_off.append(beta_vec)
        lamda_min_off.append(lamda_min_vec)
        alpha_out.append(alpha_vec)
        forced_expl_steps_vec.append(forced_expl_steps)
        tracking_steps_vec.append(tracking_steps)
        if print_flag:
            print("Cumulative error distribution:", np.mean(err_vec))
            print("Cumulative sample complexity:", np.mean(sample_complexity_vec))
    return alpha_vec, forced_expl_steps, tracking_steps, N_vec, lamda_min_vec, theta_hat,A_t, beta_off, Z_off, sample_complexity_vec, best_arm_vec, err_vec

def PCL_TaS(N_run, theta, Phi, g, u, delta, sigma, alpha, p_X, a_star, print_flag = True, P = None, r = None):
    """Run Passive Contextual Linear Track-and-Stop"""
    C = Phi.shape[0]
    K = Phi.shape[1]
    d = Phi.shape[2]
    lamda_min_vec,best_arm_vec, sample_complexity_vec, err_vec, Z_off, beta_off = [], [], [], [], [], []
    gamma_hat = np.zeros((C, K, d))
    if P is None:
        P = np.eye(d)
        r = d
    for trial in tqdm(range(N_run)):
        Z_vec, beta_vec = [], []
        if print_flag:
            print("Simulation number:", trial)
        t, N_x, N_xa, w, theta_hat, A_t, A_t_x, Z_t, beta = reset(C, K, d)
        while (Z_t < beta) or (find_min_eig(P@A_t@P.T -g*np.eye(r)) < 0):
            lamda_min_vec.append(find_min_eig(A_t))
            
            # Context-Action-Reward cycle
            c = np.random.choice(C, p = p_X) # sample context
            a = np.random.choice(K, p = alpha[c]) # sample action
            rew = np.random.normal(Phi[c, a]@theta, sigma) # sample reward

            # Update theta_hat_t
            A_t += np.outer(Phi[c, a], Phi[c, a])
            A_t_inv = np.linalg.pinv(P@A_t@P.T)
            w += Phi[c, a]*rew
            theta_hat = P.T@A_t_inv@P@w

            # Update beta_t
            beta = sigma**2*(1 + u)*np.log((np.linalg.det(P@A_t@P.T/(u*g) + np.eye(r)))**(1/2)/delta)
            beta_vec.append(beta)
            a_star_hat = np.argmax(Phi@theta_hat, axis = 1)
            for c in range(C):
                for a in range(K):
                    gamma_hat[c, a] = Phi[c, a_star_hat[c]] - Phi[c, a]

            # Compute Z
            if find_min_eig(P@A_t@P.T) > 1e-6:
                # Z_t = compute_Z(theta_hat, Phi, A_t_inv, P)
                Z_t = t/compute_T_star_passive(Phi, theta_hat, a_star_hat, p_X, alpha)
                Z_vec.append(Z_t)
            t += 1
            
        if print_flag:
            print("Z", Z_t, "beta", beta) #print("t:",t,"Z:",(Z_t),"beta", beta,"lambda_min",find_min_eig(A_t))

        sample_complexity_vec.append(t)
        best_arm_vec.append(a_star_hat)
        err_vec.append(not(eq(a_star, a_star_hat)))
        
        if print_flag:
            print("Cumulative error distribution:", np.mean(err_vec),"\nCumulative sample complexity:", np.mean(sample_complexity_vec))
        Z_off.append(Z_vec)
        beta_off.append(beta_vec)
    return lamda_min_vec, theta_hat,A_t, beta_off, Z_off, sample_complexity_vec, best_arm_vec, err_vec

##### Algebra
def is_basis_vector(vector, vectors):
  A = np.array(vectors) # coefficient matrix
  b = np.array(vector) # right-hand side vector
  try:
    np.linalg.solve(A, b) # Solve the system of linear equations
  except np.linalg.LinAlgError:
    return True # If the system has no solution, the vector is a basis vector
  else:
    return False  # If the system has a unique solution, the vector is not a basis vector

def find_minimal_size_basis(vectors):
  basis_vectors = []
  for vector in vectors:
    if is_basis_vector(vector, basis_vectors):
      basis_vectors.append(vector)
  return basis_vectors


def eq(a, b):
    """Vector equality: Returns boolean True if a == b"""
    return int(np.all(np.array(a) == np.array(b)))

def find_nth_min_eig(A, n):
    """Compute nth minimum eigenvalue of A"""
    return np.linalg.eigvals(A)[-n]

def find_min_eig(A):
    """Compute minimum eigenvalue of A"""
    return np.min(np.linalg.eigvals(A))

##### utilities

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)