import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class FourBox:
    def __init__(self, C, κ, ϵ, F):
        self.C = C
        self.κ = κ
        self.ϵ = ϵ
        self.F = F

        self.A = A_matrix(C, κ, ϵ)
        self.B = B_matrix(C)

    def __repr__(self):
        return f'FourBox(C={self.C}, κ={self.κ}, ϵ={self.ϵ}, F={self.F})'
    
    def timescales(self):
        eigvals = np.linalg.eigvals(self.A)
        tau = -1 / eigvals
        return tau
    
    def step(self, n, method='analytical'):
        if method == 'analytical':
            return step_analytical(self.A, self.B, self.F, n)
        elif method == 'numerical':
            return step_numerical(self.A, self.B, self.F, n)
        else:
            raise ValueError("Method must be 'analytical' or 'numerical'.")
    
    def observe(self, x):
        κ = self.κ
        ϵ = self.ϵ
        F = self.F
        tas = x[:, 0]
        rtmt = F - κ[0]*tas + (1 - ϵ)*κ[3]*(x[:, 2] - x[:, 3])
        y = np.column_stack((tas, rtmt))
        return y
    
    def forward(self, n, method='analytical'):
        x = self.step(n, method)
        y = self.observe(x)
        return y

def A_matrix(C, κ, ϵ):
    A = np.zeros((4, 4))
    A[0, :] = np.array([-(κ[0] + κ[1]), κ[1], 0, 0]) / C[0]
    A[1, :] = np.array([κ[1], -(κ[1] + κ[2]), κ[2], 0]) / C[1]
    A[2, :] = np.array([0, κ[2], -(κ[2] + ϵ*κ[3]), ϵ*κ[3]]) / C[2]
    A[3, :] = np.array([0, 0, κ[3], -κ[3]]) / C[3]
    return A

def B_matrix(C):
    B = np.zeros((4, 1))
    B[0] = 1/C[0]
    return B

def step_numerical(A, B, F, n):
        def ode_system(t, y):
            gradient = A @ y.reshape(-1, 1) + B * F
            return gradient.flatten()

        y0 = np.zeros(4)
        t_span = (0, n)
        t_eval = np.arange(1, n + 1)
        sol = solve_ivp(ode_system, t_span, y0, 'RK45', t_eval)
        return sol.y.T

def step_analytical(A, B, F, n):
    # Eigendecomposition: A = V D V_inv
    eigvals, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)
    # Time steps (n,)
    t = np.arange(1, n + 1)
    # Compute exp(D * t) for all t: shape (n, 4)
    exp_diag = np.exp(np.outer(t, eigvals))  # (n, 4)
    # For each t, expA_t = V @ diag(exp_diag[t]) @ V_inv
    # Stack all expA_t: (n, 4, 4)
    expA_t = np.einsum('ij,nj,jk->nik', V, exp_diag, V_inv)
    # delta = expA_t - I
    delta = expA_t - np.eye(4)
    # (n, 4, 4) @ (4, 1) -> (n, 4, 1)
    delta_BF = np.matmul(delta, B * F)
    # Solve A x = delta_BF for each t (vectorized)
    # np.linalg.solve(A, ...) expects (..., 4), so squeeze last dim
    x = np.linalg.solve(A, delta_BF.squeeze(-1).T).T  # (n, 4)
    return x

def pack(C, κ, ϵ, F):
    theta = np.concatenate((C, κ, [ϵ], [F]))
    log_theta = np.log(theta)
    return log_theta

def unpack(log_theta):
    theta = np.exp(log_theta)
    C = theta[:4]
    κ = theta[4:8]
    ϵ = theta[8]
    F = theta[9]
    return C, κ, ϵ, F

def penalty(tau, target=np.array([1., 10., 100., 1000.])):
    log_ratio = np.log(tau) - np.log(target)
    return np.sum(log_ratio**2)

def mse_loss(log_theta, y, alpha=0):
    C, κ, ϵ, F = unpack(log_theta)
    model = FourBox(C, κ, ϵ, F)
    y_pred = model.forward(y.shape[0], method='analytical')
    mse = np.mean((y - y_pred)**2) # data loss
    tau = model.timescales()
    penalty_value = penalty(tau) # penalty on timescales
    return mse + alpha*penalty_value

def fit_model(y, alpha=0, C_init=None, κ_init=None, ϵ_init=None, F_init=None):
    if C_init is None:
        C_init = [5., 20., 80., 150.]
    if κ_init is None:
        κ_init = [1., 1.5, 0.75, 0.5]
    if ϵ_init is None:
        ϵ_init = 1.
    if F_init is None:
        F_init = 5.9
    
    log_theta_init = pack(C_init, κ_init, ϵ_init, F_init)
    res = minimize(mse_loss, log_theta_init, args=(y, alpha), method='L-BFGS-B')
    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)
    C, κ, ϵ, F = unpack(res.x)
    model = FourBox(C, κ, ϵ, F)
    return model, res
