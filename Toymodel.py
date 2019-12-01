# -*- coding: utf-8 -*-

###############################################################################
# Example toy model
# This program computes the left and the right diagram of the figure
# # for better numerical results choose N = 1000 (line 161 and 175) and M = 10000 (line 176)
###############################################################################

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

###############################################################################
# a is the scale factor
# n corresponds to the element t^n of the group G
# k_arr corresponds to a set of real numbers in K_{id}
###############################################################################

###############################################################################
# Definiton of functions
###############################################################################
# hess_V computes the Hessian matrix of the potential V
# Input: a coreesponds to the scale factor
# Output: a 3x2x2 numpy array
#         for n in {0, 1, 2} the 2x2 array hess_V(a)[n] correspondends to \partial_{t^n}\partial_{t^n}V(y_0)
def hess_V(a):
    re = np.zeros((3, 2, 2))
    re[1] = 6*a**(-8)*np.diag([-2*a**(-6)+1, 26*a**(-6)-7])
    re[2] = 2**(-4)*3*a**(-8)*np.diag([-1, 7])
    return re

# f_V computes the function f_V
# Input: a coreesponds to the scale factor
# Output: a 5x2x2 numpy array
#         for n in {-2, -1, 0, 1, 2} the 2x2 numpy array f_V(a)[n] corresponds to f_V(t^n)
def f_V(a):
    re = np.zeros((5, 2, 2))
    hess = hess_V(a)
    for n in supp_V:
        re[0] = re[0] + 2*hess[n]
        re[-n] = re[-n] - hess[n]
        re[n] = re[n] - hess[n]
    return re

# gg computes the functions g_R and g_{R,0,0}
# Input: a coreesponds to the scale factor
# Output: a list with two items
#         the first item is an 3x6x2 numpy array and corresponds to g_R
#         the second item is an 3x6x2 numpy array and corresponds to g_{R,0,0}
#         for n in {0, 1, 2} the 6x2 numpy array gg(a)[0][n] corresponds to g_R(t^n)
#         for n in {0, 1, 2} the 6x2 numpy array gg(a)[1][n] corresponds to g_{R,0,0}(t^n)
def gg(a):
    # Definition of the basis {b_1, b_2, b_3}
    b = np.zeros((6, 3))
    b[:, 0] = np.array([1, 0, 1, 0, 1, 0]) # b[:, 0] corresponds to b_1
    b[:, 1] = np.array([0, 1, 0, 1, 0, 1]) # b[:, 1] corresponds to b_2
    b[:, 2] = np.array([0, 0, -a, 0, -2*a, 0]) # b[:, 2] corresponds to b_3
    b_00 = b[:, 0:2] # b_00 corresponds to {b_1, b_2}
    # Let u_1,...,u_k be an orthonormal basis of a subspace, and let A denote the
    # n\times k matrix whose colums are u_1,...,u_k. Then the projection matrix is AA^T.
    # Calculation of orthonormal bases q and q_00
    q, r = np.linalg.qr(b)
    q_00, r_00 = np.linalg.qr(b_00)
    # Calculation of the projection matrix
    p = np.identity(6) - q@q.T # p is a projection matrix for the kernel \psi(U_{iso}(R))
    p_00 = np.identity(6) - q_00@q_00.T # p_00 is a projection matrix for the kernel \Psi(U_{iso,0,0}(R))
    # Calculation of g_R and g_R00 with supp(g_R) = supp(g_R00) = R
    # g_R00 corresponds to g_{R,0,0}
    g_R = np.zeros((3, 6, 2))
    g_R00 = np.zeros((3, 6, 2))
    for n in R:
        g_R[n]  = p[:, 2*n:2*n+2]
        g_R00[n]  = p_00[:, 2*n:2*n+2]
    return (g_R, g_R00)

# chi computes the function \chi_k
# Input: k_arr is a numpy array of real numbers and corresponds to numbers of K_{id}
#        n corresponds to t^n
#        a corresponds to the scale factor
# Output: a numpy array of complex numbers which correspond to the complex numbers \chi_k(t^n), k in k_arr
def chi(k_arr, n, a):
    return np.exp(np.pi*2j*n*a*k_arr)

# fou_f_V computes the function \fourier(f_V)
# Input: k_arr is a numpy array of real numbers and corresponds to numbers of K_{id}
#        a corresponds to the scale factor
# Output: a (np.size(k_arr))x2x2 numpy array of complex numbers which correspond to the 2x2 matrices \fourier(f_V)(\chi_k), k in k_arr
def fou_f_V(k_arr, a):
    f = f_V(a)
    re = np.zeros((np.size(k_arr), 2, 2), dtype=complex)
    for n in supp_f_V:
        re = re + np.multiply.outer(chi(k_arr, n, a), f[n])
    return re

# fou_gg computes the functions \fourier(g_R) and \fourier(g_{R,0,0})
# Input: k_arr is a numpy array of real numbers and corresponds to numbers of K_{id}
#        a is the scale factor
# Output: a list with two items
#         the first item is a (np.size(k_arr))x6x2 numpy array of complex numbers which correspond to the 6x2 matrices \fourier(g_R)(\chi_k), k in k_arr
#         the second item is a (np.size(k_arr))x6x2 numpy array of complex numbers which correspond to the 6x2 matrices \fourier(g_{R,0,0})(\chi_k), k in k_arr
def fou_gg(k_arr, a):
    g_R, g_R00 = gg(a)
    re = np.multiply.outer(k_arr, np.zeros((6, 2), dtype=complex))
    re_0 = np.multiply.outer(k_arr, np.zeros((6, 2), dtype=complex))
    for n in R:
        re = re + np.multiply.outer(chi(k_arr, n, a), g_R[n])
        re_0 = re_0 + np.multiply.outer(chi(k_arr, n, a), g_R00[n])
    return (re, re_0)

# lambdas_min_array computes \lambda_{min}(\fourier(f_V)(Ind \chi_k),\fourier(g_R)(Ind \chi_k)) and \lambda_{min}(\fourier(f_V)(Ind \chi_k),\fourier(g_{R,0,0})(Ind \chi_k))
# Input: k_arr is a numpy array of real numbers and corresponds to numbers of K_{id}
#        a corresponds to the scale factor
# Output: a real (np.size(k_arr))x2 numpy array
#         the first column corresponds to the real numbers \lambda_{min}(\fourier(f_V)(Ind \chi_k),\fourier(g_R)(Ind \chi_k)), k in k_arr
#         the second column corresponds to the real numbers \lambda_{min}(\fourier(f_V)(Ind \chi_k),\fourier(g_{R,0,0})(Ind \chi_k)), k in k_arr
def lambdas_min_array(k_arr, a):
    fou_f = fou_f_V(k_arr, a)
    fou_g_R, fou_g_R00 = fou_gg(k_arr, a)
    re = np.zeros((np.size(k_arr), 2))
    for i in range(np.size(k_arr)):
        b = linalg.eigvalsh(fou_f[i], fou_g_R[i].conjugate().T@fou_g_R[i])
        re[i, 0] = np.min(b)
        b = linalg.eigvalsh(fou_f[i], fou_g_R00[i].conjugate().T@fou_g_R00[i])
        re[i, 1] = np.min(b)
    return re

# k_array computes an appropriate set of k-values in K_{id}
# Input: N is an integer (large enough)
# Output: a numpy array of N real numbers which are uniformly distributed on K_{id}
def k_array(N, a):
    # We need this buffer since at k=0 and k=1/a the matrix (fourier(g_R)(\chi_k))^H*fourier(g_R(\chi_k)) is singular.
    buffer = 1/N/2*a
    return np.linspace(buffer, 1/a-buffer, N)

# lambdas computes \lambda_a and \lambda_{a, 0, 0}
# Input: N is an integer (large enough)
#        a corresponds to the scale factor
# Output: two numbers (an onedimensional numpy array of length two)
#         the first number corresponds to \lambda_a
#         the second number corresponds to \lambda_{a, 0, 0}
def lambdas(N, a):
    k_arr = k_array(N, a)
    lambdasmin = lambdas_min_array(k_arr, a)
    return np.amin(lambdasmin, axis=0)
###############################################################################


###############################################################################
# supp_V corresponds to the support of V, i.e. {t^1, t^2}
supp_V = [1, 2]
# supp_f_V corresponds to the support of f_V, i.e. {t^{-2}, t^{-1}, t^0, t^1, t^2}
supp_f_V = [-2, -1, 0, 1, 2]
# R corresponds to \mathcal R, i.e. {t^0, t^1, t^2}
R = [0, 1, 2]
###############################################################################


###############################################################################
# Plot of the left diagram
###############################################################################
N = 100 # choose the natural number N big enough for good numerical results, e.g. 1000
a = 1.22 # a corresponds to the scale factor
k_arr = k_array(N, a) # k_arr is a numpy array of N numbers which are uniformly distributed on K_{id}
lambdas_min_arr = lambdas_min_array(k_arr, a) # lambdas_min_arr is a numpy array with the values of the two functions
plt.rcParams['text.usetex'] = True
plt.figure(1)
plt.plot(k_arr, lambdas_min_arr) # plot of the two functions
plt.savefig('ToyModel_1.pdf')
###############################################################################


###############################################################################
# Plot of the right diagram
###############################################################################
N = 100 # choose the natural number N big enough for good numerical results, e.g. 1000
M = 100 # choose the natural number M big enough for good numerical results, e.g. 10000
a_arr = np.linspace(1.11, 1.25, N) # a_arr is a numpy array of N real numbers which are uniformly distributed on (1.11, 1.25) and correspond to a set of scale factors
lambda_a_arr = np.zeros(N)
lambda_a00_arr = np.zeros(N)
for i in range(N):
    lambda_a_arr[i], lambda_a00_arr[i] = lambdas(M, a_arr[i])
    if lambda_a_arr[i-1]<=0 and lambda_a_arr[i]>0:
        N_cri = i
plt.figure(2)
plt.ylim(bottom=-0.2, top=0.8)
plt.plot(a_arr[N_cri:N], lambda_a_arr[N_cri:N], a_arr, lambda_a00_arr)
plt.plot([(16/7)**(1/6), (26/7)**(1/6)], [0, 0], "o", color="tab:orange")
plt.savefig('ToyModel_2.pdf')
plt.figure(3)
plt.plot(a_arr[0:N_cri], lambda_a_arr[0:N_cri])
plt.plot([(16/7)**(1/6)], [0], "o", color="tab:blue")
plt.savefig('ToyModel_2_star.pdf')
###############################################################################