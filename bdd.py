import sklearn as skl
import sklearn.datasets as skd
from sklearn.neighbors import kneighbors_graph
import numpy as np

# Workaround for mlrose issue. Must get fixed by mlrose!
import six, sys; sys.modules['sklearn.externals.six'] = six 

import mlrose
import time


def cq_fitness(a, K, d, max_a):
	sum_a = 1.0
	#print(f"{K.shape} {a.shape} {d.shape}")
	f_quad = np.matmul(a,np.matmul(K,a)) + np.matmul(a,d) # Fitness term (unconstrained)
	a = sum_a * (a * (1.0/np.sum(a))) # Normalise
	return  np.Inf if np.max(a) > max_a else f_quad # Apply constraint

def find_valid_initial_state(K, d, max_a):
	dim = len(K)
	init_state = max_a*np.random.rand(dim)
	i = 0
	while np.isinf(cq_fitness(init_state, K, d, max_a)):
		i += 1
		init_state = max_a*np.random.rand(dim)
	#print(fitness_f(init_state))
	#print(i)
	return init_state

compute_similarity = lambda x, a, X_t, kernel_function: np.sum([a[i]*kernel_function(x,X_t[i,:]) for i in range(len(a)) if a[i]>0])

def solve_svdd_direct(X_t, kernel_fn, nu):
	n_t = len(X_t)
	K = skl.metrics.pairwise.pairwise_kernels(X_t, metric=kernel_fn, n_jobs=-1)
	d = -np.diag(K)
	max_a = (1.0/nu) * (1.0/n_t)
	sum_a = 1.0

	init_state = find_valid_initial_state(K, d, max_a)
	print(f"starting with fitness of {cq_fitness(init_state, K, d, max_a)} and initial sparisty of {n_t - np.count_nonzero(init_state)}")

	problem = mlrose.ContinuousOpt(length = n_t, fitness_fn = mlrose.CustomFitness(lambda a: cq_fitness(a, K, d, max_a)), maximize = False, min_val=0,max_val=max_a,step=max_a/100)
	solution, value = mlrose.random_hill_climb(problem,max_attempts=1000,max_iters=np.Inf,restarts=1,init_state=init_state)
	
	solution = sum_a * (solution * (1.0/np.sum(solution)))

	print(f'Final fitness: {value} Final max coeff: {np.max(solution)} Final sparsity: {n_t - np.count_nonzero(solution)}')

	return solution

def solve_bdd_with_uniform_prior(X_t, kernel_fn, nu):
	n_t = len(X_t)

	K = (skl.metrics.pairwise.pairwise_kernels(X_t, metric=kernel_fn, n_jobs=-1))
	D1 = np.sum(K, axis=1) # See Eq. 11 in the paper	d = -np.diag(K)

	C = np.eye(n_t)
	m = - D1

	return solve_bdd_generic_with_kernel_matrix(K, nu, C, m)

def solve_bdd_semisupervised(X_t, U, kernel_fn, nu): # U is the matrix of unlabeled data samples
	n_t = len(X_t)
	n_u = len(U)
	
	K_all = (skl.metrics.pairwise.pairwise_kernels( np.vstack((X_t, U)) , metric=kernel_fn, n_jobs=-1))
	D1 = np.sum(K_all, axis=1) # See Eq. 11 in the paper

	W = kneighbors_graph(K_all, 4, mode='distance', metric= 'euclidean',  include_self=False).toarray()
	D = np.diag(np.sum(W, axis=1)) # Diagonal matrix with each diagonal element being the summ of its corresponding row in W 
	L = D - W # Laplcian. See Eq. 14 and the pargraph above it
	L_inv = np.linalg.pinv(L) # Use pseudo-inverse to avoid singular matrix errors

	C = np.linalg.inv( L_inv[0:n_t, 0:n_t] )
	m = - D1[0:n_t]
	K = K_all[0:n_t, 0:n_t]

	return solve_bdd_generic_with_kernel_matrix(K, nu, C, m)

def solve_bdd_generic_with_data_samples(X_t, kernel_fn, nu, C, m):
	K = (skl.metrics.pairwise.pairwise_kernels(X_t, metric=kernel_fn, n_jobs=-1))
	return solve_bdd_generic_with_kernel_matrix(K, nu, C, m)

def solve_bdd_generic_with_kernel_matrix(K, nu, C, m):
	n_t = len(K)
	C_inv = np.linalg.inv(C)
	max_a = (1.0/nu) * (1.0/n_t)
	sum_a = 1.0

	D1 = np.sum(K, axis=1) # See Eq. 11 in the paper
	
	d = -2 * (D1 + np.matmul(C_inv, m))
	Q = n_t * K + C_inv
	

	init_state = find_valid_initial_state(Q, d, max_a)
	print(f"starting with fitness of {cq_fitness(init_state, Q, d, max_a)} and initial sparisty of {n_t - np.count_nonzero(init_state)}")

	problem = mlrose.ContinuousOpt(length = n_t, fitness_fn = mlrose.CustomFitness(lambda a: cq_fitness(a, Q, d, max_a)), maximize = False, min_val=0,max_val=max_a,step=max_a/100)
	solution, value = mlrose.random_hill_climb(problem,max_attempts=1000,max_iters=np.Inf,restarts=1,init_state=init_state)
	
	solution = sum_a * (solution * (1.0/np.sum(solution)))

	print(f'Final fitness: {value} Final max coeff: {np.max(solution)} Final sparsity: {n_t - np.count_nonzero(solution)}')

	return solution



if __name__=='__main__':

	n_t = 200
	n_v = 100
	n_u = 50
	k = 20 # Num. dimensions


	X, y  = skl.datasets.make_blobs(n_samples=[n_t + n_v + n_u//2, n_u // 4, n_u // 4 , 50 , 80 ],n_features=k)

	i_in = np.argwhere(y==0).flatten()
	i_out = np.argwhere(y>=3).flatten()
	i_out_unl_1 = np.argwhere(y==1).flatten()
	i_out_unl_2 = np.argwhere(y==2).flatten()

	X_tr = X[i_in[:n_t],:]

	X_va_in = X[i_in[n_t:n_t+n_v],:]
	X_va_out = X[ i_out,:]

	X_unl_out = X[ np.hstack((i_out_unl_2, i_out_unl_1)) ,:]
	X_unl_in = X[i_in[n_t+n_v:],:]
	U =  np.vstack((X_unl_in, X_unl_out))

	print(f"{len(X_tr)} training samples, {len(X_va_in)} inlier test samples, {len(X_va_out)} outlier test samples, {len(U)} unlabeled samples. ")


	kernel_function = lambda X,Y: skl.metrics.pairwise.rbf_kernel(np.row_stack((X,Y)),gamma=1.0/k)[0,1]
	
	a = solve_bdd_semisupervised(X_tr, U, kernel_function, nu = .01)


	print('Scores for inlier test samples')
	for x in X_va_in:
		score = compute_similarity(x, a, X_tr, kernel_function)
		print('%.2e'%score,end=' ')

	print('\n')
	print('Scores for outlier test samples')

	for x in X_va_out:
		score = compute_similarity(x, a, X_tr, kernel_function)
		print('%.2e'%score,end=' ')
		
	print('\n')
