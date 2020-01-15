import numpy as np
import math

# functions to execute dominant sets clustering algorithm
# http://homepage.tudelft.nl/3e2t5/HungKrose_ICMI2011.pdf

# fills an n_people x n_people matrix with affinity values.
def learned_affinity(n_people, truth_arr, frame, n_features):
	A = np.zeros((n_people, n_people))
	idx = 0
	always_counting_idx = -1
	for i in range(n_people):
		if frame[i*n_features+1] == 'fake':
			continue	
		for j in range(n_people):
			if frame[j*n_features+1] == 'fake':
				continue
			if i == j:
				continue
			A[i,j] += truth_arr[idx]/2
			A[j,i] += truth_arr[idx]/2
			idx += 1

	return A

# d-sets function k
def k(S, i, A):
	sum_affs = 0
	for j in range(len(S)):
		if S[j]:
			sum_affs += A[i,j]

	return 1/np.sum(S) * sum_affs

# d-sets function phi
def phi(S,i,j,A):
	return A[i,j] - k(S,i,A)

# d-sets function weight
def weight(S, i, A):
	if np.sum(S) == 1:
		return 1
	else:
		R = S.copy()
		R[i] = False
		sum_weights = 0
		for j in range(len(R)):
			if R[j]:
				sum_weights += phi(R,j,i,A) * weight(R, j, A)
				return sum_weights

## optimization function
def f(x, A):
	return np.dot(x.T, np.dot(A, x))

## iteratively finds vector x which maximizes f
def vector_climb(A, allowed, n_people, thres=1e-5):
	x = np.random.uniform(0,1,n_people)
	x = np.multiply(x, allowed)
	eps = 10
	n = 10
	while (eps > 1e-15):
		p = f(x,A)
		x = np.multiply(x, np.dot(A,x)) / np.dot(x, np.dot(A,x))
		n = f(x,A)
		eps = abs(n-p)

	groups = x > thres
 
	for i in range(n_people):
		if not allowed[i]:
			if weight(groups,i,A) > 0.0:
				return []
	return groups

# Finds vectors x of people which maximize f. Then removes those people and repeats
def iterate_climb_learned(predictions, frame, n_people, thres, n_features):
	allowed = np.ones(n_people)
	groups = []

	A = learned_affinity(n_people, predictions, frame, n_features)

	while (np.sum(allowed) > 1):
		A[allowed == False] = 0
		A[:,allowed == False] = 0
		if (np.sum(np.dot(allowed,A)) == 0):
			break
		x = vector_climb(A, allowed, n_people, thres=1e-5)
		if len(x) == 0:
			break
		groups.append(x)
		allowed = np.multiply(x == False, allowed)

	return groups

# Groups according to the algorithm in "Recognizing F-Formations in the Open World"
# https://ieeexplore.ieee.org/abstract/document/8673233
def naive_group(predictions, frame, n_people, thres, n_features):
	groups = []

	A = learned_affinity(n_people, predictions, frame, n_features)

	A = np.random.randn(n_people, n_people)
	A = A > .5
	for i in range(n_people):
		A[i, i] = True

	A = 1 * A
	while np.sum(A) > 0:
		most_overlap = -float("inf")
		pos = (-1, -1)

		for i in range(n_people-1):
			B_i = A[i]
			for j in range(i+1, n_people):
				B_j = A[j]
				overlap = B_i.dot(B_j)
				
				if overlap > most_overlap:
					most_overlap = overlap
					pos = (i, j)
		if most_overlap <= 0:
			break

		group = (A[pos[0]] + A[pos[1]]) > .5
		groups.append(group)
		for i in range(n_people):
			if group[i]:
				A[i, :] = 0
				A[:, i] = 0

	return groups




