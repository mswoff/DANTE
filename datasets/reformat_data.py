import numpy as np
import re
import argparse
from shift_data import *

# primary set of functions for reformatting the data for use in the Tensorflow
# need to have created features and groups datasets
# main function gives coordiantes, affinities, and timechanges


# generates feature and ground-truth group matrices from data files
def import_data(dataset):
	dataset = str(dataset)
	path = "../datasets/" # this should get you to these files, edit if it doesn't
	Positions = np.genfromtxt(path + dataset + "/DS_utils/features.txt", dtype = 'str')
	Groups = np.genfromtxt(path + dataset + "/DS_utils/group_names.txt", dtype = 'str', delimiter = ',')
	return Positions, Groups

# run this to generate Groups_at_time, Groups is from import_gc_data()
# Groups is of the form: time < ID001 ID002 > < ID003 > etc.
# returns dictionary from time to array of group arrays
# eg. time -> [[ID001, ID002], [ID003], ...]
def add_time(Groups):
	Groups_at_time = {}
	for groups in Groups:
		groups_arr = re.split(" < | > < ", groups)
		Groups_at_time[groups_arr[0]] = []
		last_index = -1

		for group in groups_arr[1:]:
			last_index += 1
			Groups_at_time[groups_arr[0]].append(re.split(" ",group))

		# remove last > character
		if len(groups_arr[1:])==0:
			continue
		Groups_at_time[groups_arr[0]][last_index] = Groups_at_time[groups_arr[0]][last_index][:-1]

	return Groups_at_time

def compute_data_shift(Positions, time, n_people, augment_flipped_data=True):
	frame_idx = list(Positions[:,0]).index(time)
	frame = Positions[frame_idx]
	people = frame[1:]

	ij_frames = []
	flipped_frames = []
	# don't shift fake (placeholder) people
	for i in range(n_people):
		if people[i*4+1] == 'fake':
			continue
		for j in range(n_people):
			if i == j:
				continue
			if people[j*4+1] == 'fake':
				continue
			time_name = str(time) + ':' + str(i) + ':' + str(j) + ':'

			ij_frame = shift_indiv_data_standard(people, i, j, time_name, n_people)
			ij_frames.append(ij_frame)
			
			if augment_flipped_data:
			    flipped_frame = augment_frame_standard(ij_frame)
			    flipped_frames.append(flipped_frame)
	
	if augment_flipped_data:
            shifted_coordinates = np.concatenate((ij_frames, flipped_frames), axis=0)
	else:
	    shifted_coordinates = ij_frames
	return shifted_coordinates
	
# 	if augment_flipped_data:
# 		shifted_coordinates = np.concatenate(shifted_coordinates, flipped_frames)
# 
# 	return shifted_coordinates
	# if len(Shifted_Coordinates) == 0:
# 		Shifted_Coordinates = ij_frames
# 	else:
# 		Shifted_Coordinates = np.concatenate((Shifted_Coordinates, ij_frames), axis = 0)
#
# 	Shifted_Coordinates = np.concatenate((Shifted_Coordinates, flipped_frames), axis = 0)
# 	return Shifted_Coordinates

# Positions = matrix with row of form ID001 feature1, feature2, ..., ID002, feature1, ...
# Groups_at_time = dictionary from time to array of group arrays
# n_people = max number of people at a time. could be "fake"
# dataset = name of dataset. eg coffeebreak

## returns a matrix where each row for a person i,j is: time:i:j:orientation, ID001, feature1, feature2, ... ID00i, feature1, .., ID00j, ..., feature1, ...
##  each row is the coordinates at time:i:j:orientation, and contains the adjusted features (angles to sin and cos for instance) centered at people i and j
def shift_all_data_standard(Positions, Groups_at_time, n_people, dataset):
	print('standard shifting')
	Shifted_Coordinates = []

	np.savetxt('../datasets/' + dataset + '/coordinates.txt', Shifted_Coordinates, fmt = '%s')
	f = open('../datasets/' + dataset + '/coordinates.txt', 'ab')
	for time in Groups_at_time:
		frame_shifted_coordinates = compute_data_shift(Positions, time, n_people, augment_flipped_data=True)
		# Shifted_Coordinates = np.concatenate(Shifted_Coordinates = frame_shifted_coordinates)
		np.savetxt(f,frame_shifted_coordinates, fmt='%s')
		# np.savetxt(f,Shifted_Coordinates, fmt='%s')
		Shifted_Coordinates = []
	f.close()

	return Shifted_Coordinates

# Positions = matrix with row of form ID001 feature1, feature2, ..., ID002, feature1, ...
# Groups_at_time = dictionary from time to array of group arrays
# n_people = max number of people at a time. could be "fake"
# n_features = number of features before augmentation including name. eg (ID001, x, y, theta) = 4
# n_augmented_features = number of features after augmentation including name. eg (ID001, x, y, cos(theta), sin(theta)) = 5
# velocities = boolen if we use velocities as features
# dataset = name of dataset. eg coffeebreak

## returns a matrix where each row for a person i,j is: time:i:j:orientation, ID001, feature1, feature2, ... ID00i, feature1, .., ID00j, ..., feature1, ...
##  each row is the coordinates at time:i:j:orientation, and contains the adjusted features (angles to sin and cos for instance) centered at people i and j
def shift_all_data_nonstandard(Positions, Groups_at_time, n_people, n_features, n_augmented_features, velocities, dataset):
	Shifted_Coordinates = []
	prev_people = []
	counter = 0

	np.savetxt('../datasets/' + dataset + '/coordinates.txt', Shifted_Coordinates, fmt = '%s')
	f = open('../datasets/' + dataset + '/coordinates.txt', 'ab')
	for time in Groups_at_time:
		frame_idx = list(Positions[:,0]).index(time)
		frame = Positions[frame_idx]
		people = frame[1:]

		ij_frames = []
		flipped_frames = []

		# this is for velocities in the FM dataset. every 200 frames, the location changes, so velocities become
		# pointless
		if counter%200 == 0:
			print(counter)
			prev_people = people

		# skip fake (placeholder people)
		for i in range(n_people):
			if people[i*n_features+1] == 'fake':
				continue
			for j in range(n_people):
				if i == j:
					continue
				if people[j*n_features+1] == 'fake':
					continue
				time_name = str(time) + ':' + str(i) + ':' + str(j) + ':'

				ij_frame = shift_indiv_data_nonstandard(people, i, j, time_name, n_people=n_people, n_features=n_features, velocities =velocities, prev_people = prev_people)

				flipped_frame = augment_frame_nonstandard(ij_frame, n_features=n_augmented_features)


				ij_frames.append(ij_frame)
				flipped_frames.append(flipped_frame)

		counter += 1
		prev_people = people
		if len(Shifted_Coordinates) == 0:
			Shifted_Coordinates = ij_frames
		else:
			Shifted_Coordinates = np.concatenate((Shifted_Coordinates, ij_frames), axis = 0)
		
		Shifted_Coordinates = np.concatenate((Shifted_Coordinates, flipped_frames), axis = 0)

		np.savetxt(f,Shifted_Coordinates, fmt='%s')
		Shifted_Coordinates = []
	f.close()

	return Shifted_Coordinates


# Shifted_Coordinates = output of shift_all_data
# Groups_at_time dictionary from time to groups

## returns affinities = matrix of affinities at time:i:j:orientation
## returns timechanges = array of all the indicies of affinities matrix at which the actual time changes
def affinities_and_timechanges(Shifted_Coordinates, Groups_at_time):
	affinities = []
	i = 0
	prev_time_arr = [-1,-1]
	timechanges = []
	for frame in Shifted_Coordinates:
		time = frame[0]
		time_arr = [time.split(':')[0], time.split(':')[3]]
		if prev_time_arr[0] != time_arr[0] or prev_time_arr[1] != time_arr[1]:
			timechanges.append(i)
		i += 1
		prev_time_arr = time_arr

		affinity = indiv_affinities_at_time(time, Groups_at_time)
		affinities.append([time, affinity])
	affinities = np.array(affinities)
	affinities = affinities.astype(str)
	return affinities, timechanges

# returns groundtruth affinity. (if people i and j are in the same group)
def indiv_affinities_at_time(time, Groups_at_time):
	time_arr = time.split(':')
	group_time = time_arr[0]
	i = int(time_arr[1])
	j = int(time_arr[2])
	for g in Groups_at_time[group_time]:
		if 'ID_00' + str(i+1) in g:
			if 'ID_00' + str(j+1)in g:
				return 1
			return 0
	return 0


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help="which dataset to use", required=True)
    parser.add_argument('-p', '--n_people', help="number of people in network", type=int, default=15)
    parser.add_argument('-f', '--n_features', help="number of features being used, including name", type=int, default=4)
    parser.add_argument('-a', '--n_aug_features', help="number of augmented features being used, not including name", type=int, default=4)
    parser.add_argument('-v', '--use_velocities', help="use velocities as a feature", type=bool, default=False)

    return parser.parse_args()

# once you have made, in the DS_utils folder, a features.txt file and a group_names.txt file,
# run the main function to generate affinities.txt, coordinates.txt, and timechanges.txt
# affinites.txt is a file which has the ground truth affinity for every combination of people i and j at every time,
#    flipped horizontally, vertically, or both. (denoted time:i:j:orientation)
#    orientation 000 is normal, 001 is flipped over the horizontal axis, and person i is on the left and person j on the right (so j:i is flipped over the vertical axis)
# timechanges.txt is an array of integers, where each integer represents the row of affinities.txt where the time changes
# coordinates.txt at each row is the coordinates at time:i:j:orientation, and contains the adjusted features (angles to sin and cos for instance) centered at people i and j
if __name__ == "__main__":
	args = get_args()
	dataset = args.dataset
	n_features = args.n_features
	use_velocities = args.use_velocities
	n_people = args.n_people
	n_augmented_features = args.n_aug_features

	print("importing data..")
	Positions, Groups = import_data(dataset)
	print("data imported")
	Groups_at_time = add_time(Groups)
	print('groups at time made')

	if n_features == 4 and use_velocities == False:
		shift_all_data_standard(Positions, Groups_at_time, n_people, dataset)
	else:
		shift_all_data_nonstandard(Positions, Groups_at_time, n_people=n_people, n_features=n_features, n_augmented_features = n_augmented_features, velocities = use_velocities, dataset = dataset)

	Shifted_Coordinates = np.genfromtxt('../datasets/' + dataset + '/coordinates.txt', dtype = 'str', delimiter = ' ')
	print("data shifted")
	affinites, timechanges = affinities_and_timechanges(Shifted_Coordinates, Groups_at_time)
	print("shifted affinity file generated")

	np.savetxt('../datasets/' + dataset + '/timechanges.txt', timechanges, fmt = '%s')
	np.savetxt('../datasets/' + dataset + '/affinities.txt', affinites, fmt = '%s')
	print("Coordinates (features) file saved")
	print("file generation of reformatted data done. ready for build_dataset.py")


