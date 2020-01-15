import math
import numpy as np
import random

# "standard" form is (x, y, theta) for each person
# returns adjusted coordinates centered between i and j (x, y, cos(theta), sin(theta))
def shift_indiv_data_standard(people, i, j, time, n_people):
	x1 = float(people[i*4+1])
	y1 = float(people[i*4+2])
	x2 = float(people[j*4+1])
	y2 = float(people[j*4+2])
	a = .5*(x1+x2)
	b = .5*(y1+y2)
	dx = x1-x2
	dy = y1-y2
	b0 = dx/np.sqrt(dx**2+dy**2)
	b1 = dy/np.sqrt(dx**2+dy**2)


	new_frame = []
	new_frame.append(time + "000")

	feasible_surroundings = []
	for person in range(n_people):
		if person == i or person == j or people[person*4+1] == 'fake':
			continue
		else:
			feasible_surroundings.append(person)

	for person in range(n_people):
		# if find fake person, we replace with real person not i or j. this allows us to always have the same
		# number of surrounding people in the NN, and takes advantage of the symmetric function of surroundings
		if people[person*4+1] == 'fake':
			if len(feasible_surroundings) == 0:
				# new_frame.append('fake')
				new_frame.append(50000)
				new_frame.append(50000)
				new_frame.append(0.0)
				new_frame.append(0.0)
				continue

			p = random.choice(feasible_surroundings)
		else:
			# this is the normal case
			p = person
		
		name = people[p*4]
		x = float(people[p*4+1])
		y = float(people[p*4+2])
		theta = float(people[p*4+3])

		# shift, project each x and y
		x_shift = x-a
		y_shift = y-b
		x_proj = b0*x_shift + b1*y_shift
		y_proj = b1*x_shift - b0*y_shift

		tx = np.cos(theta)
		ty = np.sin(theta)

		tx_proj = b0*tx + b1*ty
		ty_proj = b1*tx - b0*ty

		# don't add i,j ppl for global context, save for end of feature array
		if p == i:
			x_i_name = name
			x_i_proj = x_proj
			y_i_proj = y_proj
			txi_proj = tx_proj
			tyi_proj = ty_proj
			continue

		elif p == j:
			x_j_name = name
			x_j_proj = x_proj
			y_j_proj = y_proj
			txj_proj = tx_proj
			tyj_proj = ty_proj
			continue

		new_frame.append(x_proj)
		new_frame.append(y_proj)
		new_frame.append(tx_proj)
		new_frame.append(ty_proj)

	# repeat individual coordinates for last 2 rows, appending i and j people to the end of array
	new_frame.append(x_i_proj)
	new_frame.append(y_i_proj)
	new_frame.append(txi_proj)
	new_frame.append(tyi_proj)

	new_frame.append(x_j_proj)
	new_frame.append(y_j_proj)
	new_frame.append(txj_proj)
	new_frame.append(tyj_proj)

	return new_frame


# "standard" form is originall (x, y, theta) for each person
# and then the input to this function is the shifted_coordinates centered between i and j (x, y, cos(theta), sin(theta))
# returns frame flipped over the horizontal axis (flipped vertically)
def augment_frame_standard(frame):
	time = frame[0]
	people = frame[1:]
	time_h = time[:-3] + "001"

	frame_h = [time_h]
	for i in range(int(len(people)/4)):
		x = float(people[i*4+0])
		y = float(people[i*4+1])
		tx = float(people[i*4+2])
		ty = float(people[i*4+3])

		x_h = x
		y_h = -1*y
		tx_h = tx
		ty_h = -1*ty

		frame_h.append(x_h)
		frame_h.append(y_h)
		frame_h.append(tx_h)
		frame_h.append(ty_h)

	return frame_h


# "standard" form is (x, y, theta) for each person. nonstandard is otherfeatures
# for other datasets, the user may need to add different flags and cases
# in these cases, set n_features = to number of features in non-adjusted features.txt
# set velocities = True if you want to use velocity estimates of people
# 	note: we only do this in FM_SYNTH, in which the location changes every 200 frames
#	and thus we don't include a velocity every 100 frames. the user would need to change this depending
#   on the dataset
# prev_people, used for velocity calculation, an array of the people from the previous timestamp
def shift_indiv_data_nonstandard(people, i, j, time, n_people, n_features, velocities, prev_people):
	x1 = float(people[i*n_features+1])
	y1 = float(people[i*n_features+2])
	x2 = float(people[j*n_features+1])
	y2 = float(people[j*n_features+2])
	a = .5*(x1+x2)
	b = .5*(y1+y2)
	dx = x1-x2
	dy = y1-y2
	b0 = dx/np.sqrt(dx**2+dy**2)
	b1 = dy/np.sqrt(dx**2+dy**2)


	new_frame = []
	new_frame.append(time + "000")

	feasible_surroundings = []
	for person in range(n_people):
		if person == i or person == j or people[person*n_features+1] == 'fake':
			continue
		else:
			feasible_surroundings.append(person)

	for person in range(n_people):
		# if find fake person, replace with real person not i or j	
		if people[person*n_features+1] == 'fake':
			if len(feasible_surroundings) == 0:
				new_frame.append(50000)
				new_frame.append(50000)
				if not velocities:
					if n_features > 3:
						new_frame.append(math.pi/2)
					if n_features == 5:
						new_frame.append(math.pi/2)
				elif velocities:
					new_frame.append(2)
					new_frame.append(10)
				continue
			p = random.choice(feasible_surroundings)
		else:
			# otherwise, we're good
			p = person
		
		name = people[p*n_features]
		x = float(people[p*n_features+1])
		y = float(people[p*n_features+2])
		if n_features > 3:
			theta = float(people[p*n_features+3])
		if n_features == 5:
			theta_head = float(people[p*n_features+4])
		# theta = float(orientations[p*2+1])
		if n_features > 3:
			if theta > math.pi:
				theta = theta-2*math.pi

		if n_features == 5 and theta_head > math.pi:
			theta_head = theta_head - 2*math.pi


		# shift, project each x and y
		x_shift = x-a
		y_shift = y-b
		x_proj = b0*x_shift + b1*y_shift
		y_proj = b1*x_shift - b0*y_shift

		if velocities:
			vx = float(people[p*n_features+1])-float(prev_people[p*n_features+1])
			vy = float(people[p*n_features+2])-float(prev_people[p*n_features+2])
			vx_proj = b0*vx + b1*vy
			vy_proj = b1*vx - b0*vy

		# and for theta also
		if not velocities:
			if n_features > 3:
				tx = np.cos(theta)
				ty = np.sin(theta)

				tx_proj = b0*tx + b1*ty
				ty_proj = b1*tx - b0*ty

			if n_features == 5:
				# in our personal case this meant there were head and body angles
				tx_head = np.cos(theta_head)
				ty_head = np.sin(theta_head)

				tx_head_proj = b0*tx_head + b1*ty_head
				ty_head_proj = b1*tx_head - b0*ty_head


		# don't add i,j ppl for global context
		if p == i:
			x_i_name = name
			x_i_proj = x_proj
			y_i_proj = y_proj
			if n_features > 3:
				tx_i_proj = tx_proj
				ty_i_proj = ty_proj
			if n_features == 5:
				tx_i_head_proj = tx_head_proj
				ty_i_head_proj = ty_head_proj
			if velocities:
				vx_i_proj = vx_proj
				vy_i_proj = vy_proj
			continue

		elif p == j:
			x_j_name = name
			x_j_proj = x_proj
			y_j_proj = y_proj
			if n_features > 3:
				tx_j_proj = tx_proj
				ty_j_proj = ty_proj
			if n_features == 5:
				tx_j_head_proj = tx_head_proj
				ty_j_head_proj = ty_head_proj
			if velocities:
				vx_j_proj = vx_proj
				vy_j_proj = vy_proj
			continue


		new_frame.append(x_proj)
		new_frame.append(y_proj)
		if not velocities:
			if n_features > 3:
				new_frame.append(tx_proj)
				new_frame.append(ty_proj)
			if n_features == 5:
				new_frame.append(tx_head_proj)
				new_frame.append(ty_head_proj)
		elif velocities:
			new_frame.append(vx_proj)
			new_frame.append(vy_proj)

	# repeat individual coordinates for last 2 rows
	new_frame.append(x_i_proj)
	new_frame.append(y_i_proj)
	if not velocities:
		if n_features > 3:
			new_frame.append(tx_i_proj)
			new_frame.append(ty_i_proj)
		if n_features == 5:
			new_frame.append(tx_i_head_proj)
			new_frame.append(ty_i_head_proj)
	if velocities:
		new_frame.append(vx_i_proj)
		new_frame.append(vy_i_proj)

	new_frame.append(x_j_proj)
	new_frame.append(y_j_proj)
	if not velocities:
		if n_features > 3:
			new_frame.append(tx_j_proj)
			new_frame.append(ty_j_proj)
		if n_features == 5:
			new_frame.append(tx_j_head_proj)
			new_frame.append(ty_j_head_proj)
	if velocities:
		new_frame.append(vx_j_proj)
		new_frame.append(vy_j_proj)

	return new_frame




# loops through a frame of features and flips them over the horizontal axis (vertically)
# ammounts to leaving the x coordinates and flipping the y
# assumes data is in the form x coordinate, y coordinate, x coordinate, y coordinate, etc
#     eg (x,y, cos(theta), sin(theta), x velocity, y velocity)
# n_features is the number of features in the adjusted (expanded) shifted coordinate
def augment_frame_nonstandard(frame, n_features):
	time = frame[0]
	people = frame[1:]
	time_h = time[:-3] + "001"

	frame_h = [time_h]

	for i in range(int(len(people)/n_features)):
		for j in range(0,n_features):
			feature = float(people[i*n_features+j])
			if j%2 == 0:
				frame_h.append(feature)
			else:
				frame_h.append(-1*feature)

	return frame_h

