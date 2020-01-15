import numpy as np

groups = np.genfromtxt('queue_groups.txt', dtype = 'str')
print(np.arange(4001,4600))
np.savetxt('queue_groups_2.txt', features[:,0:31], fmt='%5s')