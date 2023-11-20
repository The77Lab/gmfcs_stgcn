import numpy as np
import random

def Shear(input_data):
    limit = 0.1
    Shear = np.array([[1, random.uniform(-limit, limit)],
                     [random.uniform(-limit, limit), 1]])
    output = np.matmul(input_data[:,:,:,:2], Shear)
    output = np.concatenate((output, input_data[:,:,:,2:]), axis=3)
    return output


def Flip(input_data):
    order = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9,
             12, 11, 14, 13, 16, 15]
    output = input_data[:, :, order, :]
    return output

#coco neighbours information
neighbours = {}
neighbours[0] = [0, 1, 2, 5, 6]
neighbours[1] = [0, 1, 3]
neighbours[2] = [0, 2, 4]
neighbours[3] = [1, 3]
neighbours[4] = [2, 4]
neighbours[5] = [0, 5, 7, 11]
neighbours[6] = [0, 6, 8, 12]
neighbours[7] = [5, 7, 9]
neighbours[8] = [6, 8, 10]
neighbours[9] = [7, 9]
neighbours[10] = [8, 10]
neighbours[11] = [5, 11, 12, 13]
neighbours[12] = [6, 11, 12, 14]
neighbours[13] = [11, 13, 15]
neighbours[14] = [12, 14, 16]
neighbours[15] = [13, 15]
neighbours[16] = [14, 16]


def masking(input_data):
    output = input_data.copy()
    mask_joint = random.randint(0, 16)
    # remove the joint and its neighbors
    mask_list = neighbours[mask_joint]
    # remove the joint and its neighbors from all frames, use numpy array methods
    output[:, :, mask_list, :] = 0
    return output