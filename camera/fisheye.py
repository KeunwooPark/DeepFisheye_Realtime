import numpy as np

#### fisheye functions ####
def stereographic_func(theta):
    return 2 * np.tan(theta / 2)

def equidistant_func(theta):
    return theta

def equisolid_func(theta):
    return 2 * np.sin(theta / 2)

def orthographic_func(theta):
    return np.sin(theta)

def stereographic_func_inv(y):
    return 2 * np.arctan2(y, 2)

def equidistant_func_inv(y):
    return y

def equisolid_func_inv(y):
    return 2 * np.arcsin(y / 2)

def orthographic_func_inv(y):
    return np.arcsin(y)
#############################

fisheye_func = {'stereographic': stereographic_func, \
                'equidistant': equidistant_func, \
                'equisolid': equisolid_func, \
                'orthographic': orthographic_func}

fisheye_func_inv = {'stereographic': stereographic_func_inv, \
                'equidistant': equidistant_func_inv, \
                'equisolid': equisolid_func_inv, \
                'orthographic': orthographic_func_inv}


def map_to_fisheye(joint_mat, fisheye_type, f, img_shape):
    center = (int(img_shape[0]/2), int(img_shape[1]/2))

    x = joint_mat[:,0]
    y = joint_mat[:,1]
    z = joint_mat[:,2]

    theta = np.arctan2(np.sqrt(x*x + y*y), z)
    phi = np.arctan2(y, x)
    r = f * fisheye_func[fisheye_type](theta)
    _x = r * np.cos(phi)
    _y =  r * np.sin(phi)
    fish_x = np.around(center[0] + _x).astype(int)
    fish_y = np.around(center[1] + _y).astype(int)

    fish_x = fish_x[:,np.newaxis]
    fish_y = fish_y[:,np.newaxis]
    fish_mat = np.hstack((fish_x, fish_y))
    return fish_mat
