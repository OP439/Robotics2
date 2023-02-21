import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.morphology import binary_dilation
import rospy
import sys

rect_list = np.array([[0, -9, 2.5, 2],
                    [8.75, -9.25, 2.75, 1.25],
                    [4.5, -5.5, 4, 2.75],
                    [-0.25, -1.0, 3.75, 2.0],
                    [-4.5, -5.5, 4, 2.75],
                    [-8.75, -9.25, 2.75, 1.25],
                    [-9.5, 1.5, 1.25, 2.75],
                    [-5, 1, 1.25, 3.5],
                    [-0.25, 1, 2.75, 3.5],
                    [5, 1, 1.25, 3.5],
                    [9.25, 1.5, 1.25, 2.75],
                    [5.5, 6.0, 3.75, 1.75],
                    [0.0, 6.0, 1.75, 3.0],
                    [-5.5, 6, 3.75, 1.75],
                    [-10, 0, 0.1, 20],
                    [10, 0, 0.1, 20],
                    [0, -10, 20, 0.1],
                    [0, 10, 20, 0.1]], dtype=float)

xmin = -10.1
xmax = 10.1
ymin = -10.1
ymax = 10.1

DENIRO_width = 1

scale = 16

def generate_map():
    # Number of pixels in each direction
    N_x = int((xmax - xmin) * scale)
    N_y = int((ymax - ymin) * scale)
    
    # Initialise the map to be an empty 2D array
    img = np.zeros([N_x, N_y], dtype = np.float)

    for x1, y1, w, h in rect_list:
        x0 = int((x1 - w/2 - xmin) * scale)
        y0 = int((y1 - h/2 - ymin) * scale)
        x3 = int((x1 + w/2 - xmin) * scale)
        y3 = int((y1 + h/2 - ymin) * scale)
        
        # Fill the obstacles with 1s
        img[y0:y3, x0:x3] = 1
    return img, scale, scale
    

def expand_map(img, robot_width):
    # We are passed robot_width as DENIRO_width which is set to 1m
    # 1m is 16 px wide so we have to convert from metres to pixels
    # This means that one pixel is 0.0625m wide
    # We are creating a pixel mask for deniro
    # The expanding of the map based on this mask is a technique called dilation
    robot_px = int(robot_width * scale)   # size of the robot in pixels x axis, returns 16 here
    
    ############################################################### TASK A
    # SQUARE MASK
    # create a square array of ones of the size of the robot in pixels
    robot_mask = np.ones((robot_px, robot_px))
    #Below is the matrix that this returns. 
#     [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
    
    # CIRCULAR MASK - optional for individual students
    # create a square array of the size of the robot
    # where a circle the size of the robot is filled with ones
    
    ##### NEW VERSION - filled in in the middle
    # radius of the robot is 8 pixels (16/2)
    radius=robot_px/2
    
    # Create an array of ints from -8 to 8 and square them
    # Needed for comparison in the next step
    A = np.arange(-radius,radius+1)**2
    # [64 49 36 25 16  9  4  1  0  1  4  9 16 25 36 49 64]
    
    # A[:,None] turns the above array into a column vector instead, is equivalent to doing .reshape(-1,1) here
    # A[:,None] + A is then a 16x16 matrix where we add every element of the row vector to a row of the column vector
    # The above step returns below where top left element is 64+64 and below that is 49+64 and so on
#     [[128 113 100  89  80  73  68  65  64  65  68  73  80  89 100 113 128]
#      [113  98  85  74  65  58  53  50  49  50  53  58  65  74  85  98 113]
#      [100  85  72  61  52  45  40  37  36  37  40  45  52  61  72  85 100]
#      [ 89  74  61  50  41  34  29  26  25  26  29  34  41  50  61  74  89]
#      [ 80  65  52  41  32  25  20  17  16  17  20  25  32  41  52  65  80]
#      [ 73  58  45  34  25  18  13  10   9  10  13  18  25  34  45  58  73]
#      [ 68  53  40  29  20  13   8   5   4   5   8  13  20  29  40  53  68]
#      [ 65  50  37  26  17  10   5   2   1   2   5  10  17  26  37  50  65]
#      [ 64  49  36  25  16   9   4   1   0   1   4   9  16  25  36  49  64]
#      [ 65  50  37  26  17  10   5   2   1   2   5  10  17  26  37  50  65]
#      [ 68  53  40  29  20  13   8   5   4   5   8  13  20  29  40  53  68]
#      [ 73  58  45  34  25  18  13  10   9  10  13  18  25  34  45  58  73]
#      [ 80  65  52  41  32  25  20  17  16  17  20  25  32  41  52  65  80]
#      [ 89  74  61  50  41  34  29  26  25  26  29  34  41  50  61  74  89]
#      [100  85  72  61  52  45  40  37  36  37  40  45  52  61  72  85 100]
#      [113  98  85  74  65  58  53  50  49  50  53  58  65  74  85  98 113]
#      [128 113 100  89  80  73  68  65  64  65  68  73  80  89 100 113 128]]
    # We then take the square root of each of the values in the above array
    dists = np.sqrt(A[:,None] + A)
    
    # We then do a comparison of the results from the previous step with our maximum distance which is 8
    # This is a boolean comparison so we return 1 if our condition is true and 0 if the condition is false
    # We add 0.5 to the radius because deniro is not a straight line so needs some turning space
    robot_mask = (np.abs(dists<=radius+0.5)).astype(int)
#     [[0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0]
#      [0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0]
#      [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
#      [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
#      [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
#      [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
#      [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
#      [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
#      [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
#      [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
#      [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
#      [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
#      [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
#      [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
#      [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
#      [0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0]
#      [0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0]]
    
    # this is where we do the dilation - we imported this function
    expanded_img = binary_dilation(img, robot_mask)
    
    return expanded_img
                        
            
    
    

def main(task):
    if task == 'view':
        print("============================================================")
        print("Generating the map")
        print("------------------------------------------------------------")
        img, xscale, yscale = generate_map()
        plt.imshow(img, vmin=0, vmax=1, origin='lower')
        plt.show()
    
    elif task == 'expand':
        print("============================================================")
        print("Generating the C-space map")
        print("------------------------------------------------------------")
        img, xscale, yscale = generate_map()
        c_img = expand_map(img, DENIRO_width)
        plt.imshow(c_img, vmin=0, vmax=1, origin='lower')
        plt.show()
        
    

if __name__ == "__main__":
    tasks = ['view', 'expand']
    if len(sys.argv) <= 1:
        print('Please include a task to run from the following options:\n', tasks)
    else:
        task = str(sys.argv[1])
        if task in tasks:
            print("Running Coursework 2 -", task)
            main(task)
        else:
            print('Please include a task to run from the following options:\n', tasks)



