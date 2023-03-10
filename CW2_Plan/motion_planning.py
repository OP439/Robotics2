# PLEASE NOTE YOU MAY GET INDENT ERRORS DEPENDING ON WHAT PLUGINS YOU USE
# TO COMMENT AND UNCOMMENT CODE BLOCKS

from math import cos, sin, acos, asin, atan2, sqrt, pi
from numpy import genfromtxt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from map import generate_map, expand_map, DENIRO_width
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rospy
import sys
import atexit

deniro_position = np.array([0, -6.0])
deniro_heading = 0.0
deniro_linear_vel = 0.0
deniro_angular_vel = 0.0

map = generate_map()

initial_position = np.array([0.0, -6.0])
goal = np.array([8.0, 8.0])


def deniro_odom_callback(msg):
    '''This function extracts important information about the position, orientation, and velocity of robot
    deniro from an incoming ROS message of type nav_msgs/Odometry and stores it in global variables that can be 
    accessed and modified from other parts of the code.'''
    global deniro_position, deniro_heading, deniro_linear_vel, deniro_angular_vel
    deniro_position = np.array(
        [msg.pose.pose.position.x, msg.pose.pose.position.y])
    r = R.from_quat([msg.pose.pose.orientation.x,
                     msg.pose.pose.orientation.y,
                     msg.pose.pose.orientation.z,
                     msg.pose.pose.orientation.w])
    deniro_heading = r.as_euler('xyz')[2]
    deniro_linear_vel = np.sqrt(
        msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2)
    deniro_angular_vel = msg.twist.twist.angular.z


def set_vref_publisher():
    '''Sets up a ROS node, creates a publisher and subscriber object, and returns the publisher object so that it can be
    used to send velocity commands to a robot. The function also includes a time delay to ensure that the node is 
    properly initialized before it begins executing other code. '''
    rospy.init_node("motion_planning_node")

    wait = True
    while (wait):
        now = rospy.Time.now()
        if now.to_sec() > 0:
            wait = False

    vref_topic_name = "/robot/diff_drive/command"
    # rostopic pub /robot/diff_drive/command geometry_msgs/Twist -r 10 -- '[0.0, 0.0, 0.0]' '[0.0, 0.0, -0.5]'
    pub = rospy.Publisher(vref_topic_name, Twist, queue_size=1000)

    odom_topic_name = "odom"
    sub = rospy.Subscriber(odom_topic_name, Odometry, deniro_odom_callback)
    return pub


def cmd_vel_2_twist(v_forward, omega):
    twist_msg = Twist()
    twist_msg.linear.x = v_forward
    twist_msg.linear.y = 0
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = omega
    return twist_msg


class MotionPlanner():

    def __init__(self, map, scale, goal):
        self.vref_publisher = set_vref_publisher()
        self.pixel_map = map
        self.xscale, self.yscale = scale
        self.goal = goal

        #initialise array for graph plotting
        self.posArray = []
        #plot the position array just before exiting the programme with ctrl c
        atexit.register(self.draw_path)

    def send_velocity(self, vref):
        # vref is given in cartesian coordinates (v_x, v_y)
        # DE NIRO is driven in linear and angular coordinates (v_forward, omega)
        # print("px:\t", deniro_position[0], ",\tpy:\t", deniro_position[1])
        # print("gx:\t", goal[0], ",\tgy:\t", goal[1])
        print("vx:\t", vref[0], ",\tvy:\t", vref[1])
        v_heading = atan2(vref[1], vref[0])
        heading_error = deniro_heading - v_heading
        omega = 1 * heading_error
        # only drive forward if DE NIRO is pointing in the right direction
        if abs(heading_error) < 0.1:
            v_forward = min(max(sqrt(vref[0]**2 + vref[1]**2), 0.1), 0.2)
        else:
            v_forward = 0

        twist_msg = cmd_vel_2_twist(v_forward, omega)
        print("v_fwd:\t", v_forward, ",\tw:\t", omega)

        self.vref_publisher.publish(twist_msg)

    def map_position(self, world_position):
        world_position = world_position.reshape((-1, 2))
        map_x = np.rint(world_position[:, 0] *
                        self.xscale + self.pixel_map.shape[0] / 2)
        map_y = np.rint(world_position[:, 1] *
                        self.yscale + self.pixel_map.shape[1] / 2)
        map_position = np.vstack((map_x, map_y)).T
        return map_position

    def world_position(self, map_position):
        map_position = map_position.reshape((-1, 2))
        world_x = (map_position[:, 0] -
                   self.pixel_map.shape[0] / 2) / self.xscale
        world_y = (map_position[:, 1] -
                   self.pixel_map.shape[1] / 2) / self.yscale
        world_position = np.vstack((world_x, world_y)).T
        return world_position

    def run_planner(self, planning_algorithm):
        '''Runs the input planning algorithm and sends velocity commands to the robot '''
        rate = rospy.Rate(25)
        while not rospy.is_shutdown():
            vref, complete = planning_algorithm()
            self.send_velocity(vref)
            if complete:
                print("Completed motion")
                break
            rate.sleep()

    def setup_waypoints(self):
        # TASK B i - Choosing Waypoints

        # Refer to section 3.1.1 of the report for more information

        # The initial position of the robot is given by the global variable deniro_position
        # The goal position is given by the global variable goal
        # Waypoints between the initial and goal positions can be added to the array waypoints

        # Create an array of waypoints for the robot to navigate via to reach the goal
        # fill this in with your waypoints
        waypoints = np.array([initial_position,
                              [3, -2],
                              [2.5, 8],
                              goal])
        # convert waypoints to pixel coordinates
        waypoints = np.vstack([initial_position, waypoints, self.goal])
        pixel_goal = self.map_position(self.goal)
        pixel_waypoints = self.map_position(waypoints)

        print('Waypoints:\n', waypoints)
        print('Waypoints in pixel coordinates:\n', pixel_waypoints)

        # TASK B ii - Calculating the Path Length

        # Refer to section 3.1.2 of the report for more information

        # The distance between each waypoint is calculated and the total path length is calculated
        # The total path length is printed to the terminal

        # calculate the total path length of the waypoints
        path_length = 0
        # loop through the waypoints and calculate the distance between each waypoint
        for i in range(1, waypoints.shape[0]):
            # add the distance between the current waypoint and the previous waypoint to the
            # total path length
            path_length += np.linalg.norm(waypoints[i, :] -
                                          waypoints[i - 1, :])
        # round path length to 2 decimal places
        path_length = round(path_length, 2)
        print('Total path length:\t', path_length)

        # Plotting
        # Plot the expanded map with the waypoints
        plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')
        plt.scatter(pixel_waypoints[:, 0], pixel_waypoints[:, 1])
        plt.plot(pixel_waypoints[:, 0], pixel_waypoints[:, 1])
        plt.show()

        self.waypoints = waypoints
        self.waypoint_index = 0

    def waypoint_navigation(self):
        complete = False

        # get the current waypoint
        current_waypoint = self.waypoints[self.waypoint_index, :]
        # calculate the vector from DE NIRO to waypoint
        waypoint_vector = current_waypoint - deniro_position
        # calculate the distance from DE NIRO to waypoint
        distance_to_waypoint = np.linalg.norm(waypoint_vector)
        # calculate the unit direction vector from DE NIRO to waypoint
        waypoint_direction = waypoint_vector / distance_to_waypoint

        # Calculate a reference velocity based on the direction of the waypoint
        vref = waypoint_direction * 0.5

        # If we have reached the waypoint, start moving to the next waypoint
        if distance_to_waypoint < 0.05:
            self.waypoint_index += 1    # increase waypoint index

        # If we have reached the last waypoint, stop
        if self.waypoint_index > self.waypoints.shape[0]:
            vref = np.array([0, 0])
            complete = True
        return vref, complete

    def potential_field(self):
        # COMMENT OUT WHICHEVER PART YOU WANT TO IGNORE
        # TASK C Part i below
        # See report section 4.1 for for this information
#         complete = False

#         # compute the positive force attracting the robot towards the goal
#         # vector to goal position from DE NIRO
#         goal_vector = goal - deniro_position
#         # distance to goal position from DE NIRO
#         distance_to_goal = np.linalg.norm(goal_vector)
#         # unit vector in direction of goal from DE NIRO
#         pos_force_direction = goal_vector / distance_to_goal

#         # potential function
#         pos_force_magnitude = 1  #Part i asks for the attraction to the goal to be constant hence value of 1
#         # tuning parameter - done manually
#         K_att = 1 # I achieved best results with 1 here     # tune this parameter to achieve    desired results

#         # positive force, pos_force_direction is unit vector in direction of goal from DE NIRO
#         ## The mathematical representation of this force can be seen in report section 4.1
#         ## This is the most basic possible representation
#         positive_force = K_att * pos_force_direction * pos_force_magnitude  # normalised positive force

#         # compute the negative force repelling the robot away from the obstacles
#         # if self.pixel_map == 1 that means there is an obstacle as with DENIRO inflation
#         # each one of these is an "i" component that is summed in report section 4.1
#         obstacle_pixel_locations = np.argwhere(self.pixel_map == 1)
#         # coordinates of every obstacle pixel, switching the columns around here
#         obstacle_pixel_coordinates = np.array([obstacle_pixel_locations[:, 1], obstacle_pixel_locations[:, 0]]).T
#         # coordinates of every obstacle pixel converted to world coordinates, convert pixel coords to world coords
#         # see report section 3.1.1 for more information on where the goal, obstacles and origin are and why they are converted.
#         obstacle_positions = self.world_position(obstacle_pixel_coordinates)

#         # vector to each obstacle pixel from DE NIRO, about 48000 obstacle pixel each with x and y coords
#         # (48000, 2) array, each row represents x and y component of vector from deniro to obstacle pixel
#         obstacle_vector = obstacle_positions - deniro_position   # vector from DE NIRO to obstacle

#         # distance to obstacle from DE NIRO, for each pixel, |x| = sqrt(x^2+y^2)
#         # .reshape(-1,1) returns array with one column and however many rows
#         distance_to_obstacle = np.linalg.norm(obstacle_vector, axis=1).reshape((-1, 1))  # magnitude of vector
#         # unit vector in direction of obstacle from DE NIRO, for each pixel
#         force_direction = obstacle_vector / distance_to_obstacle   # normalised vector (for direction)

#         # potential function, distance_to_obstacle is again to each obstacle pixel
#         force_magnitude = -1/distance_to_obstacle #we are asked to give equation 2 from the doc so inversely proportional
#         # tuning parameter - manual process, see report section 4.1 for the effect of different values.
#         K_rep = 15.5  # I achieved best results with this value here

#         # force from each individual obstacle pixel
#         obstacle_force = force_direction * force_magnitude
#         # total negative force on DE NIRO, summing the effect of each obstacle pixel on deniro
#         # np.sum term corresponds to the sum in related Equation
#         # .shape[0] returns the number of rows in the array which corresponds to N in section 4.1 of the report(here number of obstacle pixels)
#         negative_force = K_rep * np.sum(obstacle_force, axis=0) / obstacle_pixel_locations.shape[0]
# TASK C Part i above

# TASK C Part ii v1 below
#         complete = False

#         # compute the positive force attracting the robot towards the goal
#         # vector to goal position from DE NIRO
#         goal_vector = goal - deniro_position
#         # distance to goal position from DE NIRO
#         distance_to_goal = np.linalg.norm(goal_vector)
#         # unit vector in direction of goal from DE NIRO
#         pos_force_direction = goal_vector / distance_to_goal

#         # potential function
#         # this is what has changed since section 4.1 of the report, we are now relating the distance to the goal to the force.
#         pos_force_magnitude = 1/distance_to_goal     # your code here!
#         # tuning parameter - below value was found to reach goal, see report section 4.2 for effect of different values
#         K_att = 100050     # tune this parameter to achieve desired results

#         # positive force
#         positive_force = K_att * pos_force_direction * pos_force_magnitude  # normalised positive force

#         # compute the negative force repelling the robot away from the obstacles
#         obstacle_pixel_locations = np.argwhere(self.pixel_map == 1)
#         # coordinates of every obstacle pixel
#         obstacle_pixel_coordinates = np.array([obstacle_pixel_locations[:, 1], obstacle_pixel_locations[:, 0]]).T
#         # coordinates of every obstacle pixel converted to world coordinates
#         obstacle_positions = self.world_position(obstacle_pixel_coordinates)

#         # vector to each obstacle from DE NIRO
#         # report section 4.1 has a nice graphical representation of this process
#         # it is more complex than it looks here as the subtraction is happening for every single pixel
#         # this is numpy making the code look simpler
#         obstacle_vector = obstacle_positions - deniro_position   # vector from DE NIRO to obstacle

#         # distance to obstacle from DE NIRO
#         distance_to_obstacle = np.linalg.norm(obstacle_vector, axis=1).reshape((-1, 1))  # magnitude of vector
#         # unit vector in direction of obstacle from DE NIRO
#         force_direction = obstacle_vector / distance_to_obstacle   # normalised vector (for direction)

#         # potential function
#         # the high exponent here means that the force will be small until the robot is less than 1m from the obstacle
#         # see report section 4.2 for information about the physical real world meaning of this and to see graphs of the effect
#         force_magnitude = -1/distance_to_obstacle**4   # your code here!
#         # tuning parameter - reaches goal with this value. See report section 4.2 for effect of different values
#         K_rep = 140000     # tune this parameter to achieve desired results

#         # force from an individual obstacle pixel
#         obstacle_force = force_direction * force_magnitude
#         # total negative force on DE NIRO
#         # this end equation is the same as in section 4.1 in the report
#         # summing all the forces and dividing by the number of obstacles (obstacle pixels here)
#         negative_force = K_rep * np.sum(obstacle_force, axis=0) / obstacle_pixel_locations.shape[0]
        # TASK C Part ii v1 above

        # TASK C Part ii v2 below
        # see V2 Flow Field Implementation in section 4.1.2 of the report for more details
        complete = False

        # POSITIVE FORCE 1 - only y force, does not depend on deniro distance to goal
        # Left-most image in Figure 18 of report section 4.1.2
        # compute the positive force attracting the robot towards the goal
        # vector to goal position from DE NIRO
        goal_vector = goal - deniro_position
        # distance to goal position from DE NIRO
        distance_to_goal = np.linalg.norm(goal_vector)
        # unit vector in direction of goal from DE NIRO
        pos_force_direction = goal_vector / distance_to_goal
        # keep only the y component of this force (set the x component of the force to 0)
        # this is done as we have a negative force pushing the robot in only the x direction so
        # we prefer to approach obstacles head on instead of from the side
        pos_force_direction[0] = 0

        # potential function
        # coonstant force towards goal as it is far away  # your code here!
        # ends up being quite a small force except where the robot is far away from the goal and obstacles
        pos_force_magnitude = 1
        # tuning parameter
        K_att = 5     # tune this parameter to achieve    desired results

        # positive force
        positive_force = K_att * pos_force_direction * \
            pos_force_magnitude  # normalised positive force

        # POSITIVE FORCE 2 - both for x and y, depends on deniro distance to goal
        # compute the positive force attracting the robot towards the goal
        # second to left image of Figure 18 in report section 4.1.2
        # vector to goal position from DE NIRO
        goal_vector = goal - deniro_position
        # distance to goal position from DE NIRO
        distance_to_goal = np.linalg.norm(goal_vector)
        # unit vector in direction of goal from DE NIRO
        pos_force_direction = goal_vector / distance_to_goal

        # potential function
        # relate this force to deniro distance from goal, make it have a wider effect by dividing distance by 2
        # for the linear case, this essentially doubles the force experienced
        # when a higher exponent is added to the distance, then this affects the range at which the force starts to grow rapidly
        # this works as when the distance is for example 1.5m from objective, this magnitude will count it as 0.75 which, when squared, 
        # and used as the denominator, will cause a large value for the total magnitude
        pos_force_magnitude = 1/(distance_to_goal/2)  # **2
        # tuning parameter
        K_att = 20     # tune this parameter to achieve    desired results

        # positive force, add it to the previous force
        positive_force += K_att * pos_force_direction * \
            pos_force_magnitude  # normalised positive force

        # MAP CALCULATIONS
        # compute the negative force repelling the robot away from the obstacles
        obstacle_pixel_locations = np.argwhere(self.pixel_map == 1)
        # coordinates of every obstacle pixel
        obstacle_pixel_coordinates = np.array(
            [obstacle_pixel_locations[:, 1], obstacle_pixel_locations[:, 0]]).T
        # coordinates of every obstacle pixel converted to world coordinates
        obstacle_positions = self.world_position(obstacle_pixel_coordinates)

        # NEGATIVE FORCE 1 - obstacles repel deniro at very close range
        # second from right image in Figure 18 of report section 4.1.2
        # vector to each obstacle from DE NIRO
        obstacle_vector = obstacle_positions - \
            deniro_position   # vector from DE NIRO to obstacle

        # distance to obstacle from DE NIRO
        distance_to_obstacle = np.linalg.norm(
            obstacle_vector, axis=1).reshape((-1, 1))  # magnitude of vector
        # unit vector in direction of obstacle from DE NIRO
        # normalised vector (for direction)
        force_direction = obstacle_vector / distance_to_obstacle

        # potential function
        # force is huge when deniro is very close to obstacle but decays rapidly as he is further
        # multiplying distance_to_obstacles by 1.5 to require a closer distance for the exponent to kick in
        # if the distance is 0.8m, what is used to the power of 8 is 0.8*1.5 which is 1.2 which raised to the power of eight 
        # will be a far larger number. having -1 over that number will result in a smaller number and therefore smaller force
        # this is done as we want NEGATIVE FORCE 2 to act before this one as NEGFOR2 points the robot towards the goal and this force
        # simply repels the robot
        force_magnitude = -1/(distance_to_obstacle*1.5)**8   # your code here!
        # tuning parameter
        K_rep = 15.5     # tune this parameter to achieve desired results

        # force from an individual obstacle pixel
        obstacle_force = force_direction * force_magnitude
        # total negative force on DE NIRO
        negative_force = K_rep * \
            np.sum(obstacle_force, axis=0) / obstacle_pixel_locations.shape[0]

        # NEGATIVE FORCE 2 - objects move deniro towards x coordinate of goal when he is perpendicular to the force
        # middle image of Figure 18 in report section 4.1.2
        # this is the force where both a vector from obstacles to goal and from obstacles to deniro is used
        # vector from each obstacle pixel to goal
        goal_vector = obstacle_positions - goal
        # distance from each obstacle pixel to goal
        distance_to_goal = np.linalg.norm(goal_vector)
        # unit vector in direction of goal from each obstacle pixel
        pos_force_direction = goal_vector / distance_to_goal
        # set the y components of the force direction from obstacle pixels to goal to 0 (force will only move deniro towards x value of goal)
        pos_force_direction[:, 1] = 0

        # vector from each obstacle pixel to deniro
        obstacle_vector = obstacle_positions - deniro_position

        # distance to obstacle from DE NIRO
        distance_to_obstacle = np.linalg.norm(
            obstacle_vector, axis=1).reshape((-1, 1))  # magnitude of vector
        # unit vector in direction of obstacle from DE NIRO
        # normalised vector (for direction)
        force_direction = obstacle_vector / distance_to_obstacle

        # potential function
        deniro_obstacle_vector = obstacle_positions - deniro_position
        distance_to_obstacle_deniro = np.linalg.norm(
            deniro_obstacle_vector, axis=1).reshape((-1, 1))  # magnitude of vector

        # take x component of the force between deniro and each obstacle pixel (essentially dot product of each row with [1,0])
        # both are unit vectors so no need to divide by the size of the vector
        force_direction_x = np.matmul(force_direction, np.array([1, 0]))
        
        # this next section is quite computationally expensive as we are doing many comparisons which adds to the complexity of the algo
        # in reality, there would not be as many comparisons as the robot would only get vectors for obstacles it could sense with a distance
        # sensor instead of getting vectors to each pixel of an obstacle as is the case here
        # cos(theta) = force_direction_x[index]  cos(85deg)=0.1 cos(95deg)=-0.1
        anglegt85 = (force_direction_x < 0.1).astype(int)
        # 1 for true, 0 for false but as integers
        anglelt95 = (force_direction_x > -0.1).astype(int)
        # multiply both together to see which values satisfy both conditions (numpy needed it done this way)
        # essentially an and condition, if both conditions are True, it will be 1*1=1, if only one condition is True,
        # it will be True*False=False or 1*0=0
        angles_between = anglegt85*anglelt95

        # again make sure that the force only takes effect when deniro is close
        # force acts earlier than NEGATIVE FORCE 1 as we are only multiplying distance to obstacles by 1 instead of 1.5
        # taking the same example from earlier of being 0.8m from the obstacle, it will now be 0.8^8 which will be a very small number
        # which when used in the denominator will cause a very large force 
        # distance at which it starts acting can be tuned
        force_magnitude = -1 / \
            (distance_to_obstacle_deniro*1)**8   # your code here!
        # tuning parameter
        K_rep = 20000     # tune this parameter to achieve desired results

        # multiply each row of the pos_force_direction (force towards x value of goal) by vector of scalars of 0 or 1
        # ignores forces towards the x value of the goal that are caused by obstacle pixels where
        # deniro is not below the obstacle in the y coordinate
        # this is a lot easier to see visually in the middle image of Figure 18 in report section 4.1.2
        obstacle_force = pos_force_direction * \
            angles_between[:, np.newaxis] * \
            force_magnitude  # *keep_or_remove[:,np.newaxis]

        # add these negative forces together with +=
        negative_force += K_rep * \
            np.sum(obstacle_force, axis=0) / obstacle_pixel_locations.shape[0]

        # TASK C Part ii v2 above

        print("deniro_position:", deniro_position)
        self.posArray.append(deniro_position)

        # Uncomment these lines to visualise the repulsive force from each obstacle pixel
        # Make sure to comment it out again when you run the motion planner fully
        # plotskip = 100   # only plots every 10 pixels (looks cleaner on the plot)
        # plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')
        # plt.quiver(obstacle_pixel_coordinates[::plotskip, 0], obstacle_pixel_coordinates[::plotskip, 1], obstacle_force[::plotskip, 0] * self.xscale, obstacle_force[::plotskip, 1] * self.yscale)
        # plt.show()

        print("positive_force:", positive_force)
        print("negative_force:", negative_force)

        # Reference velocity is the resultant force
        vref = positive_force + negative_force

        # If the goal has been reached, stop
        print("distance_to_goal:", distance_to_goal)
        if distance_to_goal < 0.20:
            vref = np.array([0, 0])
            complete = True
            self.draw_path()
        return vref, complete

    # function that draws the path that the robot takes using posArray
    def draw_path(self):
        # get the position coordinates from the posArray
        posArray = np.array(self.posArray)
        # convert the position coordinates to map pixel coordinates
        pixel_posArray = self.map_position(posArray)

        # calculate the distance travelled by the robot in metres
        distance = 0
        for i in range(1, len(posArray)):
            distance += np.linalg.norm(posArray[i] - posArray[i-1])
        print("distance travelled:", distance, "m")

        # plot the path
        plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')
        plt.scatter(pixel_posArray[:, 0], pixel_posArray[:, 1])
        plt.show()

    def generate_random_points(self, N_points):
        # TASK D
        N_accepted = 0  # number of accepted samples
        # empty array to store accepted samples
        accepted_points = np.empty((1, 2))
        # empty array to store rejected samples
        rejected_points = np.empty((1, 2))

        while N_accepted < N_points:    # keep generating points until N_points have been accepted

            # generate random coordinates using a uniform distribution between -10 and 10
            points = np.random.uniform(-10, 10, (N_points - N_accepted, 2))
            # generate random coordinates using a normal distribution between -10 and 10 witha  mean of
            # and std of 3 meaing that there is a less than 0.1% chance of a value being generated
            # outside of our range 68% should lie between 6 and -6
            # points = np.random.normal(0, 3, (N_points - N_accepted, 2))
            # get the point locations on our map
            pixel_points = self.map_position(points)
            # create an empty array of rejected flags
            rejected = np.zeros(N_points - N_accepted)

            # Loop through the generated points and check if their pixel location corresponds to an obstacle in self.pixel_map
            for i in range(N_points - N_accepted):
                rejected[i] = self.pixel_map[int(
                    pixel_points[i, 1]), int(pixel_points[i, 0])]

            new_accepted_points = pixel_points[np.argwhere(
                rejected == 0)].reshape((-1, 2))
            new_rejected_points = pixel_points[np.argwhere(
                rejected == 1)].reshape((-1, 2))
            # keep an array of generated points that are accepted
            accepted_points = np.vstack((accepted_points, new_accepted_points))
            # keep an array of generated points that are rejected (for visualisation)
            rejected_points = np.vstack((rejected_points, new_rejected_points))
            # update the number of accepted points
            N_accepted = accepted_points.shape[0] - 1

        # throw away that first 'empty' point we added for initialisation
        accepted_points = accepted_points[1:, :]
        rejected_points = rejected_points[1:, :]

        # visualise the accepted and rejected points
        plt.imshow(self.pixel_map, vmin=0, vmax=1,
                   origin='lower')  # setup a plot of the map

        # plot accepted points in blue
        plt.scatter(accepted_points[:, 0], accepted_points[:, 1], c='b')
        # plot rejected points in red
        plt.scatter(rejected_points[:, 0], rejected_points[:, 1], c='r')

        deniro_pixel = self.map_position(initial_position)
        goal_pixel = self.map_position(goal)
        # plot DE NIRO as a white point
        plt.scatter(deniro_pixel[0, 0], deniro_pixel[0, 1], c='w')
        # plot the goal as a green point
        plt.scatter(goal_pixel[0, 0], goal_pixel[0, 1], c='g')

        plt.show()

        # calculate the position of the accepted points in world coordinates
        world_points = self.world_position(accepted_points)
        # add DE NIRO's position to the beginning of these points, and the goal to the end
        world_points = np.vstack((initial_position, world_points, goal))

        return world_points

    def create_graph(self, points):
        # section 5.2.1
        # tempory min max values selected from section 5.2.1
        # mindist = 2
        # maxdist = 5
        # section 5.3.1
        # final min max values selected from section 5.3.1
        mindist = 1.75
        maxdist = 5.5

        # Calculate a distance matrix between every node to every other node
        distances = cdist(points, points)

        # Create two dictionaries
        graph = {}  # dictionary of each node, and the nodes it connects to
        # dictionary of each node, and the distance to each node it connects to
        distances_graph = {}

        plt.imshow(self.pixel_map, vmin=0, vmax=1,
                   origin='lower')  # setup a plot of the map

        for i in range(points.shape[0]):    # loop through each node
            # get nodes an acceptable distance of the current node
            points_in_range = points[(distances[i] >= mindist) & (
                distances[i] <= maxdist)]
            # get the corresponding distances to each of these nodes
            distances_in_range = distances[i, (distances[i] >= mindist) & (
                distances[i] <= maxdist)]

            # if there are any nodes in an acceptable range
            if points_in_range.shape[0] > 0:

                # set up arrays of nodes with edges that don't collide with obstacles, and their corresponding distances
                collision_free_points = np.empty((1, 2))
                collision_free_distances = np.empty((1, 1))

                # loop through the nodes an acceptable distance of the current node
                for j in range(points_in_range.shape[0]):

                    # get the current node position on the map
                    pxA = self.map_position(points[i])
                    # get the node in range position on the map
                    pxB = self.map_position(points_in_range[j])

                    # check if there is a collision on the edge between two points
                    collision = self.check_collisions(
                        points[i], points_in_range[j])

                    if collision:
                        # if there is a collision, plot the edge in red
                        plt.plot([pxA[0, 0], pxB[0, 0]], [
                                 pxA[0, 1], pxB[0, 1]], c='r')
                        pass
                    else:
                        # if there is no collision, add the node in range to the array of nodes that have no collisions
                        collision_free_points = np.append(
                            collision_free_points, points_in_range[j].reshape((1, 2)), axis=0)
                        # add the corresponding distance to the array of distances
                        collision_free_distances = np.append(
                            collision_free_distances, distances_in_range[j].reshape((1, 1)))
                        # plot the edge in blue
                        plt.plot([pxA[0, 0], pxB[0, 0]], [
                                 pxA[0, 1], pxB[0, 1]], c='b')

                # after we've looped through every point, update the two dictionaries
                graph[str(points[i])] = collision_free_points[1:]
                distances_graph[str(points[i])] = collision_free_distances[1:]

        # Plotting
        deniro_pixel = self.map_position(initial_position)
        goal_pixel = self.map_position(goal)

        plt.scatter(deniro_pixel[0, 0], deniro_pixel[0, 1], c='w')
        plt.scatter(goal_pixel[0, 0], goal_pixel[0, 1], c='g')

        plt.show()

        return graph, distances_graph

    def check_collisions(self, pointA, pointB):
        # Section 5.2.2
        # Calculate the distance between the two points
        # using code based of the equaiton 3
        # pointA[0] = Ax, pointA[1] = Ay pointB[0] = Bx pointB[1] = By
        distance = ((pointA[0] - pointB[0])**2 +
                    (pointA[1] - pointB[1])**2)**0.5
        # Calculate the UNIT direction vector pointing from pointA to pointB7
        # using code based of the equaiton 4 and figure 25
        direction = np.array(
            [-(pointA[0] - pointB[0])/distance, -(pointA[1] - pointB[1])/distance])
        # resolution set to less than half the width of a pixel
        # for reasons described under resolution in section 5.2.2
        resolution = 0.03

        # Create an array of points to check collisions at
        edge_points = pointA.reshape(
            (1, 2)) + np.arange(0, distance, resolution).reshape((-1, 1)) * direction.reshape((1, 2))
        # Convert the points to pixels
        edge_pixels = self.map_position(edge_points)

        for pixel in edge_pixels:   # loop through each pixel between pointA and pointB
            # if the pixel collides with an obstacle, the value of the pixel map is 1
            collision = self.pixel_map[int(pixel[1]), int(pixel[0])]

            if collision == 1:
                return True     # if there's a collision, immediately return True
        return False    # if it's got through every pixel as hasn't returned yet, return False

    def dijkstra(self, graph, edges):
        goal_node = goal
        nodes = list(graph.keys())

        # Create a dataframe of unvisited nodes
        # Section 5.3.1
        # Initialise each cost to a very high number becuase of reasons
        # explained in 5.3.1
        initial_cost = 1000000000000000000000000.0

        unvisited = pd.DataFrame({'Node': nodes, 'Cost': [
                                 initial_cost for node in nodes], 'Previous': ['' for node in nodes]})
        unvisited.set_index('Node', inplace=True)
        # Set the first node's cost to zero
        unvisited.loc[[str(initial_position)], ['Cost']] = 0.0

        # Create a dataframe of visited nodes (it's empty to begin with)
        visited = pd.DataFrame({'Node': [''], 'Cost': [0.0], 'Previous': ['']})
        visited.set_index('Node', inplace=True)

        # Take a look at the initial dataframes
        print('--------------------------------')
        print('Unvisited nodes')
        print(unvisited.head())
        print('--------------------------------')
        print('Visited nodes')
        print(visited.head())
        print('--------------------------------')
        print('Running Dijkstra')

        # Dijkstra's algorithm!
        # Keep running until we get to the goal node
        while str(goal_node) not in visited.index.values:

            # Go to the node that is the minimum distance from the starting node
            current_node = unvisited[unvisited['Cost']
                                     == unvisited['Cost'].min()]
            print(current_node)
            # the node's name (string)
            current_node_name = current_node.index.values[0]
            # the distance from the starting node to this node (float)
            current_cost = current_node['Cost'].values[0]
            # a list of the nodes visited on the way to this one (string)
            current_tree = current_node['Previous'].values[0]

            # get all of the connected nodes to the current node (array)
            connected_nodes = graph[current_node.index.values[0]]
            # get the distance from each connected node to the current node
            connected_edges = edges[current_node.index.values[0]]

            # Loop through all of the nodes connected to the current node
            for next_node_name, edge_cost in zip(connected_nodes, connected_edges):
                # the next node's name (string)
                next_node_name = str(next_node_name)

                if next_node_name not in visited.index.values:  # if we haven't visited this node before
                    # Section 5.3.1
                    # calculates the cost of going from the initial node to the next node via the current node
                    next_cost_trial = current_cost + edge_cost
                    # the previous best cost we've seen going to the next node
                    next_cost = unvisited.loc[[
                        next_node_name], ['Cost']].values[0]

                    # if it costs less to go the next node from the current node, update then next node's cost and the path to get there
                    if next_cost_trial < next_cost:
                        unvisited.loc[[next_node_name],
                                      ['Cost']] = next_cost_trial
                        # update the path to get to that node
                        unvisited.loc[[next_node_name], ['Previous']
                                      ] = current_tree + current_node_name

            # remove current node from the unvisited list
            unvisited.drop(current_node_name, axis=0, inplace=True)

            # add current node to the visited list
            visited.loc[current_node_name] = [current_cost, current_tree]

        print('--------------------------------')
        print('Unvisited nodes')
        print(unvisited.head())
        print('--------------------------------')
        print('Visited nodes')
        print(visited.head())
        print('--------------------------------')

        # Optimal cost (float)
        optimal_cost = visited.loc[[str(goal_node)], ['Cost']].values[0][0]
        # Optimal path (string)
        optimal_path = visited.loc[[str(goal_node)], ['Previous']].values[0][0]

        # Convert the optimal path from a string to an actual array of waypoints to travel to
        string_waypoints = optimal_path[1:-1].split('][')
        optimal_waypoints = np.array(
            [np.fromstring(waypoint, sep=' ') for waypoint in string_waypoints])
        # add the goal as the final waypoint
        optimal_waypoints = np.vstack((optimal_waypoints, goal))

        print('Results')
        print('Goal node: ', str(goal_node))
        print('Optimal cost: ', optimal_cost)
        print('Optimal path:\n', optimal_waypoints)
        print('--------------------------------')

        # Plotting
        optimal_pixels = self.map_position(optimal_waypoints)
        plt.plot(optimal_pixels[:, 0], optimal_pixels[:, 1], c='b')

        deniro_pixel = self.map_position(initial_position)
        goal_pixel = self.map_position(goal)

        plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')
        plt.scatter(deniro_pixel[0, 0], deniro_pixel[0, 1], c='w')
        plt.scatter(goal_pixel[0, 0], goal_pixel[0, 1], c='g')

        plt.show()

        # Setup the waypoints for normal waypoint navigation
        self.waypoints = optimal_waypoints
        self.waypoint_index = 0


def main(task):
    # load the map and expand it
    img, xscale, yscale = generate_map()
    c_img = expand_map(img, DENIRO_width)

    # load the motion planner
    planner = MotionPlanner(c_img, (xscale, yscale), goal=goal)

    if task == 'waypoints':
        print("============================================================")
        print("Running Waypoint Navigation")
        print("------------------------------------------------------------")
        planner.setup_waypoints()
        planner.run_planner(planner.waypoint_navigation)

    elif task == 'potential':
        print("============================================================")
        print("Running Potential Field Algorithm")
        print("------------------------------------------------------------")
        planner.run_planner(planner.potential_field)

    elif task == 'prm':
        print("============================================================")
        print("Running Probabilistic Road Map")
        print("------------------------------------------------------------")
        points = planner.generate_random_points(N_points=100)
        graph, edges = planner.create_graph(points)
        planner.dijkstra(graph, edges)
        planner.run_planner(planner.waypoint_navigation)


if __name__ == "__main__":
    tasks = ['waypoints', 'potential', 'prm']
    if len(sys.argv) <= 1:
        print('Please include a task to run from the following options:\n', tasks)
    else:
        task = str(sys.argv[1])
        if task in tasks:
            print("Running Coursework 2 -", task)
            main(task)
        else:
            print('Please include a task to run from the following options:\n', tasks)
