import numpy as np
import PyKDL

import rospy

import baxter_interface

from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
from gazebo_msgs.srv import (SpawnModel, DeleteModel)
from geometry_msgs.msg import (PoseStamped, Pose, Point, Quaternion)
from math import pi, atan2,sin, cos,sqrt
import tf
import time
import os
import sys

import baxter_pykdl as b_pykdl
import matplotlib.pyplot as plt

rospy.init_node('baxter_kinematics')
rate = rospy.Rate(100)  # rate = 100 Hz

################################################
#### AUXILIARY FUNCTIONS########################
def DPinv(J, eps=1e-06):
    """ Pseudo-inverse of J """
    H = np.matmul(J,J.T)
    I = np.identity(H.shape[0])
    H = H + eps * I
    J_pinv = np.matmul(J.T, np.linalg.pinv(H))
    return J_pinv

#quat = np.array = qx,qy,qz,qw
def quat2angax(quat):
    """ Convert quaternion to angle and axis"""
    v = sqrt(np.sum(np.power(quat[0:3], 2)))  # get the magnitude of the axis components of the quaternion
    
    if v > 1e-10:
        r = quat[0:3] / v   # convert axis components of the quaternion to unit axis
        angle = 2 * atan2(v, quat[3])   # calculate angle 
    else:   # if the axis component has no magnitude, it is pointing in Z direction with zero angle
        r = np.array([0, 0, 1])
        angle = 0.

    angle = atan2(sin(angle), cos(angle))
    return angle,r

def axang2quat(angle,r):
    """ Convert angle and axis to quaternion """
    quat = np.zeros(4)  # set up the quaternion
    
    quat[0:3] = sin(angle/2.)*r # first 3 values are the real part in the direction of the axis
    quat[3] = cos(angle/2.)     # last value is the imaginary part

    return quat


def quaternionProduct(quat1,quat2):
    """ computes the product of two quaternions """
    quat_prod = np.zeros(4) # set up resulting quaternion as zeros

    eta1 = quat1[3]     # imaginary part
    xi1 = quat1[0:3]    # real part

    eta2 = quat2[3]     # imaginary part
    xi2 = quat2[0:3]    # real part

    eta_prod = eta1*eta2-np.matmul(xi1.T,xi2)       # imaginary part
    xi_prod = eta1*xi2+eta2*xi1+np.cross(xi1,xi2)   # real part
    
    # fill in the resulting quaternion
    quat_prod[0:3] = xi_prod
    quat_prod[3] = eta_prod

    return quat_prod


def load_obstacle():
    """ load the obstacle used in the second part of this coursework """

    poses = [Pose(position=Point(x=0.25, y=0.2, z=1.6))]    # pose
    files = ["ball"]                                        # file
    names = ["ball"]                                        # name
    reference_frame = "world"                               # reference frame

    dir = os.getcwd()
    for pose, file, name in zip(poses,files,names):
        # load the model xml from SDF
        model_xml=open(dir + "/models/" + file + "/model.sdf", "r").read()
        # spawn the model
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf(name, model_xml, "/",
                                pose, reference_frame)


def load_gazebo_models():
    """ load the Gazebo models used in this code """
    # poses of the models
    poses = [Pose(position=Point(x=0.75, y=0.45, z=0.0)),
             Pose(position=Point(x=0.75, y=-0.45, z=0.0)),
             Pose(position=Point(x=0.75, y=0.5, z=0.9)),
             Pose(position=Point(x=0.75, y=-0.5, z=0.9)),
             Pose(position=Point(x=0.75, y=0.0, z=0.9)),
             Pose(position=Point(x=0.75, y=0., z=0.88))]
             
    # files containing the models
    files = ["cafe_table", "cafe_table",
             "pick_plate", "place_plate",
             "middle_plate",
             "box"]
             
    # names of the models in Gazebo (each model needs a unique name)
    names = ["cafe_table_1", "cafe_table_2",
             "pick_plate", "place_plate",
             "middle_plate", "brick"]
    
    # poses are given in the reference frame of the world
    reference_frame = "world"

    dir = os.getcwd()
    for pose, file, name in zip(poses,files,names):
        # Load model xml from SDF
        model_xml = ''
        with open (dir+"/models/"+file+"/model.sdf", "r") as model_file:
            model_xml=model_file.read().replace('\n', '')
            
        # spawn the model
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            resp_sdf = spawn_sdf(name, model_xml, "/",
                                pose, reference_frame)
        except:
            pass


def delete_gazebo_models():
    """ deletes the gazebo models used in this code """
    models = ["ball","cafe_table_1", "cafe_table_2", "pick_plate", "place_plate",
                "middle_plate", "brick"]
    for model in models:    # loop through the models
        # delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        # resp_delete = delete_model(str(model))
        cmd = 'gz model -m '+model+' -d'    # send the command to Gazebo to delete the model
        os.system(cmd)
        print("DELETING", model)
    
    
#########################################################################

#### PART TO BE FILLED IN ################################################

class VelocityController(b_pykdl.baxter_kinematics):
    def __init__(self, limb, link_name):
        """ Velocity controller for the baxter robot arm """
        super(VelocityController, self).__init__(limb)

        # Desired link chain
        self._limb = limb
        self._limb_name = limb
        self._link = limb + '_' + link_name
        self._link_frame = PyKDL.Frame()
        self._link_chain = self._kdl_tree.getChain(self._base_link,
                                                  self._link)

        self._gripper = baxter_interface.Gripper(limb)

        self.joint_limits = np.array([0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0])
        self.joint_des = np.array([-0.3824899 , -0.51624778, -1.03770004,  2.47872251,  2.94037957,
        1.85541637, -2.33059887])

        # KDL Solvers for the secondary link (in this case, the elbow)
        # Forward kinematics
        self._fk_p_kdl_link = PyKDL.ChainFkSolverPos_recursive(self._link_chain)
        self._fk_v_kdl_link = PyKDL.ChainFkSolverVel_recursive(self._link_chain)

        # inverse kinematics
        self._ik_v_kdl_link = PyKDL.ChainIkSolverVel_pinv(self._link_chain)
        self._ik_p_kdl_link = PyKDL.ChainIkSolverPos_NR(self._link_chain,
                                                   self._fk_p_kdl_link,
                                                   self._ik_v_kdl_link)
        self._jac_kdl_link = PyKDL.ChainJntToJacSolver(self._link_chain)
        
        # open the gripper
        self._gripper.open()
        rospy.sleep(2.0)

    def link_forward_position_kinematics(self, joint_values=None):
        """ forward position kinematics for secondary link (in this case, the elbow) """
        end_frame = PyKDL.Frame()
        q_kdl = PyKDL.JntArray(joint_values.shape[0])
        for i in range(joint_values.shape[0]):
            q_kdl[i] = joint_values[i]
        self._fk_p_kdl_link.JntToCart(q_kdl,
                                 end_frame)
        pos = end_frame.p
        rot = PyKDL.Rotation(end_frame.M)
        rot = rot.GetQuaternion()
        return np.array([pos[0], pos[1], pos[2],
                         rot[0], rot[1], rot[2], rot[3]]).reshape((self._num_jnts, 1))

    def move_to_joint_position(self, joint_values = np.zeros(7), timeout=15):
        """ move the arm to specified joint positions """
        if joint_values.shape[0] > 0:
            q = {self._limb_name+'_s0': joint_values[0],
                            self._limb_name+'_s1': joint_values[1],
                            self._limb_name+'_e0': joint_values[2],
                            self._limb_name+'_e1': joint_values[3],
                            self._limb_name+'_w0': joint_values[4],
                            self._limb_name+'_w1': joint_values[5],
                            self._limb_name+'_w2': joint_values[6]}
            self._limb_interface.move_to_joint_positions(q, timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def set_joint_positions(self, joint_values=np.zeros(7)):
        """ directly set the joint positions of the arm """
        q_jnt = {self._limb_name + '_s0': joint_values[0],
                     self._limb_name + '_s1': joint_values[1],
                     self._limb_name + '_e0': joint_values[2],
                     self._limb_name + '_e1': joint_values[3],
                     self._limb_name+ '_w0': joint_values[4],
                     self._limb_name + '_w1': joint_values[5],
                     self._limb_name + '_w2': joint_values[6]}
        self._limb_interface.set_joint_positions(q_jnt, raw=True)

    def set_joint_velocities(self, joint_values=np.zeros(7)):
        """ set the joint velocities of the arm """
        qd_jnt = {self._limb_name+ '_s0': joint_values[0],
                 self._limb_name + '_s1': joint_values[1],
                 self._limb_name + '_e0': joint_values[2],
                 self._limb_name + '_e1': joint_values[3],
                 self._limb_name+ '_w0': joint_values[4],
                 self._limb_name+ '_w1': joint_values[5],
                 self._limb_name+ '_w2': joint_values[6]}
        self._limb_interface.set_joint_velocities(qd_jnt)

    def link_jacobian(self, joint_values=None):
        """ computes the Jacobian of the secondary link (in this case, the elbow) """
        # take only initial 4 joints (up to elbow)
        q_elbow = np.copy(joint_values[0:4])

        nj = q_elbow.shape[0]           # number of joints up to the secondary link
        nj_tot = self._num_jnts         # total number of joints in the robot arm
        jacobian = PyKDL.Jacobian(nj)   # set up a PyKDL Jacobian of the right size for the secondary link
        q_kdl = PyKDL.JntArray(nj)      # set up a PyKDL joint array to calculate the Jacobian
        for i in range(nj):             # loop through the joints up to the secondary link
            q_kdl[i] = q_elbow[i]       # fill in the PyKDL joint array

        ############################
        # Task E:
        # Fill in the function to compute elbow Jacobian. Inputs are the joint values in KDL and the jacobian matrix
        self._jac_kdl_link.JntToJac([], [])

        J_link = self.kdl_to_mat(jacobian) # convert the jacobian from PyKDL format to numpy array
        
        # after computing the jacobian for the elbow, we need to convert it to a full size jacobian 
        J = np.zeros((6, nj_tot)) # Total Jacobain matrix. Fill in with J_link of elbow
        J = []

        return J[0:3,:] # take only linear part of the jacobian

    def ee_IK_solver(self, joint_values, P_des, quat_des, dt, nullspace_en, vel_elbow):
        """ solves the inverse velocity kinematics of the end effector """
        
        nj = joint_values.shape[0]  # number of joints
        
        # current pose calculated using forward position kinematics
        Pose_vect = self.forward_position_kinematics(joint_values)
        P = Pose_vect[0:3].reshape(-1)      # position
        quat = Pose_vect[3:].reshape(-1)    # orientation

        # calculate the orientation error of the end effector
        # we need to calculate the relative quaternion between the final orientation and the initial orientation
        # this is q_init^-1 * q_final
        
        # first take the inverse of the initial orientation quaternion       
        quat_inv = np.copy(quat)
        quat_inv[0:3] = -quat_inv[0:3]

        # then compute the quaternion product of the inverse initial orientation quaternion with the final orientation quaternion
        quat_err = quaternionProduct(quat_des, quat_inv)
        # convert this to an axis and rotation around that axis
        delta_angle, r = quat2angax(quat_err)   # angle and axis of error

        ##########################
        ##### Task D
        # compute linear and angular velocities given P_des, P, delta_angle, r, and dt
        dP = []  # linear displacement. Note that P_des is the desired end-effector position at the next time instant.
        dw = np.array([0,0,0]) # angular displacement. To be edited only by groups for part ii
        
        # twist is [linear velocity, angular velocity]
        twist = np.hstack((dP, dw))

        ##########################
        ##### Task E
        # compute Jacobian of the end-effector and solve velocity IK with pseudoinverse. Use self.jacobian() with joint_values as inputs
        J_ee = self.jacobian([])    # your code here, replace [] with the correct variable
        J_ee = np.asarray(J_ee) # convert to np.array

        J_pinv = DPinv([], eps=1e-10) # your code here, replace [] to compute the pseudoinverse of J_ee
        qd = [] # your code here, replace [] with a matrix multiplication to compute joint velocities from twist

        if nullspace_en:
            ##########################
            #####Task F
            # Compute projector into the ee Jacobian null space and joint velocity to reach desired configuration
            I = np.eye(nj)
            Proj = []   # this is the null space projector, N. Replace [] with your calculation.

            ##########################
            #####Task G part i
            # Compute secondary joint velocities to reach desired configuration q_desired, given the current joint values joint_values
            # and sampling time dt
            q_desired = self.joint_des

            qd2 = []    # replace [] with secondary joint velocities calculated from q_desired, joint_values, and sampling time dt.
            qd2 = 0.01 * qd2 # note that q_desired is the final joint configuration, so we artificially slow the speed down here (as if it is interpolating over a longer time period).

            ##########################
            #####Task G part ii
            J_elbow = self.link_jacobian(joint_values)
            
            # uncomment the lines below
            # J_pinv_elbow = DPinv([], 1e-6)    # replace [] with the elbow Jacobian
            # qd2 = []  # and replace [] with a matrix multiplication to compute secondary joint velocities given vel_elbow



            qd = qd + np.matmul(Proj, qd2)  # projection in Null space

        return qd

    def Velocity_IK_solver(self, type, joint_values, T, xyz_des, rpy_des=[0, 0, 0]):
        """ handles all the velocity motion """
        # convert world frame coordinates to DE NIRO base coordinates with a z offset
        z_base_offset = 0.93
        
        # set up timing
        t = 0   # initial time
        dt = 1. / 100   # timestep (in seconds)

        # initial pose
        Pose_vect = self.forward_position_kinematics(joint_values)  # initial pose (position and orientation)
        P_init = Pose_vect[0:3].reshape(-1)     # get the position as xyz coordinates
        quat_init = Pose_vect[3:].reshape(-1)   # get the orientation as a quaternion

        if type != "path":  # if we aren't following a path (the circle motion), the final end effector position is known
            P_final = np.array(xyz_des) - np.array([0, 0, z_base_offset])
        
        # calculate the desired orientation of the end effector
        roll, pitch, yaw = rpy_des  # given to the function as roll pitch yaw
        quat_final = tf.transformations.quaternion_from_euler(roll, pitch, yaw)     # converted to quaternion
        
        # calculate the orientation error of the end effector
        # we need to calculate the relative quaternion between the final orientation and the initial orientation
        # this is q_init^-1 * q_final
        
        # first take the inverse of the initial orientation quaternion
        quat_inv = np.copy(quat_init)
        quat_inv[0:3] = -quat_inv[0:3]  
        
        # then compute the quaternion product of the inverse initial orientation quaternion with the final orientation quaternion
        quat_err_final = quaternionProduct(quat_inv, quat_final)
        # convert this to an axis and rotation around that axis
        angle_final, r_final = quat2angax(quat_err_final)
        
        # set up arrays to store values during the motion
        X = []
        Y = []
        Z = []
        Roll = []
        Pitch = []
        Yaw = []

        X_expect = []
        Y_expect = []
        Z_expect = []
        Roll_expect = []
        Pitch_expect = []
        Yaw_expect = []

        X_des = []
        Y_des = []
        Z_des = []
        Roll_des = []
        Pitch_des = []
        Yaw_des = []

        P_des = []
        quat_des = []
        vel_elbow = []
        
        # nullspace flag
        nullspace_en = False
        
        # loop for the duration of the motion
        while t <= T:
            
            # use linear inerpolation to go from initial orientation to final orientation
            angle_des = angle_final * t / T
            quat_err_des = axang2quat(angle_des, r_final)               # in init_frame
            quat_des = quaternionProduct(quat_init, quat_err_des)       # in 0 frame
            rpy_des = tf.transformations.euler_from_quaternion(quat_des.reshape(-1))    # get desired roll pitch yaw

            # if we're just moving to a pose
            if type == "go2pose":
                # perform linear interpolation from initial to final pose
                P_des = (P_final - P_init) * t / T + P_init #desired path at each instant t
            
            # if we're following a path
            elif type == "path":
                radius = 0.05   # radius of the circle
                alpha = 2. * pi * t / T - pi    # angle around the circle as a function of time
                P_center = P_init + np.array([0, radius, 0])    # center of the circle (relative to the world)
                Pr = radius * np.array([0, cos(alpha), sin(alpha)])     # position of the end effector on the circle (relative to the center)
                P_des = P_center + Pr   # position of the end effector on the circle (relative to the world)
                self._gripper.close()   # keep the gripper closed
                
            # if we're performing nullspace motion    
            elif type == "nullspace":
                # perform linear interpolation of the end effector pose
                P_des = (P_final - P_init) * t / T + P_init #desired path at each instant t
                nullspace_en = True

                # desired elbow motion
                radius = 0.01
                alpha = -pi / 4. * t / T + pi / 4.
                alpha_d = -pi / 4. * 1. / T
                vel_dir = np.array([0, -sin(alpha), cos(alpha)])
                # circular velocity to elbow
                vel_elbow = radius * alpha_d * vel_dir
                self._gripper.close()
            
            # solve inverse velocity kinematics to get desired joint velocities
            qd = self.ee_IK_solver(joint_values, P_des, quat_des, dt, nullspace_en, vel_elbow)
            
            # get the current robot state
            q_current = self.current_robot_state()[0]
            
            # use forward kinematics to get current end effector pose
            Pose_current = self.forward_position_kinematics(q_current.reshape(-1))
            # calculate the expected end effector pose
            Pose_expect = self.forward_position_kinematics(joint_values)
            
            # end effector position (current and expected)
            P_current = Pose_current[0:3]
            P_expect = Pose_expect[0:3]
            
            # end effector orientation (current and expected)
            rpy_current = tf.transformations.euler_from_quaternion(Pose_current[3:].reshape(-1))
            rpy_expect = tf.transformations.euler_from_quaternion(Pose_expect[3:].reshape(-1))
            
            # commanded joint velocity = desired joint velocity + error between expected joint values and current joint values
            qd_command = qd + 0.1 / dt * (joint_values - q_current.reshape(-1))
            
            # set the joint velocities
            self.set_joint_velocities(qd_command)
            
            # update joint values for next iteration
            joint_values = joint_values + qd * dt

            # printout everything
            print("************")
            print("time ", t, " / ", T)
            print("P des ", P_des)
            print("P current ", P_current)
            print("P expect ", P_expect)
            print("rpy des ", rpy_des)
            print("rpy current ", rpy_current)
            print("rpy expect ", rpy_expect)
            print("joint_values", joint_values)
            # print("joint_values current", q_current.reshape(-1))

            # store values for plotting etc. afterwards
            X.append(P_current[0])
            Y.append(P_current[1])
            Z.append(P_current[2])
            Roll.append(rpy_current[0])
            Pitch.append(rpy_current[1])
            Yaw.append(rpy_current[2])

            X_expect.append(P_expect[0])
            Y_expect.append(P_expect[1])
            Z_expect.append(P_expect[2])
            Roll_expect.append(rpy_expect[0])
            Pitch_expect.append(rpy_expect[1])
            Yaw_expect.append(rpy_expect[2])

            X_des.append(P_des[0])
            Y_des.append(P_des[1])
            Z_des.append(P_des[2])
            Roll_des.append(rpy_des[0])
            Pitch_des.append(rpy_des[1])
            Yaw_des.append(rpy_des[2])

            rate.sleep()
            t = t + dt
            
        ######## after motion has completed
        # final pose at time T
        Pose_expect = self.forward_position_kinematics(joint_values)
        q_current = self.current_robot_state()[0]
        Pose_current = self.forward_position_kinematics(q_current.reshape(-1))

        # block robot at final joint positions
        self.move_to_joint_position(joint_values, 1)

        # plt.figure()
        # plt.plot(range(len(X)), X, '-r',label='x curr')
        # plt.plot(range(len(X)), Y, '-g',label='y curr')
        # plt.plot(range(len(X)), Z, '-b',label='z curr')
        # plt.plot(range(len(X)), X_expect, '-.r',label='x expect')
        # plt.plot(range(len(X)), Y_expect, '-.g',label='y expect')
        # plt.plot(range(len(X)), Z_expect, '-.b',label='z expect')
        # plt.plot(range(len(X)), X_des, '--r',label='x des')
        # plt.plot(range(len(X)), Y_des, '--g',label='y des')
        # plt.plot(range(len(X)), Z_des, '--b',label='z des')
        # plt.title('Positions')
        # plt.xlabel('time (s)')
        # plt.legend(loc="upper left")
        #
        # plt.figure()
        # plt.plot(range(len(X)), Roll, '-r',label='roll curr')
        # plt.plot(range(len(X)), Pitch, '-g',label='pitch curr')
        # plt.plot(range(len(X)), Yaw, '-b',label='yaw curr')
        # plt.plot(range(len(X)), Roll_expect, '-.r', label='roll _expect')
        # plt.plot(range(len(X)), Pitch_expect, '-.g', label='pitch _expect')
        # plt.plot(range(len(X)), Yaw_expect, '-.b', label='yaw _expect')
        # plt.plot(range(len(X)), Roll_des, '--r',label='roll des')
        # plt.plot(range(len(X)), Pitch_des, '--g',label='pitch des')
        # plt.plot(range(len(X)), Yaw_des, '--b',label='yaw des')
        # plt.title('RPY')
        # plt.xlabel('time (s)')
        # plt.legend(loc="upper left")
        # plt.show()

        return joint_values, Pose_current


def reachPose(Arm, q, xyz_des, rpy_des=[0,0,0]):
    """ high level function to pick up an object at pose xyz_des, rpy_des """
    # approach from 0.18 m above
    xyz_approach = [xyz_des[0], xyz_des[1], xyz_des[2] + 0.18]
    q, Pose_current = Arm.Velocity_IK_solver("go2pose", q.reshape(-1), 10, xyz_approach, rpy_des)

    # move down to the object and grip
    q, Pose_current = Arm.Velocity_IK_solver("go2pose", q.reshape(-1), 5, xyz_des, rpy_des)
    Arm._gripper.close()
    rospy.sleep(1.0)
    
    # move up again
    q, Pose_current = Arm.Velocity_IK_solver("go2pose", q.reshape(-1), 5, xyz_approach, rpy_des)

    return q


def Circle(Arm, q, xyz_des, rpy_des=[0,0,0]):
    """ high level function to start following a circular path  """
    # Go to intial pose to start circular motion
    q, Pose_current = Arm.Velocity_IK_solver("go2pose", q.reshape(-1), 10, xyz_des, rpy_des)

    # Perfrom Circle
    q, Pose_current = Arm.Velocity_IK_solver("path", q.reshape(-1), 15, [], rpy_des)

    return q


def NullSpace(Arm,q,xyz_des,rpy_des = [0,0,0]):
    """ high level function to start performing null space motion to avoid an obstacle """
    # Perform Null Space Motion to avoid obstacle
    load_obstacle() # load the obstacle
    rospy.sleep(2.0)    # wait 2 seconds to make sure obstacle has spawned correctly
    q, Pose_current = Arm.Velocity_IK_solver("nullspace", q.reshape(-1), 10, xyz_des, rpy_des) # perform nullspace motion
    return q


def main(task):
    
    cmd = 'rosrun baxter_tools tuck_arms.py -u'
    os.system(cmd)
    cmd = 'rosrun baxter_tools tuck_arms.py -u'
    os.system(cmd)
    cmd = 'rosrun baxter_tools tuck_arms.py -u'
    os.system(cmd)

    rospy.sleep(2)

    load_gazebo_models()

    rospy.sleep(2)

    # Robot Arm object for velocity control
    Arm = VelocityController('left', 'lower_elbow')
    Arm._gripper.open()
    rospy.sleep(1.0)

    # set initial robot configuration
    q, qd, tau = Arm.current_robot_state()
    q[6] = -45 * pi / 180
    Arm.move_to_joint_position(q.reshape(-1), 2)
    
    ######################################################
    ## Task C:
    # fill in the desired poses
    xyz_des_pick = []       # your code here!
    xyz_des_circle = []     # your code here!
    rpy_des = [-np.pi, 0, np.pi]  # this is used only for groups. For individuals it has no effect
    if task == "go2pose":
        q = reachPose(Arm, q, xyz_des_pick, rpy_des)

    elif task == "path":
        q = reachPose(Arm, q, xyz_des_pick, rpy_des)
        q = Circle(Arm, q, xyz_des_circl, rpy_des)

    elif task == "nullspace":
        q = reachPose(Arm, q, xyz_des_pick, rpy_des)
        q = Circle(Arm, q, xyz_des_circle, rpy_des)

        xyz_des_ns = xyz_des_circle
        q = NullSpace(Arm, q, xyz_des_ns, rpy_des)

    print("TASK COMPLETED")
    delete_gazebo_models()
    rospy.sleep(2)


if __name__ == '__main__':
    tasks = ['go2pose', 'path', 'nullspace']
    if len(sys.argv) <= 1:
        print 'Please include a task to run from the following options:\n', tasks
    else:
        task = str(sys.argv[1])
        if task in tasks:
            print "Running Velocity Control -", task
            main(task)
        else:
            print 'Please include a task to run from the following options:\n', tasks
    rospy.sleep(2)
    rospy.signal_shutdown("FINISHED")
