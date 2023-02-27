import os
import sys
import copy
import time
import rospy
import struct
import argparse
import numpy as np
import tf
# baxter specific imports
import baxter_interface
from baxter_pykdl import baxter_kinematics
from baxter_core_msgs.msg import (SEAJointState)
from baxter_core_msgs.srv import (SolvePositionIK, SolvePositionIKRequest)
from dynamic_reconfigure.server import (Server)
from baxter_examples.cfg import (JointSpringsExampleConfig)
# ros specific imports
from gazebo_msgs.srv import (SpawnModel, DeleteModel)
from geometry_msgs.msg import (PoseStamped, Pose, Point, Quaternion)
from std_msgs.msg import (Header, Empty, Float64MultiArray)


DE_NIRO_height_offset = np.array([0, 0, 0.93])
p0 = np.array([0.50, 0.60, 0.76])
z_padding = 1e-2
brick_width = 0.086 + 1e-2     # width of a brick
brick_depth = 0.062 + 1e-2     # depth of a brick
brick_height = 0.192 + 1e-2    # height of a brick
pos = p0 + np.array([0, 0, brick_depth/2 + z_padding])
orientation = np.array([np.pi/2, np.pi/2, np.pi/2])
brick_pose = (pos, orientation)
sponge_xyz = np.array([0.75, -0.3, 0.76]) + np.array([0, 0, brick_depth/2 + z_padding])
sponge_pose = (sponge_xyz, orientation)


################################################
#### AUXILIARY FUNCTIONS########################
def array_to_pose(xyz, rpy=None):
    """ convert position and orientation to ROS message pose """
    position = Point(x=xyz[0], y=xyz[1], z=xyz[2])
    if rpy is not None:
        # convert roll pitch yaw to quaternion
        q = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        pose = Pose(position=position, orientation=orientation)
    else:
        pose = Pose(position=position)
    return pose


def load_tables():
    """ load the tables and plates """
    poses = [Pose(position=Point(x=0.75, y=0.45, z=0.0)),
             Pose(position=Point(x=0.75, y=-0.45, z=0.0)),
             Pose(position=Point(x=0.75, y=0.5, z=0.78)),
             Pose(position=Point(x=0.75, y=-0.5, z=0.78)),
             Pose(position=Point(x=0.75, y=0.0, z=0.78))]
    files = ["cafe_table", "cafe_table", "pick_plate", "place_plate", "middle_plate"]
    names = ["cafe_table_1", "cafe_table_2", "pick_plate", "place_plate", "middle_plate"]   
    reference_frame = "world" 

    for pose, file, name in zip(poses,files,names):
        # Load xml from SDF
        model_xml = ''
        with open ("models/"+file+"/model.sdf", "r") as model_file:
            model_xml=model_file.read().replace('\n', '')
        # Spawn plate model
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            resp_sdf = spawn_sdf(name, model_xml, "/",
                                pose, reference_frame)
        except:
            pass 


def delete_tables():
    """ delete the tables and plates """
    models = ["cafe_table_1", "cafe_table_2", "pick_plate", "place_plate", "middle_plate"]
    for model in models:
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model(model)
        except:
            pass


def load_wall():
    """ load the wall to destroy """
    p0 = np.array([0.35, -0.70, 0.76])  # position of the first brick
    brick_width = 0.086 + 1e-2     # width of a brick
    brick_depth = 0.062 + 1e-2     # depth of a brick
    brick_height = 0.192 + 1e-2    # height of a brick
    z_padding = 0.01    # add padding in the z direction
    x_spacing = 0.1     # spacing between bricks in the x direction
    N_rows = 4  # number of rows in the wall (1 row = 1 vertical brick and 1 horizontal brick on top of it)
    N_cols = 6  # number of columns in the wall (1 column = 1 vertical brick, or 0.5 horizontal brick on top of it)
    poses = []
    names = []
    reference_frame = "world" 
    for i in range(N_rows):
        # layer of | | | | | |
        for j in range(N_cols):
            xyz = p0 + np.array([x_spacing * j, 0, (brick_height + brick_width) * i + brick_height/2 + z_padding])
            rpy = np.array([0, 0, np.pi/2])
            pose = array_to_pose(xyz, rpy)
            poses.append(pose)
            names.append('brick' + str(i) + str(j))
        # layer of  __ __ __
        for k in range(int(N_cols / 2)):
            xyz = p0 + np.array([x_spacing * (2 * k + 0.5), 0, (brick_height + brick_width) * (i + 1) - brick_width / 2])
            rpy = np.array([np.pi/2, 0, np.pi/2])
            pose = array_to_pose(xyz, rpy)
            poses.append(pose)
            names.append('brick' + str(i) + str(k) + 'horizontal')
            
    for pose, name in zip(poses, names):
        file = "Brick"
        model_xml = ''
        with open ("models/"+file+"/model.sdf", "r") as model_file:
            # load xml from SDF
            model_xml=model_file.read().replace('\n', '')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            # spawn the model
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            resp_sdf = spawn_sdf(name, model_xml, "/",
                                pose, reference_frame)
        except Exception as e:
            print(e)
            pass 

def delete_wall():
    """ delete the wall """
    N_rows = 4
    N_cols = 6
    names = []
    for i in range(N_rows):
        for j in range(N_cols):
            names.append('brick' + str(i) + str(j))
        for k in range(int(N_cols / 2)):
            names.append('brick' + str(i) + str(k) + 'horizontal')
    for name in names:
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp_delete = delete_model(name)
        except Exception as e:
            print(e)
            pass


def load_brick():
    """ load the heavy brick used to demolish the wall """
    file = "Brick_Heavy"
    xyz, rpy = brick_pose
    pose = array_to_pose(xyz, rpy)
    name = 'brick_hit'
    model_xml = ''
    reference_frame = "world" 
    with open ("models/"+file+"/model.sdf", "r") as model_file:
        model_xml=model_file.read().replace('\n', '')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf(name, model_xml, "/",
                            pose, reference_frame)
    except Exception as e:
        print(e)
        pass


def delete_brick():
    """ delete the heavy brick used to demolish the wall """
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model('brick_hit')
    except Exception as e:
        print(e)
        pass


def load_sponge():
    """ load the sponge used to clean the table """
    file = "Sponge"
    xyz, rpy = sponge_pose
    pose = array_to_pose(xyz, rpy)
    name = 'sponge'
    model_xml = ''
    reference_frame = "world" 
    with open ("models/"+file+"/model.sdf", "r") as model_file:
        model_xml=model_file.read().replace('\n', '')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf(name, model_xml, "/",
                            pose, reference_frame)
    except Exception as e:
        print(e)
        pass


def delete_sponge():
    """ delete the sponge used to clean the table """
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model('sponge')
    except Exception as e:
        print(e)
        pass


class BaxterArm(object):
    """ Class to operate the Baxter robot arm """
    def __init__(self, limb, verbose=True):
        # initialise the arm
        self._limb_name = limb 
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        self._start_angles = np.zeros((7, 1))
        
        # verbose flag (for greater detail when debugging)
        self._verbose = verbose 
        
        # enable the robot
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        
        # set up the kinematic solvers (PyKDL)
        self.kin = baxter_kinematics(limb)
    
    def ik_request(self, pose):
        """ uses Baxter's internal inverse kinematics solver to calculate inverse position kinematics """
        # set up the pose message sent to the solver
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)   # send the pose to the solver, get response (resp)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
            
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):  # check if the response is valid
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
                         
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print '[INFO] IK joint solution found!!!'
                print '========================= Joint Space Target State ===================='
                for joint in limb_joints:
                    print joint+': ', np.round(limb_joints[joint],4)
                print("------------------")
            return limb_joints
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
    
    def apply_torque(self, torques):
        """ sets the joint torques of the robot arm """
        command_torques = {self._limb_name+'_s0': torques[0], 
                           self._limb_name+'_s1': torques[1], 
                           self._limb_name+'_e0': torques[2], 
                           self._limb_name+'_e1': torques[3],
                           self._limb_name+'_w0': torques[4],
                           self._limb_name+'_w1': torques[5],
                           self._limb_name+'_w2': torques[6]}
        self._limb.set_joint_torques(command_torques)

    def move_to_joint_position(self, joint_angles):
        """ move the robot arm to given joint angles """
        if joint_angles is not None:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
            
    def gripper_open(self):
        """ Open the gripper """
        print '[ACTION] Open '+self._limb_name+' gripper...'
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        """ Close the gripper """
        print '[ACTION] Close '+self._limb_name+' gripper...'
        self._gripper.close()
        rospy.sleep(1.0)
    
    def servo_to_joint_position(self, q):
        """ High level function to move to joint position q
        q is an array or list, this converts it to a dictionary readable by the
        Baxter API before sending to the usual move_to_joint_position function """
        print '========================= Joint Space Target State ========================='
        joint_angles = {self._limb_name+'_s0': q[0], 
                        self._limb_name+'_s1': q[1], 
                        self._limb_name+'_e0': q[2], 
                        self._limb_name+'_e1': q[3],
                        self._limb_name+'_w0': q[4],
                        self._limb_name+'_w1': q[5],
                        self._limb_name+'_w2': q[6]}
        self.move_to_joint_position(joint_angles)

    def servo_to_pose(self, xyz, rpy):
        """ High level function to move to end effector pose defined by position xyz and orientation rpy """
        
        print '========================= Task Space Target State =========================='
        print 'position:'
        print 'x: ', np.around(xyz[0], 4)
        print 'y: ', np.around(xyz[1], 4)
        print 'z: ', np.around(xyz[2], 4)
        print 'orientation:'
        print 'roll:', np.around(rpy[0], 4)
        print 'pitch:', np.around(rpy[1], 4)
        print 'yaw:', np.around(rpy[2], 4)
        roll, pitch, yaw = rpy
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        print 'quaternion:' 
        print 'x: ', np.around(quaternion[0], 4)
        print 'y: ', np.around(quaternion[1], 4)
        print 'z: ', np.around(quaternion[2], 4)
        print 'w: ', np.around(quaternion[3], 4)
        print("------------------")
        
        # set up the pose ROS message for the inverse kinematics solver
        ik_pose = Pose()
        ik_pose.position.x = xyz[0]
        ik_pose.position.y = xyz[1]
        ik_pose.position.z = xyz[2] 
        ik_pose.orientation.x = quaternion[0]
        ik_pose.orientation.y = quaternion[1]
        ik_pose.orientation.z = quaternion[2]
        ik_pose.orientation.w = quaternion[3]

        print '[ACTION] Finding inverse kinematic solution...'
        joint_angles = self.ik_request(ik_pose)     # inverse kinematics from the solver
        print '[ACTION] Move the '+self._limb_name+' arm to the target pose!!!'
        self.move_to_joint_position(joint_angles)   # move to these joint positions
        print '[INFO] Target pose is achieved!!!'
    

def main():
    # delete all the models 
    delete_sponge()
    delete_brick()
    delete_wall()
    delete_tables()
    
    # Load Gazebo Models via Spawning Services
    
    # set up the control frequency and timing
    freq = 1000.0  # Hz
    dt = 1/freq # s
    rate = rospy.Rate(freq)
    
    ################ Initialise baxter arms
    print "Initialising right arm"
    # right arm
    right_arm = BaxterArm('right')
    right_arm.gripper_open()
    right_arm._limb.move_to_neutral()
    q_right, _, _ = right_arm.kin.current_robot_state()
    
    # move the right arm competely out of the way
    q_right[0] -= 1.25
    right_arm.servo_to_joint_position(q_right)
    
    print "Initialising left arm"
    # left arm
    left_arm = BaxterArm('left')
    left_arm.gripper_open()
    left_arm._limb.move_to_neutral()
    q_left, _, _ = left_arm.kin.current_robot_state()
    
    ################ load the objects
    print "Loading the environment"
    load_tables()   # load two tables + plates in front of DE NIRO
    print "Tables spawned"
    load_wall()     # load the brick wall
    print "Wall spawned"
    load_brick()    # load the brick to knock down the wall with
    print "Brick spawned"
    
    ################ initial position control movements
    print "Picking up projectile"
    # Move above the brick and open gripper
    xyz, _ = brick_pose
    rpy = np.array([-np.pi, 0, np.pi])
    xyz = xyz + np.array([0, 0, 0.1]) - DE_NIRO_height_offset   # offset to account for DE NIRO's height
    left_arm.servo_to_pose(xyz, rpy)
    left_arm.gripper_open()
    rospy.sleep(1.0)
    
    # Move down and grasp the brick
    xyz = xyz + np.array([0, 0, -0.1])
    left_arm.servo_to_pose(xyz, rpy)
    left_arm.gripper_close()
    gripper_closed = True
    rospy.sleep(1.0)
    
    # Lift the brick up
    xyz = xyz + np.array([0, 0, 0.1])
    left_arm.servo_to_pose(xyz, rpy)
    rpy = np.array([-np.pi, 0, np.pi/2])
    left_arm.servo_to_pose(xyz, rpy)
    
    print "Aiming"
    # position of the end effector before throwing the brick
    p0 = left_arm.kin.forward_position_kinematics()
    x0 = p0[0][0]
    y0 = p0[1][0]
    z0 = p0[2][0]
    
    # the y position to release the brick
    y_stop = -0.0
    
    # initial time
    t_start = rospy.Time.now().to_sec()
    rate.sleep()
    
    # the torque control demolition loop
    print "Demolishing wall!"
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t_start     # current time (since start of loop)
        
        # get current joint angles, joint velocities, joint torques
        q, q_dot, tau = left_arm.kin.current_robot_state()
        
        # jacobian   
        J = left_arm.kin.jacobian()
        
        # forward kinematics - get the current end effector position
        p = left_arm.kin.forward_position_kinematics()
        x = p[0][0]
        y = p[1][0]
        z = p[2][0]
        
        # forward velocity kinematics - get the current end effector velocity
        p_dot = np.matmul(J, q_dot)
        x_dot = p[0][0]
        y_dot = p[1][0]
        z_dot = p[2][0]
        x_omega = p[3][0]
        y_omega = p[4][0]
        z_omega = p[5][0]
    
        # external torque
        ############################ TASK H
        # change external_force to make DE NIRO throw the brick into the wall (your code here)
        external_force = np.array([0.0, 0.0, 0.0])            # force to accelerate the end effector toward the wall
        
        position_error = np.array([x0 - x, y0 - y, z0 - z])     # position error
        velocity_error = np.array([- x_dot, - y_dot, - z_dot])   # velocity error
        
        # calculate the component of the error perpendicular to the external force
        perpendicular_position_error = position_error - external_force * np.dot(position_error, external_force) / np.dot(external_force, external_force)
        perpendicular_velocity_error = velocity_error - external_force * np.dot(velocity_error, external_force) / np.dot(external_force, external_force)

        proportional_feedback = 100.0 * perpendicular_position_error
        derivative_feedback = 10.0 * perpendicular_velocity_error
        
        print 'proprtional feedback:\n', proportional_feedback
        print 'derivative feedback:\n', derivative_feedback
        print 'command force:\n', external_force
        
        external_force = external_force + proportional_feedback + derivative_feedback
        
        # convert the external force that the end effector is applying to joint torques
        external_torque = np.matmul(J[:3].T, external_force.reshape((3, 1)))
        
        # if the arm moves past the y point, open the gripper
        if y < y_stop + 0.1 and gripper_closed:
            left_arm.gripper_open()
            gripper_closed = False
        
        # then stop the arm from moving
        if y < y_stop:
            # external torque
            external_force = np.zeros((3, 1))
            # proportional and derivative control to stop the end effector from moving in x y z
            external_force += 100.0 * np.array([x0 - x, y_stop - y, z0 - z]).reshape((3, 1))
            external_force += 10.0 * np.array([- x_dot, - y_dot, - z_dot]).reshape((3, 1))
            external_torque = np.matmul(J[:3].T, external_force)    # just the position part of the jacobian (not orientation)
            break
        
        # apply the torques to the arm
        left_arm.apply_torque(external_torque)
        
        # sleep
        rate.sleep()
    
    ################ cleanup operation - for groups only
    print "Cleaning up :("
    # move the left arm back to neutral position, then move it out of the way
    left_arm.servo_to_joint_position(q_left)
    q_left[0] += 1.25
    left_arm.servo_to_joint_position(q_left)
    
    # load the yellow sponge to clean the table
    load_sponge()
    
    # move the right arm to neutral
    right_arm._limb.move_to_neutral()
    
    # move the end effector above the sponge and open the gripper
    xyz, _ = sponge_pose
    rpy = np.array([-np.pi, 0, np.pi])
    xyz = xyz + np.array([0, 0, 0.1]) - DE_NIRO_height_offset
    right_arm.servo_to_pose(xyz, rpy)
    right_arm.gripper_open()
    rospy.sleep(1.0)
    
    # move down and grasp the sponge
    xyz = xyz + np.array([0, 0, -0.1])
    right_arm.servo_to_pose(xyz, rpy)
    right_arm.gripper_close()
    gripper_closed = True
    rospy.sleep(1.0)
    
    print "Wiping right"
    # position of the end effector before starting wiping
    p0 = right_arm.kin.forward_position_kinematics()
    x0 = p0[0][0]
    y0 = p0[1][0]
    z0 = p0[2][0]
    
    # stopping points for the wiping motion    
    y_left_stop = 0.5
    y_right_stop = -0.4
    
    # starting direction to wipe
    direction = 'right'
    # counter for number of wipes completed
    n_wipes = 0
    
    # initial time
    t_start = rospy.Time.now().to_sec()
    rate.sleep()
    
    # the torque control wiping loop
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t_start     # current time (since start of loop)
        
        # get current joint angles, joint velocities, joint torques
        q, q_dot, tau = right_arm.kin.current_robot_state()
        
        # jacobian   
        J = right_arm.kin.jacobian()
        
        # forward kinematics - get the current end effector position
        p = right_arm.kin.forward_position_kinematics()
        x = p[0][0]
        y = p[1][0]
        z = p[2][0]
        
        # forward velocity kinematics - get the current end effector velocity
        p_dot = np.matmul(J, q_dot)
        x_dot = p[0][0]
        y_dot = p[1][0]
        z_dot = p[2][0]
        x_omega = p[3][0]
        y_omega = p[4][0]
        z_omega = p[5][0]
        
        external_torque = np.zeros((7, 1))
        if direction == 'right':
            ############ TASK I
            # external force (your code here)
            external_force = np.array([0.0, 0.0, 0.0, 0, 0, 0]).reshape((6, 1))    # force to push down and across on the table
            
            # add proportional and derivative control to keep the end effector from deviating in the x direction
            external_force += 400.0 * np.array([0.75 - x, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))                 # proportional feedback
            external_force += 20.0 * np.array([- x_dot, - y_dot * 0.2, - z_dot, 0.0, 0.0, 0.0]).reshape((6, 1))     # derivative feedback
            
            # convert the external force that the end effector is applying to joint torques
            external_torque = np.matmul(J.T, external_force)
            
            # if the y position goes behond the right hand limit, change direction
            if y < y_right_stop or t > 5:
                direction = 'left'
                t = 0
                print 'Wiping left'
        
        if direction == 'left':
            ############ TASK I (continued)
            # external force (your code here)
            external_force = np.array([0.0, 0.0, 0.0, 0, 0, 0]).reshape((6, 1))    # force to push down and across on the table
            
            # add proportional and derivative control to keep the end effector from deviating in the x direction
            external_force += 400.0 * np.array([0.75 - x, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))                 # proportional feedback
            external_force += 20.0 * np.array([- x_dot, - y_dot * 0.2, - z_dot, 0.0, 0.0, 0.0]).reshape((6, 1))     # derivative feedback
            
            # convert the external force that the end effector is applying to joint torques
            external_torque = np.matmul(J.T, external_force)
            
            # if the y position goes beyond the left hand limit, change direction
            if y > y_left_stop or t > 5:
                direction = 'right'
                print 'Wiping right'
                t = 0
                n_wipes += 1
        
        # if the number of wipes goes beyond 2, stop the motion
        if n_wipes > 2:
            break
        
        # apply the torques to the arm
        right_arm.apply_torque(external_torque)

        # sleep
        rate.sleep()
    
    # motion finished - move both arms out of the way
    right_arm.servo_to_joint_position(q_right)
    left_arm.servo_to_joint_position(q_left)  
    
    
if __name__ == "__main__":
    """Torque control"""
    rospy.init_node("torque_control")
    print 'Initialisation...'
    try:
        cmd = 'rosrun baxter_examples xdisplay_image.py -f /home/de3robotics/Desktop/DE3Robotics/src/coursework_3/src/images/eyes.jpg'
        os.system (cmd)
        cmd = 'rosrun baxter_tools tuck_arms.py -u'
        os.system (cmd)
        cmd = 'rosrun baxter_tools tuck_arms.py -u'
        os.system(cmd)
        cmd = 'rosrun baxter_tools tuck_arms.py -u'
        os.system(cmd)
    except:
        print 'Initialitation failed...'
        print 'Terminating...'
        sys.exit()
    print 'Starting...'
    main()
    
