#!/usr/bin/python
import rospy 
import rospkg 
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import sys


def cmd_vel_2_twist(v_forward, omega):
    twist_msg = Twist()
    twist_msg.linear.x = v_forward
    twist_msg.linear.y = 0
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = omega
    return twist_msg


def main():
    rospy.init_node('set_pose')
    
    vref_topic_name = "/robot/diff_drive/command"
    #rostopic pub /robot/diff_drive/command geometry_msgs/Twist -r 10 -- '[0.0, 0.0, 0.0]' '[0.0, 0.0, -0.0]'
    pub = rospy.Publisher(vref_topic_name, Twist, queue_size=1)
    
    command_msg = cmd_vel_2_twist(0.0, 0.0)
    print(command_msg)

    state_msg = ModelState()
    state_msg.model_name = 'deniro'
    state_msg.pose.position.x = 0
    state_msg.pose.position.y = -6
    state_msg.pose.position.z = 0.75

    state_msg.pose.orientation.x = 0.0
    state_msg.pose.orientation.y = 0.0
    state_msg.pose.orientation.z = 0.70710808  
    state_msg.pose.orientation.w = 0.70710548

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        
    rate = rospy.Rate(10)
    for i in range(5):
        pub.publish(command_msg)
        rate.sleep()
        

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
