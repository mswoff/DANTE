#!/usr/bin/env python
import sys, os
fdir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(fdir, "../config")

import yaml
import rospy
import sys
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf import transformations as tft


## rosrun tf tf_echo /base_link /camera_color_optical_frame
#- Translation: [0.031, 0.032, 0.332]
#- Rotation: in Quaternion [-0.470, 0.471, -0.528, 0.527]
#            in RPY (radian) [-1.457, 0.000, -1.573]
#            in RPY (degree) [-83.496, 0.001, -90.099]

class PublishStaticTransforms(object):

    def __init__(self, run_node=True):

        rospy.init_node("publish_static_transforms", anonymous=True)

        # params
        camchain_file = rospy.get_param("~camchain", os.path.join(config_dir, 'camchain-2019-10-04-19-53-09.yaml'))
        #self.kinect_optical_frame = rospy.get_param("~kinect_optical_frame", "")

        with open(camchain_file) as f:
            self.camchain = yaml.load(f)

        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        transforms = []
        #transforms.extend(self.shutter_transform())
        transforms.append(self.camchain_transform())

        if run_node:
            rate = rospy.Rate(1)
            while not rospy.is_shutdown():
                self.broadcaster.sendTransform(transforms)
                rate.sleep()

    def camchain_transform(self):
        st = TransformStamped()
        st.header.stamp = rospy.Time.now()
        st.header.frame_id = 'base_link'
        st.child_frame_id = 'camera_base'
        # TODO: add this in?
## rosrun tf tf_echo /camera_color_optical_frame /base_link
#- Translation: [0.033, 0.327, -0.068]
#- Rotation: in Quaternion [0.470, -0.471, 0.528, 0.527]
#            in RPY (radian) [-0.015, -1.456, 1.586]
#            in RPY (degree) [-0.862, -83.407, 90.867]

        m = np.array([
            [0.999093164991004, -0.02774805942482276, 0.03229385183603588, 0.06266763945863524],
            [0.011222408503216206, 0.9032690505322983, 0.4289278259775881, -0.5165454679903098],
            [-0.04107195169015314, -0.42817644441121433, 0.9027613345927731, -0.11074038324516788],
            [0.0, 0.0, 0.0, 1.0]
        ])
        a = list(tft.euler_from_matrix(m))
        print("angle adjustment")
        print(a)
        a[0] = -0.02
        a[1] = 0.3
        a[2] = 0
        print(a)
        q = tft.quaternion_from_euler(*a)

        print("transformation")
        t = m[:, -1]
        print(t)
        t[0] = 0.05
        t[1] = -0.001
        t[2] = 0.88
        print(t)

        st.transform.translation.x = t[0]
        st.transform.translation.y = t[1]
        st.transform.translation.z = t[2]

        st.transform.rotation.x = q[0]
        st.transform.rotation.y = q[1]
        st.transform.rotation.z = q[2]
        st.transform.rotation.w = q[3]
        return st

if __name__ == "__main__":
    try:
        node = PublishStaticTransforms()
    except rospy.ROSInterruptException:
        pass
