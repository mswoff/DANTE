#!/usr/bin/env python

import os
import sys
fdir = os.path.dirname(os.path.abspath(__file__))

import rosbag
import rospy
import tf

import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from tf import transformations as tft
from enum import Enum
import math
from copy import deepcopy

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import MarkerArray, Marker
from fformation_ros.msg import TrackedPersonsMarkers, TrackedPersonMarkers

body_marker_topic = '/body_tracking_data'
base_link = 'base_link'

MARKER_TYPE = 0 # arrow

class BodyMakers(Enum):
      K4ABT_JOINT_PELVIS = 0; K4ABT_JOINT_SPINE_NAVAL = 1; K4ABT_JOINT_SPINE_CHEST = 2; K4ABT_JOINT_NECK = 3;
      K4ABT_JOINT_CLAVICLE_LEFT = 4; K4ABT_JOINT_SHOULDER_LEFT = 5; K4ABT_JOINT_ELBOW_LEFT = 6; K4ABT_JOINT_WRIST_LEFT = 7;
      K4ABT_JOINT_CLAVICLE_RIGHT = 8; K4ABT_JOINT_SHOULDER_RIGHT = 9; K4ABT_JOINT_ELBOW_RIGHT = 10; K4ABT_JOINT_WRIST_RIGHT = 11;
      K4ABT_JOINT_HIP_LEFT = 12; K4ABT_JOINT_KNEE_LEFT = 13; K4ABT_JOINT_ANKLE_LEFT = 14; K4ABT_JOINT_FOOT_LEFT = 15;
      K4ABT_JOINT_HIP_RIGHT = 16; K4ABT_JOINT_KNEE_RIGHT = 17; K4ABT_JOINT_ANKLE_RIGHT = 18; K4ABT_JOINT_FOOT_RIGHT = 19;
      K4ABT_JOINT_HEAD = 20; K4ABT_JOINT_NOSE = 21; K4ABT_JOINT_EYE_LEFT = 22; K4ABT_JOINT_EAR_LEFT = 23;
      K4ABT_JOINT_EYE_RIGHT = 24; K4ABT_JOINT_EAR_RIGHT = 26; K4ABT_JOINT_COUNT = 27


class MarkerRepublisher():
    """Re-publishes relevent kinect markers on their own topic"""

    def __init__(self):
        # Init the node
        rospy.init_node('body_markers')

        self.include_robot = rospy.get_param('~include_robot', True)

        # Subscriber
        self.maker_sub = rospy.Subscriber(body_marker_topic, MarkerArray, self.markers_callback, queue_size=5)

        # Publishers
        #self.tracks_p_pub = rospy.Publisher("/people_tracks", TrackedPersons, queue_size=5)
        self.track_first_person_head_pub = rospy.Publisher("/person_head_pose", PoseStamped, queue_size=5)
        self.track_first_person_body_pub = rospy.Publisher("/person_body_pose", PoseStamped, queue_size=5)
        self.tracks_m_pub = rospy.Publisher("/people_markers", TrackedPersonsMarkers, queue_size=5)
        self.tracks_m_viz_pub = rospy.Publisher("/people_markers_viz", MarkerArray, queue_size=5)

        # tf
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # main thread just waits now..
        rospy.spin()


    def lookup_transform(self, parent_link, header, time=None, duration=rospy.Duration(1.0)):
        if time is None:
            time = header.stamp
        try:
            trans = self.tf_buffer.lookup_transform(parent_link, header.frame_id, time, duration)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(e)
            return None
        return trans


    def do_transform(self, pose, parent_link, pose_header, duration=rospy.Duration(1.0)):
        trans = self.lookup_transform(parent_link, pose_header, pose_header.stamp, duration)
        if trans is None:
            return
        p = PoseStamped()
        p.pose = pose
        p.header = pose_header
        transformed_pose_stamped = do_transform_pose(p, trans)
        return transformed_pose_stamped


    def build_robot_track(self, stamp):
        track_m = TrackedPersonMarkers()

        head_frame = 'camera_color_frame'
        body_frame = base_link

        # head
        track_m.head = Marker()
        track_m.head.id = 1
        track_m.head.color.r = 0.286
        track_m.head.color.g = 0.714
        track_m.head.color.b = 0.808
        track_m.head.color.a = 1
        # viz setup
        track_m.head.type = MARKER_TYPE
        track_m.head.scale.x = 0.2
        track_m.head.scale.y = 0.05
        track_m.head.scale.z = 0.05
        track_m.head.ns = 'bodies'
        track_m.head.lifetime = rospy.Duration(1)
        # build header
        head_header = Header()
        head_header.frame_id = head_frame
        head_header.stamp = stamp
        track_m.head.header = head_header
        # transform
        pose_stamped = self.lookup_transform(base_link, head_header, rospy.Time.now())
        o = pose_stamped.transform.rotation
        a = list(tft.euler_from_quaternion([o.x, o.y, o.z, o.w]))
        # z forward
        q = tft.quaternion_from_euler(*a)
        track_m.head.header.frame_id = pose_stamped.header.frame_id
        track_m.head.pose.position.x = pose_stamped.transform.translation.x
        track_m.head.pose.position.y = pose_stamped.transform.translation.y
        track_m.head.pose.position.z = pose_stamped.transform.translation.z
        track_m.head.pose.orientation.x = q[0]
        track_m.head.pose.orientation.y = q[1]
        track_m.head.pose.orientation.z = q[2]
        track_m.head.pose.orientation.w = q[3]

        # body
        track_m.body = Marker()
        track_m.body.id = 2
        track_m.body.color.r = 0.286
        track_m.body.color.g = 0.714
        track_m.body.color.b = 0.808
        track_m.body.color.a = 1
        # viz setup
        track_m.body.type = MARKER_TYPE
        track_m.body.scale.x = 0.2
        track_m.body.scale.y = 0.05
        track_m.body.scale.z = 0.05
        track_m.body.ns = 'bodies'
        track_m.body.lifetime = rospy.Duration(1)
        # build header
        body_header = Header()
        body_header.frame_id = body_frame
        body_header.stamp = stamp
        track_m.body.header = body_header
        # transform
        if body_frame == base_link:
            track_m.body.header.frame_id = body_frame
            track_m.body.pose.position.x = 0
            track_m.body.pose.position.y = 0
            track_m.body.pose.position.z = 0
            track_m.body.pose.orientation.x = 0
            track_m.body.pose.orientation.y = 0
            track_m.body.pose.orientation.z = 0
            track_m.body.pose.orientation.w = 0
        else:
            pose_stamped = self.lookup_transform(base_link, body_header, rospy.Time.now())
            track_m.body.header.frame_id = pose_stamped.header.frame_id
            track_m.body.pose.position.x = pose_stamped.transform.translation.x
            track_m.body.pose.position.y = pose_stamped.transform.translation.y
            track_m.body.pose.position.z = pose_stamped.transform.translation.z
            track_m.body.pose.orientation.x = pose_stamped.transform.rotation.x
            track_m.body.pose.orientation.y = pose_stamped.transform.rotation.y
            track_m.body.pose.orientation.z = pose_stamped.transform.rotation.z
            track_m.body.pose.orientation.w = pose_stamped.transform.rotation.w

        return track_m



    def build_person_track(self, body_id, markers):
        track_m = TrackedPersonMarkers()
        track_m.track_id = body_id

        # head
        track_m.head = markers[BodyMakers.K4ABT_JOINT_HEAD.value]
        # viz setup
        track_m.head.type = MARKER_TYPE
        track_m.head.scale.x = 0.2
        track_m.head.scale.y = 0.05
        track_m.head.scale.z = 0.05
        track_m.head.ns = 'bodies'
        # transform
        pose_stamped = self.do_transform(track_m.head.pose, base_link, track_m.head.header)
        track_m.head.header.frame_id = pose_stamped.header.frame_id
        o = pose_stamped.pose.orientation
        a = list(tft.euler_from_quaternion([o.x, o.y, o.z, o.w]))
        # for some reason this is backward?
        a[2] = -a[2] + math.pi
        q = tft.quaternion_from_euler(*a)
        track_m.head.pose.position.x = pose_stamped.pose.position.x
        track_m.head.pose.position.y = pose_stamped.pose.position.y
        track_m.head.pose.position.z = pose_stamped.pose.position.z
        track_m.head.pose.orientation.x = q[0]
        track_m.head.pose.orientation.y = q[1]
        track_m.head.pose.orientation.z = q[2]
        track_m.head.pose.orientation.w = q[3]

        # body
        track_m.body = markers[BodyMakers.K4ABT_JOINT_PELVIS.value]
        # viz setup
        track_m.body.type = MARKER_TYPE
        track_m.body.scale.x = 0.2
        track_m.body.scale.y = 0.05
        track_m.body.scale.z = 0.05
        track_m.body.ns = 'bodies'
        # transform
        pose_stamped = self.do_transform(track_m.body.pose, base_link, track_m.body.header)
        track_m.body.header.frame_id = pose_stamped.header.frame_id
        o = pose_stamped.pose.orientation
        a = list(tft.euler_from_quaternion([o.x, o.y, o.z, o.w]))
        # for some reason this is backward?
        a[2] = -a[2] + math.pi
        q = tft.quaternion_from_euler(*a)
        track_m.body.pose.position.x = pose_stamped.pose.position.x
        track_m.body.pose.position.y = pose_stamped.pose.position.y
        track_m.body.pose.position.z = pose_stamped.pose.position.z
        track_m.body.pose.orientation.x = q[0]
        track_m.body.pose.orientation.y = q[1]
        track_m.body.pose.orientation.z = q[2]
        track_m.body.pose.orientation.w = q[3]

        return track_m


    def markers_callback(self, markers):
        '''
        per: https://github.com/microsoft/Azure_Kinect_ROS_Driver/pull/50#issuecomment-521847158
        body.id * 100 + joint index.
        i.e. body.id 42 has the Marker IDs
        4200 to 4225
        '''
        stamp = rospy.Time.now()
        if len(markers.markers) > 0:
            stamp = markers.markers[0].header.stamp

        markers_by_body = {}
        for marker in markers.markers:
            body_id = int(marker.id/100)
            marker_id = int(str(marker.id)[-2:])
            if body_id not in markers_by_body:
                markers_by_body[body_id] = {}
            markers_by_body[body_id][marker_id] = marker

        # marker tracking messages
        tracks_m = TrackedPersonsMarkers()
        viz_m = MarkerArray()

        # re-publish the main marker
        for body_id, markers in markers_by_body.items():
            track_m = self.build_person_track(body_id, markers)
            tracks_m.tracks.append(track_m)
            sphere = deepcopy(track_m.head)
            sphere.id = sphere.id + 1
            sphere.type = 2
            sphere.scale.x = 0.3
            sphere.scale.y = 0.3
            sphere.scale.z = 0.3
            viz_m.markers.append(track_m.head)
            viz_m.markers.append(sphere)
            cylinder = deepcopy(track_m.body)
            cylinder.id = cylinder.id + 1
            cylinder.type = 3
            cylinder.scale.x = 0.2
            cylinder.scale.y = 0.2
            cylinder.scale.z = 0.7
            cylinder.pose.orientation.x = 0
            cylinder.pose.orientation.y = 0
            cylinder.pose.orientation.z = 0
            cylinder.pose.orientation.w = 0
            viz_m.markers.append(track_m.body)
            viz_m.markers.append(cylinder)

        # if we want to include the robot
        if self.include_robot:
            track_m = self.build_robot_track(stamp)
            tracks_m.tracks.append(track_m)
            viz_m.markers.append(track_m.head)
            viz_m.markers.append(track_m.body)

        # publish markers
        self.tracks_m_pub.publish(tracks_m)
        self.tracks_m_viz_pub.publish(viz_m)

        # for debugging
        if tracks_m.tracks:
            head_pose = PoseStamped()
            head_pose.header = tracks_m.tracks[0].head.header
            head_pose.pose = tracks_m.tracks[0].head.pose
            self.track_first_person_head_pub.publish(head_pose)
            body_pose = PoseStamped()
            body_pose.header = tracks_m.tracks[0].body.header
            body_pose.pose = tracks_m.tracks[0].body.pose
            self.track_first_person_body_pub.publish(body_pose)


if __name__ == '__main__':
    try:
        node = MarkerRepublisher()
    except rospy.ROSInterruptException:
        pass
