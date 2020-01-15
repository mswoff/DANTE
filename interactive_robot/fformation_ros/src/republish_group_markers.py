#!/usr/bin/env python
import sys, os

import random
import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from fformation_ros.msg import TrackedPersonsMarkers, Groups, Group

class GroupMarkerRepublisher():
    def __init__(self):
        # Init the node
        rospy.init_node('group_marker_republisher')

        self.colors = {}

        # Subscriber
        self.marker_sub = rospy.Subscriber("/people_markers_viz", MarkerArray, self.marker_callback, queue_size=5)
        self.group_sub = rospy.Subscriber("/groups", Groups, self.group_callback, queue_size=5)

        # Publisher
        self.group_marker_pub = rospy.Publisher("/people_marker_groups_viz", MarkerArray, queue_size=5)

        # run the node in perpetuity
        rospy.spin()


    def group_callback(self, groups):
        for group in groups.groups:
            seed = np.sum(np.array(sorted(group.group)))
            random.seed(seed)
            color = [random.random(), random.random(), random.random()]
            #print(group.group, seed, color)
            for idx in group.group:
                self.colors[idx] = color
            print("colors by person idx: {}".format(self.colors))

    def marker_callback(self, markers):
        indexed_markers = {}
        for marker in markers.markers:
            idx = int(marker.id / 100)
            print("{} :IDX: {}".format(marker.id, idx))
            if idx in self.colors:
                marker.color.r = self.colors[idx][0]
                marker.color.g = self.colors[idx][1]
                marker.color.b = self.colors[idx][2]
                marker.color.a = 1

        self.group_marker_pub.publish(markers)


if __name__ == '__main__':
    try:
        group_marker_republisher = GroupMarkerRepublisher()
    except rospy.ROSInterruptException() as e:
        pass
