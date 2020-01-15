#!/usr/bin/env python
import sys, os
fdir = os.path.dirname(os.path.abspath(__file__))
cleaned_path = os.path.join(fdir, "../modules/fformation_detection/cleaned")
fformation_syspath = os.path.join(fdir, cleaned_path)
sys.path.append(fformation_syspath)
sys.path.append(os.path.join(cleaned_path, 'datasets'))


import rospy
import itertools
from tf import transformations as tft

from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from fformation_ros.msg import TrackedPersonsMarkers, Groups, Group
from datasets.reformat_data import compute_data_shift
from datasets.build_dataset import build_X
from deep_fformation import utils, dominant_sets

import numpy as np

MARKER_TYPE = 4 # line

class GroupPredictor():
    '''ROS node that predicts the groups of people in given orientation'''

    def __init__(self):
        # Init the node
        rospy.init_node('group_predictor')

        # Node parameters
        default_model_path = os.path.join(cleaned_path, 'deep_fformation/models/cocktail_party/pair_predictions_73/val_fold_4/best_val_model.h5' )
        self.model_file = rospy.get_param('~model', default_model_path)   # default model to serach for
        self.max_people = rospy.get_param('~max_people', 6) # default to 6 people (cocktail party)
        self.cutoff = rospy.get_param('~cutoff', 0.5)  # default to 0.5 for cutoff
        self.permutations = list(itertools.permutations(range(self.max_people), 2))
        self.num_features = 4
        self.group_threshold = 1e-5
        self.load_trained_model()

        # Publisher
        self.group_pub = rospy.Publisher("/groups", Groups, queue_size=5)
        self.group_marker_pub = rospy.Publisher("/group_markers", MarkerArray, queue_size=5)

        # Subscriber
        self.marker_sub = rospy.Subscriber("/people_markers", TrackedPersonsMarkers, self.marker_callback, queue_size=5)

        # run the node in perpetuity
        rospy.spin()

    def load_trained_model(self):
        reg = 1e-07
        dropout = 0.13
        batch_norm = True
        max_people = self.max_people
        d = self.num_features
        global_filters = [16, 256]
        individual_filters = [32, 32]
        combined_filters = [1024, 256]
        self.model = utils.build_model(reg, dropout, max_people, d, global_filters,
                                       individual_filters, combined_filters, no_pointnet=False, symmetric=False)

        self.model.load_weights(self.model_file)
        self.model._make_predict_function()

    def floor_angle_from_quaternion_msg(self, quaternion_msg):
        quaternion = [quaternion_msg.x, quaternion_msg.y, quaternion_msg.z, quaternion_msg.w]
        euler = tft.euler_from_quaternion(quaternion)
        return euler[2]

    def calculate_people_list(self, people_markers_msg):
        if len(people_markers_msg.tracks) == 0:
            return []
        people_list = [str(people_markers_msg.tracks[0].head.header.stamp.to_sec())]
        for person in people_markers_msg.tracks:
            id = person.track_id
            body_pose = person.body.pose
            x = body_pose.position.x
            y = body_pose.position.y
            body_quaternion = body_pose.orientation
            body_theta = self.floor_angle_from_quaternion_msg(body_quaternion)

            head_quaternion = person.head.pose.orientation
            head_theta = self.floor_angle_from_quaternion_msg(head_quaternion)

            person_features = [id, x, y, head_theta] # , body_theta]
            people_list.extend(person_features)

        expected_data_features = self.max_people*4 + 1 # self.max_people*(self.num_features+1) + 1
        if len(people_list) < expected_data_features:
            missing_entries = expected_data_features - len(people_list)  # (expected_data_features - len(people_list)) // (self.num_features + 1)
            fake_data = ['fake'] * missing_entries
            people_list.extend(fake_data)

        if len(people_list) > expected_data_features:
            # truncate tracked people if too many in frame
            people_list = people_list[:expected_data_features]
        return people_list

    def format_pose_data(self, people_markers_msg):
        people_list = self.calculate_people_list(people_markers_msg)
        if len(people_list) == 0:
            return None, None, None
        shifted_people = compute_data_shift(np.array(people_list).reshape((1, -1)),
                        str(people_markers_msg.tracks[0].head.header.stamp.to_sec()),
                        self.max_people, augment_flipped_data=False)
        indices = [person[0] for person in shifted_people]
        people_groups, people_pairs = build_X(shifted_people, self.max_people, self.num_features)
        return indices, [people_groups, people_pairs], people_list

    def calculate_group_clusters(self, people_marker_msg, frame, group_preds):
        group_bools = dominant_sets.iterate_climb_learned(group_preds, frame, self.max_people, self.group_threshold, self.num_features)
        track_ids = [people_marker_msg.tracks[i].track_id for i in range(len(people_marker_msg.tracks))]
        groups = []
        for group in group_bools:
            group_ids = [track_ids[per_idx] for per_idx, in_group in enumerate(group) if in_group]
            groups.append(group_ids)
            found_ids = set([person_id for group in groups for person_id in group])
            for person_id in track_ids:
                if person_id not in found_ids:
                    groups.append([person_id])
            print('calculate_group_clusters:', groups)
            return groups

    def create_group_msg(self, people_marker_msg, frame, group_preds):
        header = people_marker_msg.tracks[0].body.header
        group_pred_msg = Groups()
        group_pred_msg.header = header
        group_sets = self.calculate_group_clusters(people_marker_msg, frame, group_preds)
        groups = [Group(group) for group in group_sets]
        group_pred_msg.groups = groups
        return group_pred_msg

    def create_marker_msg(self, people_markers_msg, indices, group_preds):
        split_indices = [idx.strip().split(':') for idx in indices]
        pairs = [(int(idx[1]), int(idx[2])) for idx in split_indices]
        markers = []
        x_vals = []
        for i, pair in enumerate(pairs):
            person_i = people_markers_msg.tracks[pair[0]]
            person_j = people_markers_msg.tracks[pair[1]]
            marker_id = int('{}{}'.format(int(person_i.body.id/100), int(person_j.body.id/100)))
            marker = Marker()
            marker.id = marker_id
            marker.type = MARKER_TYPE
            marker.header = person_i.body.header
            marker.lifetime = person_i.body.lifetime

            # create points to connect line
            point_i = Point()
            point_i.x = person_i.body.pose.position.x
            point_i.y = person_i.body.pose.position.y
            point_i.z = person_i.body.pose.position.z

            point_j = Point()
            point_j.x = person_j.body.pose.position.x
            point_j.y = person_j.body.pose.position.y
            point_j.z = person_j.body.pose.position.z

            marker.points = [point_i, point_j]

            x_vals.append(point_i.x)
            x_vals.append(point_j.x)
            # color lines according to group predictions
            color = ColorRGBA()
            color.r=1
            color.g=0
            color.b=0
            color.a=group_preds[i]
            # print('strength:', group_preds[i])
            marker.color = color
            marker.colors = [color, color]

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            markers.append(marker)
        # print('num markers:', len(markers))
        # print('unique:', set(x_vals))
        marker_msg = MarkerArray(markers=markers)
        return marker_msg

    def marker_callback(self, people_markers_msg):
        if len(people_markers_msg.tracks) < 1:
            return
        indices, orientation, frame = self.format_pose_data(people_markers_msg)
        if orientation[0] is not None and len(orientation[0]) > 0:
            group_preds = self.model.predict(orientation)
            #print('group preds:', group_preds)

            group_pred_msg = self.create_group_msg(people_markers_msg, frame, group_preds)
            self.group_pub.publish(group_pred_msg)
            
            try:
                marker_msg = self.create_marker_msg(people_markers_msg, indices, group_preds)
                self.group_marker_pub.publish(marker_msg)
            except Exception as e:
                print(e)
                print('marker_msg:\n', marker_msg)

if __name__ == '__main__':
    try:
        group_predictor = GroupPredictor()
    except rospy.ROSInterruptException() as e:
        pass
