#!/bin/bash

rosbag record --duration=30s /tf /tf_static /rosout /clock /camera/color/image_raw /camera/color/camera_info /rgb/image_raw /rgb/camera_info /body_tracking_data /people_markers /people_markers_viz
