#!/bin/bash

rosbag record --duration=30s /tf /tf_static /rosout /clock /camera/color/image_raw /camera/color/camera_info /rgb/image_raw /rgb/camera_info /usb_cam/image_raw_throttle /usb_cam/camera_info /body_tracking_data /body_index_map/image_raw
