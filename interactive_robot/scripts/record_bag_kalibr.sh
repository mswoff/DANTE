#!/bin/bash

rosbag record --duration=90s /tf /clock /camera/color/image_raw /camera/color/camera_info /rgb/image_raw /rgb/camera_info
