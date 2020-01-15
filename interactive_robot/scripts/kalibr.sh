#!/bin/bash

kalibr_calibrate_cameras --bag $1 --topics /rgb/image_raw /camera/color/image_raw --models pinhole-equi pinhole-equi --target ~/shutter_ws/databags/april_6x6_80x80cm.yaml
