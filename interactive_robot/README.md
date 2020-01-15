# F-formation ROS

A ros wrapper (and tools) for deep fformation detection

## Running the node

3 steps are required:

1. start the fformation nodes:

    roslaunch fformation_ros fformation.launch

2. make sure the robot power is plugged in, then start the robot motors (and stand back while this may cause the robot to jerk into the commanded position)

    roslaunch shutter_photography motors.launch

3. start the robot photographer

    roslaunch shutter_photography shutter_photography.launch

## Setup

Make a virtual env named `.venv` after you `cd` into the project directory

    virtualenv -p $(which python2.7) .venv

Activate the virtual environment

    source .venv/bin/activate

Install fformation python requirements

    pip install -r fformation_ros/modules/fformation_detection/cleaned/requirements.txt

Install fformation_ros python requirements

    pip install -r fformation_ros/requirements.txt

Install tensorflow-gpu 1.14 requirements (if you have a cuda enabled gpu)

Run the node in development mode:

    tmuxinator start .tmuxinator.yml

## Extrinsic Camera Calibration

Note, this only sort of works. e.g. I got it to work once and the output was not great (e.g.  totally off).

I manually calibrated the current setup. We should refine it at some point via an automated method, like Kalibr

- Use [Kalibr](https://github.com/ethz-asl/kalibr)

Collect a 90s bag of the [Aprilgrid 6x6 0.8x0.8 (A0 Paper)][https://drive.google.com/file/d/0B0T1sizOvRsUdjFJem9mQXdiMTQ/edit?usp=sharing] marker being moved over the view of all cameras. Use `scripts/record_bag_kalibr.sh` to record this bag.

Then run `scripts/kalibr.sh` which will produce the `camchain.yaml` file, loaded by `publish_static_transforms.py`
