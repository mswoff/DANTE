# setup keys
apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# setup sources.list
echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

set -x
set -e

apt update

# install bootstrap tools
apt-get install --no-install-recommends -y \
    python-catkin-tools \
    python-rosinstall-generator \
    python-rosdep \
    python-rosinstall \
    python-vcstools

# bootstrap rosdep
rosdep init \
    && rosdep update

# install ros packages
ROS_DISTRO=melodic
apt-get install -y \
    ros-${ROS_DISTRO}-ros-core \
    ros-${ROS_DISTRO}-ros-base

# robot
apt-get install -y \
    ros-${ROS_DISTRO}-robot

# desktop full
apt-get install -y \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-desktop-full

# workspace specific dependencies
apt-get install -y \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libblas-dev \
    libgfortran-4.8-dev \
    libz-dev \
    libpcap-dev \
    libsvm-dev \
    libcanberra-gtk-module \
    libasound-dev \
    pcl-tools \
    python-webcolors \
    python-pygame \
    python-pip \
    python-pyaudio \
    python3-pip \
    python3-setuptools \
    python3.7-venv \
    xsltproc \
    ros-${ROS_DISTRO}-openni2-* \
    ros-${ROS_DISTRO}-usb-cam \
    ros-${ROS_DISTRO}-rosbash \
    ros-${ROS_DISTRO}-tf2-sensor-msgs \
    ros-${ROS_DISTRO}-rosbridge-suite \
    ros-${ROS_DISTRO}-web-video-server \
    ros-${ROS_DISTRO}-vision-opencv \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-joint-state-controller \
    ros-${ROS_DISTRO}-pcl-* \
    ros-${ROS_DISTRO}-video-stream-opencv \
    ros-${ROS_DISTRO}-rgbd-launch \
    ros-${ROS_DISTRO}-openni2-camera \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-rqt-common-plugins \
    ros-${ROS_DISTRO}-rqt-robot-plugins \
    ros-${ROS_DISTRO}-opencv-apps \
    ros-${ROS_DISTRO}-joy

# other tools
apt-get install -y --no-install-recommends \
    apt-utils \
    wget \
    sudo \
    vim \
    xauth \
    iproute2 \
    net-tools

# openface 2 ros wrapper
apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev \
    libswscale-dev python-dev python-numpy libtbb2 libtbb-dev \
    libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev libv4l-dev

# img_ros_audiovisual_tools
apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 libgstreamer1.0-dev libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly \
    ros-melodic-audio-common-msgs ffmpeg

apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
echo "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" > /etc/apt/sources.list.d/realsense.list

apt-get install -y --no-install-recommends --allow-change-held-packages \
     librealsense2-dkms \
     librealsense2=2.26.0-0~realsense0.1436 \
     librealsense2-gl=2.26.0-0~realsense0.1436 \
     librealsense2-utils=2.26.0-0~realsense0.1436 \
     librealsense2-dev=2.26.0-0~realsense0.1436 \
     librealsense2-dbg=2.26.0-0~realsense0.1436

# Azure Kinect
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
apt-get update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    k4a-tools \
    libk4a1.2-dev \
    libk4abt0.9-dev

# pip install
pip install pyglet==1.3.2 playsound google-api-core google-api-python-client google-auth google-auth-httplib2 google-cloud google-cloud-core google-cloud-speech google-cloud-texttospeech google-gax googleapis-common-protos PySide2==5.12.3 numpy scipy matplotlib ipython jupyter pandas sympy nose

#LD_LIBRARY_PATH=/usr/local/lib/python2.7/dist-packages/PySide2/Qt/lib:$LD_LIBRARY_PATH

# ALSA tools
apt-get install -y --no-install-recommends \
     alsa-utils

printf "source /opt/ros/melodic/setup.bash \n\
FILE=/home/$username/$workspacename/devel/setup.bash && test -f \$FILE && source \$FILE || true\n\
export PATH=/sbin/:\$PATH \n\
cd /home/$username/$workspacename \
" >> /home/$username/.bash_profile
