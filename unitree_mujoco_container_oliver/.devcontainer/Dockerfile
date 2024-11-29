# Base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8
# install system related packages
RUN apt-get update && \
    apt-get install -y \
    cmake \
    make \
    nano \
    vim \
    curl \
    git \
    pip \
    software-properties-common \
    build-essential \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libzmq3-dev \
    '^libxcb.*-dev' \
    libx11-dev \
    libx11-xcb-dev \
    libxi-dev \
    libxrender-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libglu1-mesa-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/* \
    libglfw3-dev\
    kbd
# Create a non-root user
# Linux default main user with UID 1000
# USERNAME is not important. files created will still be accessable
ARG USERNAME=atari
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config
# Set up sudo
RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*
###
### INSTALL SYSTEM DEPENDENCIES
###
RUN sudo add-apt-repository main -y\
  && sudo apt-get update \
  && sudo apt-get install -y \
  libeigen3-dev
RUN apt-get update && apt-get install -y \
python3-pip \
git \
&& rm -rf /var/lib/apt/lists/*
# Link NVIDIA devices
ENV DEBIAN_FRONTEND=noninteractive
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
###
### INSTALL PACKAGES
###
RUN mkdir -p /home/$USERNAME/dependencies
WORKDIR /home/$USERNAME/dependencies
#
# ROS2 installation
#
# Verify locale settings
RUN apt-get update -q && \
    apt-get upgrade -yq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends keyboard-configuration language-pack-en && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends wget curl git build-essential ca-certificates tzdata tmux gnupg2 \
        vim sudo lsb-release locales bash-completion zsh iproute2 iputils-ping net-tools dnsutils && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8
RUN locale-gen en_US.UTF-8
# Enable the Ubuntu Universe repository
RUN add-apt-repository universe
RUN apt-get update && apt-get install curl gnupg lsb-release -y
# Add the ROS 2 GPG key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# Add the ROS 2 repository to the apt sources list
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
# Update apt and upgrade system before installing ROS 2
RUN apt-get update && apt-get upgrade -y
RUN apt-mark unhold libsystemd0
# Install ROS 2 (choose between the desktop or base install)
# Desktop Install (includes RViz, demos, tutorials)
ENV ROS_DISTRO=humble
RUN apt-get install -y ros-dev-tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-desktop ros-${ROS_DISTRO}-ament-cmake && \
    apt-get install -y --no-install-recommends python3-argcomplete python3-colcon-common-extensions python3-rosdep python3-colcon-mixin python3-vcstool && \
    if [ ! "$TARGETPLATFORM" = "linux/arm64" ]; then apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-gazebo-ros-pkgs; fi && \
    apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-joint-state-publisher-gui
RUN rosdep init
RUN rosdep update
# Replace ".bash" with your shell if you're not using bashP
# Possible values are: setup.bash, setup.sh, setup.zsh
# Before compiling cyclonedds, please ensure that ros2 environment has NOT been sourced
# RUN source /opt/ros/foxy/setup.bash
#
# unitree_ros2
#
RUN git clone https://github.com/Atarilab/unitree_ros2
# https://answers.ros.org/question/402781/ros-humble-ubuntu-2204-apt-install-issue/
RUN export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
RUN apt-get install -y ros-${ROS_DISTRO}-rmw-cyclonedds-cpp
RUN apt-get install -y ros-${ROS_DISTRO}-rosidl-generator-dds-idl
WORKDIR /home/$USERNAME/dependencies/unitree_ros2/cyclonedds_ws/src
RUN git clone https://github.com/ros2/rmw_cyclonedds -b ${ROS_DISTRO}
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
WORKDIR /home/$USERNAME/dependencies/unitree_ros2/cyclonedds_ws/
# Compile cyclone-dds package
RUN colcon build --packages-select cyclonedds
# source ROS2 environment
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build"
###
### Create the project directory
###
RUN mkdir -p /home/$USERNAME/workspace
RUN mkdir -p /home/$USERNAME/workspace/project/
WORKDIR /home/$USERNAME/workspace
# Unitree mujoco
RUN git clone https://github.com/unitreerobotics/unitree_mujoco.git
# Python dependencies
RUN pip install mujoco pygame torch textual
# Downgrade numpy
RUN python3 -m pip install "numpy<2.0.0"
#
# # Install Vicon DataStream SDK
#
WORKDIR /home/$USERNAME/dependencies
RUN git clone https://github.com/Atarilab/ros2-vicon-receiver.git &&\
    cd ros2-vicon-receiver &&\
    bash ./install_libs.sh
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"
# #
# # unitree_sdk2_python
# #
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
WORKDIR /home/$USERNAME/dependencies/unitree_sdk2_python
RUN pip3 install -e .
# Good practice
USER $USERNAME
# Ensure proper permissions for the DOOM workspace
WORKDIR /home/$USERNAME/workspace/DOOM
RUN sudo chown -R $USERNAME:$USERNAME /home/$USERNAME/workspace/DOOM
RUN sudo chown -R $USERNAME:$USERNAME /home/$USERNAME/workspace/unitree_mujoco
RUN cp /home/atari/dependencies/unitree_ros2/setup.sh /home/$USERNAME/workspace/DOOM/
RUN cp /home/atari/dependencies/unitree_ros2/setup_local.sh /home/$USERNAME/workspace/DOOM/
USER root
# Source ROS 2 Environments
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc \
    echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc
RUN echo "source /home/$USERNAME/dependencies/install/setup.bash" >> /home/$USERNAME/.bashrc \
    echo "source /home/$USERNAME/dependencies/install/setup.bash" >> /root/.bashrc
WORKDIR /home/$USERNAME/workspace/DOOM/
ENV PYTHONPATH=/home/$USERNAME/workspace/DOOM/src:$PYTHONPATH
# Set the working directory and source the bashrc file
RUN bash /root/.bashrc
RUN cat /root/.bashrc >> /home/$USERNAME/.bashrc
CMD ["bash"]