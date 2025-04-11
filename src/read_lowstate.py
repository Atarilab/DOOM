# Simple DDS Read Subscriber

import sys
import time

import unitree_legged_const as go2
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


def LowStateHandler(msg: LowState_):

    # print front right hip motor states
    print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]].q)
    # print("IMU state: ", msg.imu_state)
    # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)


if __name__ == "__main__":
    # Robot Network Interface Initialization
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(LowStateHandler, 10)

    while True:
        time.sleep(10.0)
