import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread
import unitree_legged_const as go2

crc = CRC()

"""
NOTE: Make sure to run this script only if the robot is perched up on its stand.
"""
if __name__ == '__main__':

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0]=0xFE
    cmd.head[1]=0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q= go2.PosStopF
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].dq = go2.VelStopF
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

    while True:       
        #################
        #TORQUE COMMANDS#
        ################# 
        # Toque controle, set RL_2 toque
        # cmd.motor_cmd[go2.LegID["FL_2"]].q = 0.0 # Set to stop position(rad)
        # cmd.motor_cmd[go2.LegID["FL_2"]].kp = 0.0
        # cmd.motor_cmd[go2.LegID["FL_2"]].dq = 0.0 # Set to stop angular velocity(rad/s)
        # cmd.motor_cmd[go2.LegID["FL_2"]].kd = 0.0
        # cmd.motor_cmd[go2.LegID["FL_2"]].tau = 1.0 # target toque is set to 1N.m

        # # Toque controle, set RL_2 toque
        # cmd.motor_cmd[go2.LegID["FR_2"]].q = 0.0 # Set to stop position(rad)
        # cmd.motor_cmd[go2.LegID["FR_2"]].kp = 0.0
        # cmd.motor_cmd[go2.LegID["FR_2"]].dq = 0.0 # Set to stop angular velocity(rad/s)
        # cmd.motor_cmd[go2.LegID["FR_2"]].kd = 0.0
        # cmd.motor_cmd[go2.LegID["FR_2"]].tau = 1.0 # target toque is set to 1N.m

        # # Toque controle, set RL_2 toque
        # cmd.motor_cmd[go2.LegID["RL_2"]].q = 0.0 # Set to stop position(rad)
        # cmd.motor_cmd[go2.LegID["RL_2"]].kp = 0.0
        # cmd.motor_cmd[go2.LegID["RL_2"]].dq = 0.0 # Set to stop angular velocity(rad/s)
        # cmd.motor_cmd[go2.LegID["RL_2"]].kd = 0.0
        # cmd.motor_cmd[go2.LegID["RL_2"]].tau = 1.0 # target toque is set to 1N.m

        # # Toque controle, set RL_2 toque
        # cmd.motor_cmd[go2.LegID["RR_2"]].q = 0.0 # Set to stop position(rad)
        # cmd.motor_cmd[go2.LegID["RR_2"]].kp = 0.0
        # cmd.motor_cmd[go2.LegID["RR_2"]].dq = 0.0 # Set to stop angular velocity(rad/s)
        # cmd.motor_cmd[go2.LegID["RR_2"]].kd = 0.0
        # cmd.motor_cmd[go2.LegID["RR_2"]].tau = 1.0 # target toque is set to 1N.m


        # ###################
        # #POSITION COMMANDS#
        # ###################

        # Poinstion(rad) control, set FL_0 rad
        cmd.motor_cmd[go2.LegID["FL_0"]].q = 0.1  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["FL_0"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["FL_0"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["FL_0"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["FL_0"]].tau = 0.0 # Feedforward toque 1N.m

        # Position(rad) control, set FL_1 rad
        cmd.motor_cmd[go2.LegID["FL_1"]].q = 0.8  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["FL_1"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["FL_1"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["FL_1"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["FL_1"]].tau = 0.0 # Feedforward toque 1N.m

        # Position(rad) control, set FL_2 rad
        cmd.motor_cmd[go2.LegID["FL_2"]].q = -1.3  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["FL_2"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["FL_2"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["FL_2"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["FL_2"]].tau = 0.0 # Feedforward toque 1N.m



        # Position(rad) control, set FR_0 rad
        cmd.motor_cmd[go2.LegID["FR_0"]].q = -0.1  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["FR_0"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["FR_0"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["FR_0"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["FR_0"]].tau = 0.0 # Feedforward toque 1N.m

        # Position(rad) control, set FR_1 rad
        cmd.motor_cmd[go2.LegID["FR_1"]].q = 0.8  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["FR_1"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["FR_1"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["FR_1"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["FR_1"]].tau = 0.0 # Feedforward toque 1N.m

        # Position(rad) control, set FR_2 rad
        cmd.motor_cmd[go2.LegID["FR_2"]].q = -1.3  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["FR_2"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["FR_2"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["FR_2"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["FR_2"]].tau = 0.0 # Feedforward toque 1N.m



        # Position(rad) control, set RL_0 rad
        cmd.motor_cmd[go2.LegID["RL_0"]].q = 0.1  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["RL_0"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["RL_0"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["RL_0"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["RL_0"]].tau = 0.0 # Feedforward toque 1N.m

        # Position(rad) control, set RL_1 rad
        cmd.motor_cmd[go2.LegID["RL_1"]].q = 1.0  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["RL_1"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["RL_1"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["RL_1"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["RL_1"]].tau = 0.0 # Feedforward toque 1N.m

        # Position(rad) control, set RL_2 rad
        cmd.motor_cmd[go2.LegID["RL_2"]].q = -1.3  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["RL_2"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["RL_2"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["RL_2"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["RL_2"]].tau = 0.0 # Feedforward toque 1N.m



        # Position(rad) control, set RR_0 rad
        cmd.motor_cmd[go2.LegID["RR_0"]].q = -0.1  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["RR_0"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["RR_0"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["RR_0"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["RR_0"]].tau = 0.0 # Feedforward toque 1N.m

        # Position(rad) control, set RR_1 rad
        cmd.motor_cmd[go2.LegID["RR_1"]].q = 1.0  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["RR_1"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["RR_1"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["RR_1"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["RR_1"]].tau = 0.0 # Feedforward toque 1N.m
        
        # Position(rad) control, set RR_2 rad
        cmd.motor_cmd[go2.LegID["RR_2"]].q = -1.3  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["RR_2"]].kp = 40.0 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["RR_2"]].dq = 0.0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["RR_2"]].kd = 5.0  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["RR_2"]].tau = 0.0 # Feedforward toque 1N.m


        cmd.crc = crc.Crc(cmd)

        #Publish message
        if pub.Write(cmd):
            print("Publish success. msg:", cmd.crc)
        else:
            print("Waitting for subscriber.")

        time.sleep(0.002)