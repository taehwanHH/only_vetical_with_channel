import torch
import gym
import time
import glfw
import mujoco_py
from scipy.spatial.transform import Rotation as R
from param import Hyper_Param
from param_robot import Robot_Param


DEVICE = Hyper_Param['DEVICE']

Sensing_interval = Robot_Param['Sensing_interval']
End_flag = Robot_Param['End_flag']
Act_max = Robot_Param['Act_max']
Max_time= Robot_Param['Max_time']
State_normalizer = Robot_Param['State_normalizer']
h_threshold = Robot_Param['h_threshold']
rot_threshold = Robot_Param['rot_threshold']


# mujoco-py
xml_path = "sim_env.xml"
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

viewer.cam.azimuth = 180
viewer.cam.elevation = -5
viewer.cam.distance = 5

# Get the viewer's window handle
window = viewer.window

# Ensure the window is not in fullscreen mode by setting monitor to None
glfw.set_window_monitor(window, None, 100, 100, 1280, 720, glfw.DONT_CARE)
# Optionally, you can set the window title
glfw.set_window_title(window, "MuJoCo Simulation")



def sensor_read(num_robot, num_sensor):
    touch_vector = []
    for i in range(num_robot):
        for j in range(num_sensor):
            sensor_idx = sim.model.sensor_name2id(f"robot{i+1}_sensor{j + 1}")
            touch_vector.append(sim.data.sensordata[sensor_idx]/State_normalizer)

    return torch.tensor(touch_vector, dtype=torch.float32).view(1,-1).to(DEVICE)


def actuator_write(num_robot, action):
    for i in range(num_robot):
        actuator_2_idx = sim.model.actuator_name2id(f"{i+1}_actuator_joint2")
        actuator_3_idx = sim.model.actuator_name2id(f"{i+1}_actuator_joint3")

        sim.data.ctrl[actuator_2_idx] = action[2*i]
        sim.data.ctrl[actuator_3_idx] = action[2*i+1]


def box_state():
    # Get box orientation quaternion
    box_idx = sim.model.body_name2id("box")
    object_quat = sim.data.body_xquat[box_idx]
    # Convert quaternion to Euler angles
    object_euler = torch.tensor(R.from_quat(object_quat).as_euler('xyz', degrees=True), device=DEVICE,
                                dtype=torch.float32)

    # Get box position
    box_pos = sim.data.body_xpos[box_idx]
    box_z_pos = torch.tensor(box_pos[2], device=DEVICE, dtype=torch.float32)
    return object_euler, box_z_pos


class RoboticEnv:
    def __init__(self, Max_time=Max_time):
        # Define the state space and action space
        self.num_sensor_output = 5  # pressure sensor output
        self.num_robot = 4
        self.num_joint = 2
        self.state_dim = self.num_sensor_output * self.num_robot
        self.action_dim = self.num_robot * self.num_joint
        # self.state_space = gym.spaces.Box(low=0, high=1500, shape=(self.state_dim,))
        self.action_space = gym.spaces.Box(low=0, high=Act_max, shape=(self.action_dim,))

        # initialize
        self.Max_time = Max_time
        self.time_step = 0
        self.stable_time = 0
        self.state = torch.tensor([0]*self.state_dim).view(1,-1)
        self.reward = torch.tensor([0]).view(1,-1)
        self.done = False
        self.flag = 0
        self.z_pos = 0
        self.task_success = 0

    def step(self, action):
        self.time_step += 1

        actuator_write(self.num_robot, action)
        for _ in range(Sensing_interval):
            sim.step()
            viewer.render()


        next_state = sensor_read(self.num_robot, self.num_sensor_output)

        object_euler, self.z_pos = box_state()

        rotation_val = torch.square(object_euler[1]) + torch.square(object_euler[2])

        if rotation_val < rot_threshold:
            self.stable_time += 1
            if self.z_pos > h_threshold:
                self.task_success += 1

        reward =  rotation_val - torch.square(self.z_pos*10)
        reward = - reward.to(DEVICE)



        if torch.sum(next_state) == 0:
            self.flag += 1
            reward -= 1000  # Penalty term
        else:
            self.flag = 0

        if self.time_step > self.Max_time or self.flag == End_flag:
            self.done = True

        return next_state, reward, self.done, {}

    def reset(self):
        self.time_step = 0
        self.stable_time = 0
        self.task_success = 0
        self.done = False

        sim.reset()

        rand_ctrl = torch.rand(self.action_dim)*Act_max
        actuator_write(self.num_robot, rand_ctrl)
        for _ in range(Sensing_interval*5):
            sim.step()
            viewer.render()

        state = sensor_read(self.num_robot, self.num_sensor_output)

        return state














