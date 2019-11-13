import numpy as np
import robosuite as suite
from sawyer_lift import SawyerLift

env = SawyerLift(
    has_renderer=True,          # no on-screen renderer
    has_offscreen_renderer=True, # off-screen renderer is required for camera observations
    ignore_done=True,            # (optional) never terminates episode
    use_camera_obs=True,         # use camera observations
    camera_height=480,            # set camera height
    camera_width=640,             # set camera width
    camera_name='agentview',     # use "agentview" camera
    use_object_obs=True,        # no object feature when training on pixels
    camera_depth=True,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
    # print(env._get_observation())