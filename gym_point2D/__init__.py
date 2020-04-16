from gym.envs.registration import register

register(id='Point2D-v0',entry_point='gym_point2D.envs:Point2DEnv',)
register(id='Point2DSimple-v0',entry_point='gym_point2D.envs:Point2DSimpleEnv',)
