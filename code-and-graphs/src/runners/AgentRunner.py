import numpy as np
import time

from sympy.codegen import Print


class AgentRunner:
    def __init__(self, model, env):
        self.model = model
        self.env = env

    def run(self, render=False):
        obs, _ = self.env.reset()
        step_count = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if step_count % 100 == 0:
                print(step_count)
            action, _ = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

            if render:
                self.env.render(reward=reward)
                time.sleep(1)


            if terminated or truncated:
                print(f"Environment {'terminated' if terminated else 'truncated'}.")
                break

            step_count += 1

        # Print final total force