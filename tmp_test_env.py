from server.app import Env, Action

env = Env()
print('issues', env.issues)
res, reward, done, _ = env.step_env(Action(action_type='FLAG_BUG', line_number=3))
print('reward', reward, 'done', done, 'found', env.found)
print('score', len(env.found), len(env.issues), len(env.found)/len(env.issues))
