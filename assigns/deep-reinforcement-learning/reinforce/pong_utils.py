from parallelEnv import parallelEnv
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import random as rand

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


RIGHT=4
LEFT=5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
  """
	crop image and downsample to 80x80
  stack two frames together as input
	"""
  img = np.mean(image[34:-16:2, ::2]-bkg_color, axis=-1)/255.
  return img

def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
  """
	convert outputs of parallelEnv to inputs to pytorch neural net
	this is useful for batch processing especially on the GPU
	"""
  list_of_images = np.asarray(images)
  if len(list_of_images.shape) < 5:
    list_of_images = np.expand_dims(list_of_images, 1)
  #subtract bkg and crop
  list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color, axis=-1) / 255.
  batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
  return torch.from_numpy(batch_input).float().to(device)


# function to animate a list of frames
def animate_frames(frames):
  plt.axis('off')
  # color option for plotting
  # use Greys for greyscale
  cmap = None if len(frames[0].shape)==3 else 'Greys'
  patch = plt.imshow(frames[0], cmap=cmap)  
  fanim = animation.FuncAnimation(plt.gcf(), \
      lambda x: patch.set_data(frames[x]), frames = len(frames), interval=30)
  display(display_animation(fanim, default_mode='once'))


# play a game and display the animation
# nrand = number of random steps before using the policy
def play(env, policy, time=2000, preprocess=None, nrand=5):
  env.reset()
  # star game
  env.step(1)
  # perform nrand random steps in the beginning
  for _ in range(nrand):
    frame1, reward1, is_done, _ = env.step(np.random.choice([RIGHT,LEFT]))
    frame2, reward2, is_done, _ = env.step(0)
  anim_frames = []
  for _ in range(time):
    frame_input = preprocess_batch([frame1, frame2])
    prob = policy(frame_input)
    # RIGHT = 4, LEFT = 5
    action = RIGHT if rand.random() < prob else LEFT
    frame1, _, is_done, _ = env.step(action)
    frame2, _, is_done, _ = env.step(0)

    if preprocess is None:
      anim_frames.append(frame1)
    else:
      anim_frames.append(preprocess(frame1))

    if is_done:
      break
  env.close()
  animate_frames(anim_frames)
  return


def collect_trajectories(envs, policy, tmax=200, nrand=5):

  # number of parallel instances
  n=len(envs.ps)

  #initialize returning lists and start the game!
  state_list=[]
  reward_list=[]
  prob_list=[]
  action_list=[]

  envs.reset()

  # start all parallel agents
  envs.step([1]*n)

  # perform nrand random steps
  for _ in range(nrand):
    fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT],n))
    fr2, re2, _, _ = envs.step([0]*n)

  for t in range(tmax):
    # prepare the input
    # preprocess_batch properly converts two frames into
    # shape (n, 2, 80, 80), the proper input for the policy
    # this is required when building CNN with pytorch
    batch_input = preprocess_batch([fr1,fr2])

    # probs will only be used as the pi_old
    # no gradient propagation is needed
    # so we move it to the cpu
    probs = policy(batch_input).squeeze().cpu().detach().numpy()

    action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
    probs = np.where(action==RIGHT, probs, 1.0-probs)


    # advance the game (0=no action)
    # we take one action and skip game forward
    fr1, re1, is_done, _ = envs.step(action)
    fr2, re2, is_done, _ = envs.step([0]*n)

    reward = re1 + re2

    # store the result
    state_list.append(batch_input)
    reward_list.append(reward)
    prob_list.append(probs)
    action_list.append(action)

    # stop if any of the trajectories is done
    # we want all the lists to be retangular
    if is_done.any():
      break


  # return pi_theta, states, actions, rewards, probability
  return prob_list, state_list, action_list, reward_list


def states_to_prob(policy, states):
  """
  convert states to probability, passing through the policy
  """
  states = torch.stack(states)
  policy_input = states.view(-1, *states.shape[-3:])
  return policy(policy_input).view(states.shape[:-3])
