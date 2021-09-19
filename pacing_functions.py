"""sigmoid and s_curve are newly implemented, 
the others credit to https://github.com/Guzpenha/transformers_cl"""
import math
import numpy as np

# x = global_step, t = curriculum_iterations 0.90, c0 = 0.33
def linear(x, t, c0):
	return (x* ((1-c0)/t)) + c0

def root_2(x, t, c0):
	return ((x* ((1-(c0**2.0))/t)) + (c0**2.0))**(1./2)

def root_5(x, t, c0):
	return ((x* ((1-(c0**5.0))/t)) + (c0**5.0))**(1./5)

def root_10(x, t, c0):
	return ((x* ((1-(c0**10.0))/t)) + (c0**10.0))**(1./10)

def geom_progression(x, t, c0):
	return 2.0**((x* ((math.log(1,2.0)-math.log(c0,2.0))/t)) +math.log(c0,2.0))

def step(x, t, c0):
	if x <= t*0.33:
		return 0.33 
	elif x> t*0.33 and x<= t*0.66:
		return 0.66 
	else:
		return 1

def standard_training(x, t, c0):
	return 1

# new implemented pacing functions sigmoid and s_curve    
def sigmoid(x, t, c0):
  k = 10
  return 1/(1 + np.exp(-k * (x/t) + math.log(2)))

def s_curve(x, t, c0):
  beta = 3
  if x/t == 0:
    return c0
  elif x/t < 1: 
    y = ((x/t)/(1 - x/t))**-beta
    return (1-c0)/(y + 1) + c0
  else: return 1


PACING_FUNCTIONS = {
	'linear': linear,
	'root_2': root_2,
	'root_5': root_5,
	'root_10': root_10,
	'geom_progression': geom_progression,
	'step': step,
	'standard_training': standard_training,
	's_curve': s_curve,
	'sigmoid': sigmoid
}
