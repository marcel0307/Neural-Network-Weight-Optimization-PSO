
# First we implement a Neural Network using numpy (Forward Propagation only)
import numpy as np

def sigmoid(Z):
    """
    Sigmoid activation Function to use in the last layer of the network
    """
    A = 1/(1+np.exp(-Z))

    return A

def relu(Z):
    """
    Relu Activation function to introduce non-linearity to the network
    """
    A = np.maximum(0,Z)  

    return A


def initialize_parameters_deep(layer_dimension):
    """
    -Random initialization of weights and biases
    -Use a dictionary data structure to easilyy access the correct weights and biases later on

    """
    
    parameters = {}

    L = len(layer_dimension)
 
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dimension[l], layer_dimension[l-1]) 
        parameters["b" + str(l)] = np.random.randn(layer_dimension[l], 1) 

    return parameters

def linear_forward(A, W, b):

    """
    one step of forward propagation
    """
    Z = np.dot(W,A)+b

    
    return Z

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
              
        Z = linear_forward(A_prev,W,b)
        A = sigmoid(Z)      
    
    elif activation == "relu":
         
        Z = linear_forward(A_prev,W,b)
        A = relu(Z)
        

    return A

def L_model_forward(X, parameters,threshold=0.5):

    """
    entire forward propagation
    """

    A = X

    # number of layers
    L = len(parameters) // 2
    
    # Using a for loop to replicate [LINEAR->RELU] (L-1) times
    for l in range(1, L):
        A_prev = A 

        # Implementation of LINEAR -> RELU.
        A = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")


    # Implementation of LINEAR -> SIGMOID.
    AL = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")

    #Apply thresholding.
    AL = np.where(AL > threshold, 1 , 0)
            
    return AL

def BinaryCrossEntropy(y_true, y_pred):

    """
    Calculate Binary Cross Entropy as a Fitness function
    """
    
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    
    return -np.mean(term_0+term_1, axis=0)

def accuracy(y_true, y_pred):

    """
    Calculate Accuracy as a Fitness function --> suitable only for balanced dataset
    """

    #Ratio of different elements over number of predictions
    return np.count_nonzero((y_pred + y_true) != 1) / y_pred.size

def compute_fitness(X,y,parameters_flat,param_initial,metric="accuracy"):

    parameters = reconstruct_param(parameters_flat,param_initial)
    pred = L_model_forward(X, parameters,threshold=0.5)

    if metric == "accuracy":
        fitness = accuracy(y,pred)
        
    elif metric == "CrossEntropy":
        fitness = BinaryCrossEntropy(pred.reshape(-1, 1), y.reshape(-1, 1))

    return fitness


#These two functions will be used between the Neural Network and PSO Code
def flatten_param(param_dict):
    """
    Convert parameter dictionary into a 1-dimensional array to feed into the PSO Algo
    """
    return np.array([item for sublist in list(param_dict.values()) for item in sublist.ravel()])


def reconstruct_param(parameters_flat,parameters_structure):
    """
    Covert new solution from the PSO back into the dictionary structure for the forward propagation

    -Inverse operation to flatten_param function
    """
    dicti = dict()
    start = 0
    for key,value in parameters_structure.items():
        end = start + value.size
        dicti[key] = parameters_flat[start:end].reshape(value.shape)
        start = end
    return dicti



import random
import math    
import copy    
import sys    
 

 
#-------------------------
#Implementation of Particle Swarm Optimization: The core structure of the code is inspired by https://www.geeksforgeeks.org/implementation-of-particle-swarm-optimization/. However, it was heavily adpated to our use case
 
#Particle class
class Particle:
  def __init__(self, fitness, dim, minx, maxx, seed):
    self.rnd = random.Random(seed)
 
    # initialize position of the particle with 0.0 value
    self.position = [0.0 for i in range(dim)]
 
     # initialize velocity of the particle with 0.0 value
    self.velocity = [0.0 for i in range(dim)]
 
    # initialize best particle position of the particle with 0.0 value
    self.best_part_pos = [0.0 for i in range(dim)]
 
    # loop dim times to calculate random position and velocity
    # range of position and velocity is [minx, max]
    for i in range(dim):
      self.position[i] = ((maxx - minx) *
        self.rnd.random() + minx)
      self.velocity[i] = ((maxx - minx) *
        self.rnd.random() + minx)
 
    # compute fitness of particle
    self.fitness = fitness(self.position) # curr fitness
 
    # initialize best position and fitness of this particle
    self.best_part_pos = copy.copy(self.position)
    self.best_part_fitnessVal = self.fitness # best fitness
 
# particle swarm optimization function
def pso(fitness, max_iter, n, dim, minx, maxx,w,c1,c2,seed):
 
  rnd = random.Random(0)

  R = np.random.RandomState(seed) #Implement a random_state that can be specified by the user
  # create n random particles
  swarm = [Particle(fitness, dim, minx, maxx, i) for i in R.rand(n)]
 
  # compute the value of best_position and best_fitness in swarm
  best_swarm_pos = [0.0 for i in range(dim)]
  best_swarm_fitnessVal = sys.float_info.max # swarm best
 
  # computer best particle of swarm and it's fitness
  for i in range(n): # check each particle
    if swarm[i].fitness < best_swarm_fitnessVal:
      best_swarm_fitnessVal = swarm[i].fitness
      best_swarm_pos = copy.copy(swarm[i].position)
 
  # main loop of pso
  Iter = 0
  while Iter < max_iter:
        
     
    # after every 10 iterations
    # print iteration number and best fitness value so far
    if Iter % 10 == 0 and Iter > 1:
      print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)
 
    for i in range(n): # process each particle
       
      # compute new velocity of curr particle
      for k in range(dim):
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()
     
        swarm[i].velocity[k] = (
                                 (w * swarm[i].velocity[k]) +
                                 (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) + 
                                 (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k]))
                               ) 
 
 
        # if velocity[k] is not in [minx, max]
        # then clip it
        if swarm[i].velocity[k] < minx:
          swarm[i].velocity[k] = minx
        elif swarm[i].velocity[k] > maxx:
          swarm[i].velocity[k] = maxx
 
 
      # compute new position using new velocity
      for k in range(dim):
        swarm[i].position[k] += swarm[i].velocity[k]
   
      # compute fitness of new position
      swarm[i].fitness = fitness(swarm[i].position)
 
      # is new position a new best for the particle?
      if swarm[i].fitness < swarm[i].best_part_fitnessVal:
        swarm[i].best_part_fitnessVal = swarm[i].fitness
        swarm[i].best_part_pos = copy.copy(swarm[i].position)
 
      # is new position a new best overall?
      if swarm[i].fitness < best_swarm_fitnessVal:
        best_swarm_fitnessVal = swarm[i].fitness
        best_swarm_pos = copy.copy(swarm[i].position)
     
    # for-each particle
    Iter += 1
  #end_while
  return best_swarm_pos
# end pso


#Driver Code for the PSO-NN Optimization:
def PSO_NN_Classification(X,y,layer_dims = [2,16,1],metric_used = "accuracy",num_particles = 50,max_iter = 100,w = 0.729,c1 = 1.49445,c2 = 1.49445,random_state=42):
    
    layer_dims[0] = X.shape[0] #Overwrite the default number of neurons in the inout layer of the Neural Network in order to cover the case of the dimensionality not being 2
    
    parameters_initial = initialize_parameters_deep(layer_dims)
    
    param_flat = flatten_param(parameters_initial) 

    def fitness_calculation(position):
        fitnessVal = compute_fitness(X,y,np.array(position),parameters_initial,metric_used)
        return fitnessVal

    dim = param_flat.size
 
    print("num_particles  =  ",num_particles)
    print("max_iter  =  ",max_iter)
    print("\nStarting PSO Minimization algorithm:\n")

    best_position = pso(fitness_calculation, max_iter, num_particles, dim, -1.0, 1.0, w = 0.729,c1 = 1.49445,c2 = 1.49445,seed=random_state)

    fitnessVal = fitness_calculation(best_position)

    best_pos_dict = reconstruct_param(np.array(best_position),parameters_initial)

    pred = L_model_forward(X, best_pos_dict,threshold=0.5).ravel()

    if metric_used == "accuracy":
        pred *= -1
        fitnessVal = 1 - fitnessVal
    

    print(f"\nEnd PSO Minimization Algorithm \n \n Best {metric_used} achieved is",fitnessVal)


    
    
    return pred,best_pos_dict,fitnessVal