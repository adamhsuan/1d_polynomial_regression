# this is a simple ML regression model to fit a polynomial function to a single input, single output data set. 
# graphs the data and function using matplotlib

import matplotlib.pyplot as plt
import random
import numpy as np

# contants and hyperparameters to initialize data and ML model
interval = (-2,2)
num_data_points = 100
num_data_poly_params = 4
num_params = 8

max_initial_weight_data_poly = 100/interval[1]**num_data_poly_params
max_initial_weight_model = 100/interval[1]**num_params
noise_standard_dev = 1
num_polynomial_points = 100
num_epochs = 100
epoch_size = 50
num_grad_descent_steps = 1000
eta = 1
# generate data. we start by generating a set of weights for a polynomial we'll use to create data to roughly
# match. Generate a set of random x values in an interval. Then, for each of the x values generate a y value 
# using our generated polynomial and adding some randomness factor from a normal distribution. 

data_weights = [random.uniform(-1*max_initial_weight_data_poly,max_initial_weight_data_poly) for _ in range(num_data_poly_params)]
model_weights = [random.uniform(-1*max_initial_weight_model, max_initial_weight_model) for _ in range(num_params)]

def y(x, weights):
    y = 0
    for i in range(len(weights)):
        y += x**i*weights[i]
    return y

def get_polynomial_points(num_points, weights):
    x_vals = np.linspace(interval[0], interval[1], num_points)
    y_vals = np.zeros(num_points)
    for i in range(len(weights)):
        y_vals += weights[i]*x_vals**i
    return x_vals, y_vals

data_x_vals = [random.uniform(interval[0],interval[1]) for _ in range(num_data_points)]
# stores the data as tuples (x,t)
data = [(data_x_vals[i], y(data_x_vals[i], data_weights)+np.random.normal(loc=0, scale=noise_standard_dev)) for i in range(num_data_points)]
data_y_vals = [data_point[1] for data_point in data]

"""
print(data)
plt.scatter(data_x_vals, data_y_vals)
polynomial_x_vals, polynomial_y_vals = get_polynomial_points(num_polynomial_points, data_weights)
plt.plot(polynomial_x_vals, polynomial_y_vals)
plt.show()
"""

def loss(data, model_weights):
    loss = 0
    for x,t in data:
        loss += 0.5*(y(x, model_weights)-t)**2
    return loss

def loss_gradient(data, model_weights):
    loss_gradient = np.zeros(num_params)
    for x,t in data:
        loss_gradient += (y(x, model_weights)-t)*np.array([x**i for i in range(num_params)])
    return loss_gradient

def train_model(model_weights, data, eta, training_time):
    start_time = time.perf_counter()
    weights_over_epochs = []
    initial_model_weights = model_weights
    #print(f'initial loss {loss(data, model_weights)}')
    epoch = 0
    while time.perf_counter()-start_time < training_time:
        change_eta = False
        epoch_data = random.sample(data, k=epoch_size)
        for _ in range(num_grad_descent_steps):
            model_weights -= eta*loss_gradient(epoch_data, model_weights)
            if loss(epoch_data, model_weights) > 10**10:
                change_eta = True
                epoch = 0
                eta *= 0.9
                #print(f'adjusted eta. new eta: {eta}')
                if len(weights_over_epochs) == 0:
                    model_weights = initial_model_weights.copy()
                else:
                    model_weights = weights_over_epochs[-1].copy()
                break
        if not(change_eta):
            weights_over_epochs.append(model_weights.copy())
            #print(f'loss error after {epoch} epoch: {(loss(data, model_weights)-ideal_fit_loss)/ideal_fit_loss*100:.2f}%')
            epoch+=1
    return weights_over_epochs, epoch

"""
ideal_fit_loss = loss(data, data_weights)
print(f'ideal fit loss: {ideal_fit_loss}')

weights_over_epochs = train_model(model_weights, data, ideal_fit_loss, eta)


plt.figure(1)
plt.scatter(data_x_vals, data_y_vals)
polynomial_x_vals, polynomial_y_vals = get_polynomial_points(num_polynomial_points, data_weights)
plt.plot(polynomial_x_vals, polynomial_y_vals, color="green")

for n,i in enumerate(range(0, len(weights_over_epochs), int(len(weights_over_epochs)/10))):
    model_x_vals, model_y_vals = get_polynomial_points(num_polynomial_points, weights_over_epochs[i])
    plt.plot(model_x_vals, model_y_vals, color=(n/12+0.1,0,0))

plt.axis([interval[0]-(interval[1]-interval[0])/4, interval[1]+(interval[1]-interval[0])/4, min(polynomial_y_vals)*2, max(polynomial_y_vals)*2])


losses_over_epochs = [loss(data, epoch_weights) for epoch_weights in weights_over_epochs]

plt.figure(2)
plt.axis([0, num_epochs, 0, 1000])
plt.plot(losses_over_epochs)
plt.show()
"""

# find best hyperparameters (num epochs, num grad descent steps, epoch size) with graph search
# num epochs is the number we get to before the time runs out
# then we record the loss against test data 
# we test on num_data_sets_tested and take the average of the losses

import time
import json

num_grad_descent_steps_range = [1, 1000]
num_grad_descent_steps_tested = 20
num_epoch_sizes_tested = 20
num_data_sets_tested = 5
num_training_points = 50
epoch_size_range = [1, num_training_points]
training_time = 10

# (num_epochs, num_grad_descent_steps, epoch_size, loss)
hyperparameter_performances = np.zeros((num_grad_descent_steps_tested, num_epoch_sizes_tested, 4))

n=0
for _ in range(num_data_sets_tested):
    data_weights = [random.uniform(-1*max_initial_weight_data_poly,max_initial_weight_data_poly) for _ in range(num_data_poly_params)]
    model_weights = np.array([random.uniform(-1*max_initial_weight_model, max_initial_weight_model) for _ in range(num_params)])
    data_x, data_y = get_polynomial_points(num_data_points, data_weights)
    training_data_x, testing_data_x = data_x[0:num_training_points], data_x[num_training_points:]
    training_data_y, testing_data_y = data_y[0:num_training_points], data_y[num_training_points:]
    training_data = [(training_data_x[i], training_data_y[i]) for i in range(num_training_points)]
    testing_data = [(testing_data_x[i], testing_data_y[i]) for i in range(num_data_points-num_training_points)]
    for i, num_grad_descent_steps in enumerate(np.linspace(num_grad_descent_steps_range[0], num_grad_descent_steps_range[1], num_grad_descent_steps_tested)):
        num_grad_descent_steps = int(num_grad_descent_steps)
        for j, epoch_size in enumerate(np.linspace(epoch_size_range[0], epoch_size_range[1], num_epoch_sizes_tested)):
            epoch_size = int(epoch_size)
            weights_over_epochs, num_epochs = train_model(model_weights, training_data, eta, training_time)
            the_loss = loss(testing_data, weights_over_epochs[-1])
            hyperparameter_performances[i][j][0] += the_loss/num_data_sets_tested
            hyperparameter_performances[i][j][1] = num_grad_descent_steps
            hyperparameter_performances[i][j][2] = epoch_size
            hyperparameter_performances[i][j][3] = num_epochs
            print(f'num epochs: {num_epochs}, num grad descent steps: {num_grad_descent_steps}, epoch size: {epoch_size}, loss: {the_loss}')
            print(f"hyperparam testing {n/(num_grad_descent_steps_tested*num_epoch_sizes_tested*num_data_sets_tested)*100}% complete")
            n+=1
with open('hyperparameter_performances.json','w') as f:
    json.dump(hyperparameter_performances.tolist(), f)


