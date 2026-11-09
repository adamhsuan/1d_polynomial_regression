import json
import numpy as np
import matplotlib.pyplot as plt

with open('hyperparameter_performances.json', 'r') as f:
    hyperparameter_performances = json.load(f)

hpp_graph_list = []
for i in range(len(hyperparameter_performances)):
    hpp_graph_list.append([])
    for j in range(len(hyperparameter_performances[0])):
        hpp_graph_list[i].append(np.log(hyperparameter_performances[i][j][0])/np.log(10))

hpp_graph_np_array = np.array(hpp_graph_list)

best_row, best_col = np.unravel_index(hpp_graph_np_array.argmin(), hpp_graph_np_array.shape)
best_hps_info = hyperparameter_performances[best_row][best_col]
print(best_row)
print(best_col)
best_num_gradient_descent_steps, best_epoch_size, best_num_epochs = best_hps_info[1], best_hps_info[2], best_hps_info[3]
print(f'best num grad descent steps: {best_num_gradient_descent_steps}, best epoch size: {best_epoch_size}, best num epochs: {best_num_epochs}')


min_loss = hpp_graph_list[0][0]
min_loss_i = 0
min_loss_j = 0
for i in range(len(hpp_graph_list)):
    for j in range(len(hpp_graph_list[0])):
        if hpp_graph_list[i][j] < min_loss:
            min_loss = hpp_graph_list[i][j]
            min_loss_i = i
            min_loss_j = j

print(min_loss_i)
print(min_loss_j)
print(min_loss)

hpp_graph_np_array = np.rot90(hpp_graph_np_array, k=1)

#vmin = np.percentile(hpp_graph_np_array, 0)
vmax = np.percentile(hpp_graph_np_array, 90)

for i in range(len(hpp_graph_np_array)):
    for j in range(len(hpp_graph_np_array[i])):
        if hpp_graph_np_array[i][j] >= vmax:
            hpp_graph_np_array[i][j] = "nan"

current_cmap = plt.colormaps['coolwarm'].copy()
current_cmap.set_bad(color='black')

plt.imshow(hpp_graph_np_array, cmap=current_cmap, extent = [1, 1000, 1, 50], aspect='auto')
plt.gca().set_box_aspect(1)
plt.colorbar()
plt.xlabel("gradient descent steps")
plt.ylabel("data points in epoch")
plt.title("average loss over hyperparameter space")
plt.show()


