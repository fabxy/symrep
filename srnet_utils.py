import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

def plot_losses(save_names, save_path=".", label_var=None):

    _, ax = plt.subplots()

    if isinstance(save_names, str):
        save_names = [save_names]

    if isinstance(label_var, str):
        label_var = [label_var]

    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name:
                
                # states.append(joblib.load(os.path.join(save_path, file_name)))
                state = joblib.load(os.path.join(save_path, file_name))

                # if label_var:
                #     label = file_name.split('.')[0].split('_')[-1] + ": " + '/'.join([str(states[-1]['hyperparams'][var]) if var in states[-1]['hyperparams'] else str(0) for var in label_var])
                # else:
                label = file_name.split('.')[0]
                
                # plot training loss
                lines = ax.plot(state['train_loss'], label=label)

                # plot validation loss
                ax.plot(state['val_loss'], '--', color=lines[-1].get_color())

    if label_var:
        ax.set_title('/'.join(label_var))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()


def load_model(save_file, save_path=".", model_cls=None):

    # get hyperparameters
    state = joblib.load(os.path.join(save_path, save_file))
    hp = state['hyperparams']

    # create model
    model = model_cls(**hp['arch'])

    # update model
    model.load_state_dict(state['model_state'])

    return model


def get_variance(acts):
    return np.var(acts.numpy(), axis=0)


def get_node_order(acts, show=False):
    variances = get_variance(acts)
    node_order = sorted(range(variances.shape[0]), key=lambda x: variances[x], reverse=True)
    if show:
        print(sorted(variances, reverse=True))
        print(node_order)
    return node_order


def plot_acts(x_data, y_data, z_data, model=None, acts=None, nodes=[], nonzero=False, plot_size=500):
    
    _ = plt.figure()
    ax = plt.axes(projection='3d')

    x_data = x_data[:plot_size].numpy()
    y_data = y_data[:plot_size].numpy()
    
    for d in z_data:
        z_data = d[:plot_size].numpy()
        ax.scatter3D(x_data, y_data, z_data)
       
    # if agg:
    #     nonzero = False
    #     z_data = np.ones(x_data.shape) * b
    #     b = 0

    for n in nodes:

        if nonzero:
            mask = (acts[:,n].abs() > 0.0).tolist()[:plot_size]
        else:
            mask = [True] * plot_size

        z_data = acts[:,n][:plot_size].numpy()
        
        # if agg:
        #     z_data += n_data
        
        ax.scatter3D(x_data[mask], y_data[mask], z_data[mask], label=str(n))

    ax.legend()
    plt.show()