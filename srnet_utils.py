import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import OrderedDict
from scipy.stats import pearsonr

def plot_losses(save_names, save_path=".", label_var=None):

    _, ax = plt.subplots()

    if isinstance(save_names, str):
        save_names = [save_names]

    if isinstance(label_var, str):
        label_var = [label_var]

    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name:
                
                state = joblib.load(os.path.join(save_path, file_name))
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


def load_model(save_file, save_path=".", model_cls=None, model_type=None):

    # get hyperparameters
    state = joblib.load(os.path.join(save_path, save_file))
    hp = state['hyperparams']

    # create model
    model = model_cls(**hp['arch'])

    # check model type
    if "gc" in hp and hp["gc"] > 0.0:
        
        if model_type is None:
            prefix = "_ghost_model."
        else:
            prefix = f"_{model_type}_model."

        model_state = OrderedDict((k.replace(prefix, ''), v) for k, v in state['model_state'].items() if prefix in k)
    else:
        model_state = state['model_state']

    # update model
    model.load_state_dict(model_state)

    return model


def get_param_diff(save_file, save_path=".", show=False):
    """Compare live and ghost model parameters."""

    # get model state
    state = joblib.load(os.path.join(save_path, save_file))
    hp = state['hyperparams']

    # get mean absolute difference
    if "gc" in hp and hp["gc"] > 0.0:
        res = 0
        count = 0
        for param_name in state['model_state']:
            if "live" in param_name:
                diff = (state['model_state'][param_name] - state['model_state'][param_name.replace("live", "ghost")]).abs()
                res += diff.sum().numpy()
                count += diff.numel()

        res /= count
    else:
        res = 0.0

    if show:
        print(res)
    
    return res


def get_variance(acts):
    return np.var(acts.numpy(), axis=0)


def get_node_order(acts, show=False):
    variances = get_variance(acts)
    node_order = sorted(range(variances.shape[0]), key=lambda x: variances[x], reverse=True)
    if show:
        print(sorted(variances, reverse=True))
        print(node_order)
    return node_order


def node_correlations(acts, nodes, data_tup, nonzero=False):
    for n in nodes:
        print(f"\nNode {n}")

        if nonzero:
            mask = (acts[:,n].abs() > 0.0).tolist()

        for d in data_tup:
            
            corr_info = f"{pearsonr(acts[:,n], d[1])[0]:.4f}"
            if nonzero:
                corr_info += f"/{pearsonr(acts[mask,n], d[1][mask])[0]:.4f}"
            print(f"corr(n{n}, {d[0]}): {corr_info}")


def plot_acts(x_data, y_data, z_data, acts=None, nodes=[], model=None, bias=False, nonzero=False, agg=False, plot_size=500):
    
    _ = plt.figure()
    ax = plt.axes(projection='3d')

    x_data = x_data[:plot_size].numpy()
    y_data = y_data[:plot_size].numpy()
    
    for d in z_data:
        z_data = d[1][:plot_size].numpy()
        ax.scatter3D(x_data, y_data, z_data, label=d[0])

    b = 0
    if model:
        b = model.layers2[0]._parameters['bias'].item() * bias
       
    if agg:
        nonzero = False
        z_data = np.ones(x_data.shape) * b
        b = 0

    for n in nodes:
        if nonzero:
            mask = (acts[:,n].abs() > 0.0).tolist()[:plot_size]
        else:
            mask = [True] * plot_size

        n_data = acts[:,n][:plot_size].numpy()

        if model:
            w = model.layers2[0]._parameters['weight'].detach().numpy()[0, n]
            n_data = w * n_data + b

        if agg:
            z_data += n_data
        else:
            z_data = n_data
                
        ax.scatter3D(x_data[mask], y_data[mask], z_data[mask], label=str(n))

    ax.legend()
    plt.show()