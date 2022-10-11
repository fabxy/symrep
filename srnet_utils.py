import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import OrderedDict
from scipy.stats import pearsonr
import torch

def plot_losses(save_names, save_path=".", excl_names=list(), label_var=None, log=False, disc_loss=False):

    _, ax = plt.subplots()

    if isinstance(save_names, str):
        save_names = [save_names]

    if isinstance(label_var, str):
        label_var = [label_var]

    model_names = []
    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name and not any([n in file_name for n in excl_names]):
                
                state = joblib.load(os.path.join(save_path, file_name))
                label = file_name.split('.')[0]
                model_names.append(label)
                
                # plot training loss
                if disc_loss:
                    lines = ax.plot(np.array(state['train_loss']) - np.array(state['disc_loss']), label=label)
                else:
                    lines = ax.plot(state['train_loss'], label=label)

                # plot validation loss
                epochs = len(state['train_loss'])
                logs = len(state['val_loss'])
                log_freq = int(np.ceil(epochs / logs))
                x_data = np.arange(logs) * log_freq
                ax.plot(x_data, state['val_loss'], '--', color=lines[-1].get_color())

                # plot discriminator regularization loss
                if disc_loss:
                    ax.plot(state['disc_loss'], ':', color=lines[-1].get_color())

    if label_var:
        ax.set_title('/'.join(label_var))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if log:
        ax.set_yscale('log')
    ax.legend()
    plt.show()

    return model_names


def plot_corrs(save_names, save_path=".", excl_names=list(), label_var=None):

    _, ax = plt.subplots()

    if isinstance(save_names, str):
        save_names = [save_names]

    if isinstance(label_var, str):
        label_var = [label_var]

    model_names = []
    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name and not any([n in file_name for n in excl_names]):

                state = joblib.load(os.path.join(save_path, file_name))
                label = file_name.split('.')[0]
                model_names.append(label)

                # plot minimum correlation
                min_corr = [c.abs().max(dim=1).values.min() for c in state['corr_mat']]

                epochs = len(state['train_loss'])
                logs = len(min_corr)
                log_freq = int(np.ceil(epochs / logs))
                x_data = np.arange(logs) * log_freq

                ax.plot(x_data, min_corr, label=label)

    if label_var:
        ax.set_title('/'.join(label_var))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Minimum correlation")
    ax.legend()
    plt.show()

    return model_names


def plot_reg_percentage(save_names, save_path=".", excl_names=list(), label_var=None, log=False):

    _, ax = plt.subplots()

    if isinstance(save_names, str):
        save_names = [save_names]

    if isinstance(label_var, str):
        label_var = [label_var]

    model_names = []
    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name and not any([n in file_name for n in excl_names]):

                state = joblib.load(os.path.join(save_path, file_name))
                label = file_name.split('.')[0]
                model_names.append(label)

                tot_loss = np.array(state['train_loss'])
                disc_loss = np.array(state['disc_loss'])
                pred_loss = tot_loss - disc_loss

                lines = ax.plot(np.abs(disc_loss)/(np.abs(disc_loss) + np.abs(pred_loss)), label=label)

    if label_var:
        ax.set_title('/'.join(label_var))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Critic regularization percentage")
    if log:
        ax.set_yscale('log')
    ax.legend()
    plt.show()

    return model_names


def plot_disc_preds(save_names, save_path=".", excl_names=list(), label_var=None, log=False):

    _, ax = plt.subplots()

    if isinstance(save_names, str):
        save_names = [save_names]

    if isinstance(label_var, str):
        label_var = [label_var]

    model_names = []
    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name and not any([n in file_name for n in excl_names]):

                state = joblib.load(os.path.join(save_path, file_name))
                label = file_name.split('.')[0]
                model_names.append(label)

                # plot average discriminator predictions
                state['disc_preds']

                epochs = len(state['train_loss'])
                logs = len(state['disc_preds'])
                log_freq = int(np.ceil(epochs / logs))
                x_data = np.arange(logs) * log_freq

                ax.plot(x_data, state['disc_preds'], label=label)

    if label_var:
        ax.set_title('/'.join(label_var))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg. SD prediction")
    if log:
        ax.set_yscale('log')
    ax.legend()
    plt.show()

    return model_names


def plot_disc_accuracies(save_names, save_path=".", excl_names=list(), avg_hor=None, uncertainty=False):

    _, ax = plt.subplots()

    if isinstance(save_names, str):
        save_names = [save_names]

    model_names = []
    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name and not any([n in file_name for n in excl_names]):

                state = joblib.load(os.path.join(save_path, file_name))
                label = file_name.split('.')[0]
                model_names.append(label)

                # calculate accuracies
                try:
                    tot_accs = np.array(state['tot_accs'])[:,-1]
                except:
                    # legacy
                    tot_accs = np.array(state['tot_accs'])

                if avg_hor is not None:
                    avg_accs = np.array([tot_accs[max(0,i+1-avg_hor):i+1].mean() for i in range(tot_accs.shape[0])])
                else:
                    avg_accs = tot_accs
                
                # plot accuracies
                epochs = np.arange(avg_accs.shape[0])
                lines = ax.plot(avg_accs, label=label)

                # plot uncertainty
                if avg_hor is not None and uncertainty:
                    std_accs = np.array([tot_accs[max(0,i+1-avg_hor):i+1].std() for i in range(tot_accs.shape[0])])
                    ax.fill_between(epochs, avg_accs-std_accs, avg_accs+std_accs, alpha=0.25)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.show()

    return model_names


def plot_disc_losses(save_names, save_path=".", excl_names=list(), avg_hor=None, uncertainty=False, summation=True):

    _, ax = plt.subplots()

    styles = ['-', '--', ':']

    if isinstance(save_names, str):
        save_names = [save_names]

    model_names = []
    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name and not any([n in file_name for n in excl_names]):

                state = joblib.load(os.path.join(save_path, file_name))
                label = file_name.split('.')[0]
                model_names.append(label)

                # calculate losses
                tot_losses = np.array(state['tot_losses'])[:,-1,:]

                # correct gradient penalty
                if 'gp' in state['hyperparams']['disc']: 
                    tot_losses[:,-1] *= state['hyperparams']['disc']['gp']
                
                if summation:
                    tot_losses = tot_losses.sum(axis=1).reshape(-1,1)

                if avg_hor is not None:
                    avg_losses = np.array([tot_losses[max(0,i+1-avg_hor):i+1].mean(axis=0) for i in range(tot_losses.shape[0])])
                else:
                    avg_losses = tot_losses
                
                # plot losses
                epochs = np.arange(avg_losses.shape[0])
                for i in range(avg_losses.shape[1]):
                    if i == 0:
                        lines = ax.plot(avg_losses[:,i], label=label)
                    else:
                        lines = ax.plot(avg_losses[:,i], color=lines[-1].get_color(), ls=styles[i])

                # plot uncertainty
                if avg_hor is not None and uncertainty and summation:
                    std_losses = np.array([tot_losses[max(0,i+1-avg_hor):i+1].std(axis=0) for i in range(tot_losses.shape[0])])
                    ax.fill_between(epochs, (avg_losses-std_losses)[:,0], (avg_losses+std_losses)[:,0], alpha=0.25)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Losses")
    ax.legend()
    plt.show()

    return model_names


def plot_disc_gradients(save_names, save_path=".", excl_names=list(), avg_hor=None, uncertainty=False):

    _, ax = plt.subplots()

    if isinstance(save_names, str):
        save_names = [save_names]

    model_names = []
    for save_name in save_names:
        for file_name in sorted(os.listdir(save_path)):
            if save_name in file_name and not any([n in file_name for n in excl_names]):

                state = joblib.load(os.path.join(save_path, file_name))
                label = file_name.split('.')[0]
                model_names.append(label)

                # calculate gradients
                tot_grads = np.array(state['tot_grads'])[:,-1]

                if avg_hor is not None:
                    avg_grads = np.array([tot_grads[max(0,i+1-avg_hor):i+1].mean() for i in range(tot_grads.shape[0])])
                else:
                    avg_grads = tot_grads
                
                # plot gradients
                epochs = np.arange(avg_grads.shape[0])
                lines = ax.plot(avg_grads, label=label)

                # plot uncertainty
                if avg_hor is not None and uncertainty:
                    std_grads = np.array([tot_grads[max(0,i+1-avg_hor):i+1].std() for i in range(tot_grads.shape[0])])
                    ax.fill_between(epochs, avg_grads-std_grads, avg_grads+std_grads, alpha=0.25)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max. Gradient")
    ax.legend()
    plt.show()

    return model_names


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


def load_disc(save_file, save_path=".", disc_cls=None):

    # get hyperparameters
    state = joblib.load(os.path.join(save_path, save_file))
    hp = state['hyperparams']

    # create critic
    disc_in_size = hp['batch_size']
    try:
        if hp['ext_type'] == "stack":
            disc_in_size *= hp['ext_size'] + 1
        elif hp['ext_type'] == "embed":
            hp['disc']['emb_size'] = hp['ext_size'] + 1
    except: pass

    critic = disc_cls(disc_in_size, **hp['disc'])

    # update critic
    critic.load_state_dict(state['disc_state'])

    return critic


def save_preds(data, var_name, save_path=".", model_name=None):
    
    # save predictions data
    np.savetxt(os.path.join(save_path, var_name + '.gz'), data.numpy())

    # save model name
    if model_name:
        with open(os.path.join(save_path, var_name + '.info'), "w") as f:
            f.write(model_name + '\n')


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


def node_correlations(acts, nodes, data_tup, show=True):

    corr_mat = []
    for n in nodes:
        if show:
            print(f"\nNode {n}")
        
        corr = []
        for d in data_tup:
            corr.append(pearsonr(acts[:,n], d[1])[0])
            if show:
                print(f"corr(n{n}, {d[0]}): {corr[-1]:.4f}")
    
        corr_mat.append(corr)
    
    return corr_mat    


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


def extend(data, *args, ext_type=None):

    if ext_type == "stack":
        for ext_data in args:
            ext_data = ext_data.flatten(-2)
            
            if len(ext_data.shape) == 1:
                ext_data = ext_data.repeat(list(data.shape[:-1]) + [1])
            elif len(ext_data.shape) == len(data.shape):
                pass
            else:
                raise RuntimeError(f"Extension with {len(ext_data.shape)}D tensor is not supported yet.")
            
            data = torch.cat((data, ext_data), dim=-1)
        
    elif ext_type == "embed":
        data = data.unsqueeze(-1)
        for ext_data in args:
            
            if len(ext_data.shape) == 2:
                ext_data = ext_data.repeat(list(data.shape[:-2]) + [1,1])
            elif len(ext_data.shape) == len(data.shape):
                pass
            else:
                raise RuntimeError(f"Extension with {len(ext_data.shape)}D tensor is not supported yet.")
            
            data = torch.cat((data, ext_data), dim=-1)
    
    return data


def triangle(x, a=1.0, c=0.5):
    
    mask = ((x // c) % 2).astype(np.bool)
    
    y = np.zeros_like(x)
    
    y[mask] = 2*c - x[mask]
    y[~mask] = x[~mask]
    
    y -= (y//(2*c))*2*c
    y *= a
    
    return y


def triangle_cos(x, a=1.0, c=0.5):
    
    y = np.cos(np.pi/c*x)
    
    y *= -a*c/2
    y += a*c/2
    
    return y