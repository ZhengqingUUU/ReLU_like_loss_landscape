from matplotlib import interactive, widgets
import torch
import numpy as np
from model import *
from helpers import *
from tqdm import tqdm
import torch.optim as optim 
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os 
import shutil
from PIL import Image
import glob
import sys
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import copy
from matplotlib.ticker import ScalarFormatter
# Set x-axis labels in scientific notation format
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))  # Adjust the limits as needed

formatter1 = ScalarFormatter(useMathText=True)
formatter1.set_scientific(True)
formatter1.set_powerlimits((-3, 3))  # Adjust the limits as needed

FIGSIZE=(6,4)

def generate_orig_1D_dataset(plot = False, dataset_name = 'orig_1D_pics', plot_name = 'dataset.png'):
    xs = torch.FloatTensor((-0.5, -0.2, 0.2, 0.5)).reshape(4,1)
    x_bias = torch.ones(xs.shape)
    x_data = torch.cat((xs, x_bias), dim=1)
    y_data = torch.FloatTensor((0.23, 0.02, 0.02, 0.23)).reshape(4,1)
    if plot:
        os.makedirs(dataset_name, exist_ok=True)
        plt.scatter(xs, y_data, marker='*', c='r')
        plt.savefig(os.path.join(dataset_name, plot_name))
        plt.show()
        plt.clf()
    return x_data, y_data, dataset_name


def generate_5pt_dataset(plot = False, dataset_name= 'five_point_nonsym_dataset', plot_name = 'dataset.png'):
    xs = torch.FloatTensor((-1, -0.6, -0.1, 0.3, 0.7)).reshape(5,1)
    x_bias = torch.ones(xs.shape)
    x_data = torch.cat((xs, x_bias), dim=1)
    y_data = torch.FloatTensor((0.28, -0.1, 0.03, 0.23, -0.22)).reshape(5,1)
    if plot:
        os.makedirs(dataset_name, exist_ok=True)
        plt.scatter(xs, y_data, marker='*', c='r')
        plt.savefig(os.path.join(dataset_name, plot_name))
        plt.show()
        plt.clf()
    return x_data, y_data, dataset_name


def cal_betas(mu, mu_p, width):
    beta_1 = width**mu
    beta_2 = width**mu_p
    return beta_1, beta_2

def jpg2gif(save_dir, target_dir, gif_name, parse_num_to_order = True, parse_num_id = -1, duration=500):
    """jpg2gif process within one folder. parse_num_to_order is to solve the problem that omse times the pictures 
    should be reordered according to the number in their names.

    Args:
        save_dir (_type_): save directory of all the jpgs
        target_dir: save directory of the gif
        gif_name (_type_): _description_
        parse_num_to_order (bool, optional): _description_. Defaults to True.
        parse_num_id (int, optional): _description_. Defaults to -1.
        duration (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """
    frames = []
    imgs = glob.glob(os.path.join(save_dir,"*.jpg"))
    def return_num_in_str(string, parse_num_id = parse_num_id):
        return int(re.findall(r'\d+', string)[parse_num_id])
    if parse_num_to_order:
        imgs = sorted(imgs, key = return_num_in_str)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(os.path.join(target_dir,f"{gif_name}"+".gif"), format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=duration, loop=0)


def be_deterministic(TrueORFalse, random_seed = 42):
    """Note, random seed is not assigned to torch.manual_seed or np.random.seed(). It should be assigned right before
    every random operation.

    Args:
        TrueORFalse (_type_): _description_
        random_seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    if TrueORFalse:
    # torch.use_deterministic_algorithms(True) ##not compatible with pytorch version on the cluster
        torch.backends.cudnn.deterministic = True
        return random_seed
    else:
        return None

def plot_interactive_loss(interactive_plot_df, loss_display_lim, savepath):
    fig = px.line(interactive_plot_df, x = interactive_plot_df['iterations'],\
        y = interactive_plot_df['losses'])
    fig.update_yaxes(range = [0, loss_display_lim])
    fig.write_html(savepath)


def plot_output_func(x_data_plt, y_data_plt, model, global_iter, device, same_output_plot_annotate_num,\
                output_plot_step,output_pic_savedir, output_pic_prefix):
    plt.scatter(x_data_plt[:,0], y_data_plt, marker='*', c='r')
    ## the next a few lines is to figure out the range for output function plotting
    ## the behavior of near x = 0 should always be ovserved closely.
    x_data_plt_left = x_data_plt[:,0].min() - 0.2
    x_data_plt_right= x_data_plt[:,0].max() + 0.2
    x_interp_min = x_data_plt_left if x_data_plt_left<-2 else -2
    x_interp_max = x_data_plt_right if x_data_plt_right>2 else 2
    plt_x = torch.arange(x_interp_min, x_interp_max, 0.02).reshape(-1,1)
    plt_bias = torch.ones(plt_x.shape[0],1)
    plt_output = model(torch.cat((plt_x, plt_bias),dim = 1).to(device))
    plt.plot(plt_x.cpu().numpy(), plt_output.detach().cpu().numpy())
    plt.ylim([min(y_data_plt) - 0.2, max(y_data_plt)+0.2 ])
    iter_annotation = global_iter
    plt.annotate(f"epoch:{iter_annotation/1000:.0f}k", xy=(0.05, 0.90), xycoords='axes fraction',size = 20)
    ### plot y_pred and errors
    y_pred = model(x_data_plt.to(device)).detach().cpu().numpy()
    x_numpy = x_data_plt[:,0].numpy().reshape(-1)
    plt.scatter(x_data_plt[:,0], y_pred, marker = 'o', c = 'k', s = 10)
    y_gt = y_data_plt.cpu().numpy()
    error = y_pred - y_gt
    font = {'family':'serif','color':'black','size':15}   
    plt.xlabel("First Input Component", fontsize = 20,fontdict=font)
    plt.ylabel("Target/Prediction", fontsize = 20,fontdict=font)
    plt.savefig(os.path.join(output_pic_savedir, output_pic_prefix+f"{global_iter}.jpg"), dpi = 300, bbox_inches='tight')
    plt.clf()

def handle_directory(savefolder, config, loss_pic_savedir, loss_savedir, weight_savedir,\
                     error_savedir, output_pic_savedir, model_savedir, loss_file_name, loss_pic_gif_savedir):
    if os.path.exists(os.path.join(savefolder, config)):
        shutil.rmtree(os.path.join(savefolder, config)) 
    loss_pic_savedir = os.path.join(savefolder, config, loss_pic_savedir)
    loss_savedir = os.path.join(savefolder, config, loss_savedir)
    weight_savedir = os.path.join(savefolder, config, weight_savedir)
    error_savedir = os.path.join(savefolder, config, error_savedir)
    output_pic_savedir = os.path.join(savefolder, config, output_pic_savedir)
    model_savedir = os.path.join(savefolder, config, model_savedir)
    loss_pic_gif_savedir = os.path.join(savefolder, config, loss_pic_gif_savedir)
    os.makedirs(loss_pic_savedir, exist_ok=True)
    os.makedirs(loss_savedir, exist_ok=True)
    os.makedirs(weight_savedir, exist_ok=True)
    os.makedirs(output_pic_savedir, exist_ok=True)
    os.makedirs(error_savedir, exist_ok=True)
    os.makedirs(loss_pic_gif_savedir, exist_ok=True)
    loss_savepath = os.path.join(loss_savedir, loss_file_name)
    
    return loss_pic_savedir, loss_savedir, weight_savedir, error_savedir, output_pic_savedir, model_savedir, loss_savepath, loss_pic_gif_savedir

def plot_loss(iter_ls, loss_ls, loss_display_lim, loss_pic_savedir, loss_pic_name):
    plt.figure(figsize=FIGSIZE)
    plt.plot(iter_ls[1:], loss_ls[1:])
    plt.ylim([0, loss_display_lim])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig(os.path.join(loss_pic_savedir, loss_pic_name+'.jpg'),dpi = 300)
    plt.clf()
                    

def plot_loss_gif_fodder(iter_ls, xlim_max, loss_ls, loss_display_lim, loss_pic_gif_savedir, \
                         same_output_plot_annotate_num, output_plot_step):
    plt.figure(figsize=FIGSIZE)
    plt.plot(iter_ls[1:], loss_ls[1:])
    plt.ylim([0, loss_display_lim])
    font = {'family':'serif','color':'black','size':15}   
    # plt.xlabel(r"$x$", fontsize = 20,fontdict=font)
    # plt.ylabel(r"$\tilde{y}$", fontsize = 20,fontdict=font)
    plt.xlabel("Epoch", fontsize = 20,fontdict=font)
    plt.ylabel("Loss", fontsize = 20,fontdict=font)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlim([0,xlim_max])
    ## TODO: changable
    # iter_annotation = np.floor(iter_ls[-1]/(same_output_plot_annotate_num*output_plot_step))*(same_output_plot_annotate_num*output_plot_step)
    iter_annotation =iter_ls[-1] 
    ## TODO: uncomment them when needed
    plt.annotate(f"epoch:{iter_annotation/1000:.0f}k", xy=(0.05, 0.90), xycoords='axes fraction',size = 20)
    plt.savefig(os.path.join(loss_pic_gif_savedir, f'{iter_ls[-1]}'+'.jpg'),dpi = 300, bbox_inches = 'tight')
    plt.clf()

def train(model, x_data, y_data, lr=1e-3, loss_function = F.mse_loss, stopping_loss = 1e-5, max_iter = 1e6, \
         savefolder = 'results', config = 'test',\
         loss_update_step=100, display_loss = True, loss_display_lim = 0.03, loss_pic_savedir = 'pics',  loss_pic_name = 'loss',\
         loss_print_step = 2000, print_loss = True, loss_savedir = 'loss_recording', loss_file_name = 'loss_recording.txt',
         loss_xlim_max = 8e5, loss_pic_gif_savedir = 'loss_gif',\
         weight_record_step = 500, record_weight = True, weight_savedir = 'weight_recording', \
         output_plot_step = 2000, output_plot = True, output_pic_savedir = os.path.join('pics', 'output_func'),\
         output_pic_prefix = 'output_func', same_output_plot_annotate_num = 10,
         error_recording = True, error_record_step = 2000, error_savedir = "error_recording",\
         model_savedir = 'model_optimizer',
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),):
    """
    Train the initialized neural network, plot the loss curve, save input and output weight tensors, plot the 
    network output curve.
    Args:
    """
    print(config)
    print("training")

    # handling directory maneuvers
    loss_pic_savedir, loss_savedir, weight_savedir, error_savedir, output_pic_savedir,\
          model_savedir, loss_savepath, loss_pic_gif_savedir = handle_directory(savefolder, config, loss_pic_savedir,\
                                                           loss_savedir, weight_savedir,\
                                                            error_savedir, output_pic_savedir, \
                                                                model_savedir, loss_file_name, loss_pic_gif_savedir)

    optimizer = optim.SGD(model.parameters(), lr)
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    model = model.to(device)
    global_iter = 0
    loss = torch.tensor(100).float() ## just for initialization
    loss_ls = []
    iter_ls = []
    with tqdm(total=max_iter, position=0, leave=True) as pbar:
        while loss > stopping_loss and global_iter < max_iter:
            # print(global_iter)

            pbar.update()

            ## do a plot
            if global_iter % loss_update_step == 0 and display_loss:
                loss_ls.append(loss.detach().cpu().numpy())
                iter_ls.append(global_iter)
                plot_loss(iter_ls=iter_ls, loss_ls=loss_ls, loss_display_lim=loss_display_lim,\
                          loss_pic_savedir=loss_pic_savedir, loss_pic_name=loss_pic_name)
                plot_loss_gif_fodder(iter_ls = iter_ls, xlim_max = loss_xlim_max, loss_ls = loss_ls,\
                                     loss_display_lim =  loss_display_lim, \
                                    loss_pic_gif_savedir = loss_pic_gif_savedir,\
                                   same_output_plot_annotate_num = same_output_plot_annotate_num,\
                                     output_plot_step = output_plot_step )


            ## print the loss
            if global_iter % loss_print_step == 0 and print_loss:
                loss_savepath = os.path.join(loss_savedir, loss_file_name)
                with open(loss_savepath, 'a+') as f:
                    f.write(f"iter: {global_iter}, loss: {loss: .7f}\n")
            
            ##FIXME: ac-hoc version for 2-layer network
            if global_iter % weight_record_step == 0 and record_weight:
                input_weight = dict(model.named_modules())['input_layer'].weight.data
                output_weight = dict(model.named_modules())['output_layer'].weight.data
                input_weight_savepath = os.path.join(weight_savedir, f"iter{global_iter}_input.pt")
                torch.save(input_weight, input_weight_savepath)
                output_weight_savepath = os.path.join(weight_savedir, f"iter{global_iter}_output.pt")
                torch.save(output_weight, output_weight_savepath)
            ## save pictures for the output function
            if global_iter % output_plot_step == 0 and output_plot:
                model.eval()
                x_data_plt = x_data.cpu() 
                y_data_plt = y_data.cpu()
                plot_output_func(x_data_plt, y_data_plt, model, global_iter, device,\
                    same_output_plot_annotate_num, output_plot_step,output_pic_savedir, output_pic_prefix)
                model.train()
            
            ## The training routine
            output = model(x_data)
            error =  output.detach().cpu()- y_data.detach().cpu() # will be used

            if global_iter % error_record_step == 0 and error_recording:
                error_savepath = os.path.join(error_savedir, f"iter{global_iter}_error.pt")
                torch.save(error, error_savepath)

            loss = loss_function(output, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter += 1
        
        iter_arr = np.array(iter_ls)
        loss_arr = np.array(loss_ls)
        np.savez(os.path.join(loss_savedir, "loss.npz"), iter_arr = iter_arr, loss_arr = loss_arr)
    # save the model
    model_savepath = os.path.join(model_savedir,'model_and_optimizer.pt')
    os.makedirs(model_savedir, exist_ok=True)
    torch.save({
            'epoch': global_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_savepath)

    ## After the end of training, plot interactive plots
    # loss
    print("interactive plotting")
    iter_tensor = np.asarray(iter_ls).reshape(-1,1)
    interactive_plot_df = pd.DataFrame(iter_tensor, columns = ['iterations'])
    interactive_plot_df['losses'] = np.asarray(loss_ls).reshape(-1,1)
    interactive_loss_savepath = os.path.join(loss_pic_savedir, loss_pic_name+'.html')
    plot_interactive_loss(interactive_plot_df, loss_display_lim,interactive_loss_savepath)

def to_right_range(angle_tensor):
    ## move from (-pi, pi) to (0, 2pi)
    right_angle = (angle_tensor + 2*torch.pi)%(2*torch.pi)
    assert min(right_angle) >= 0
    assert max(right_angle) <= 2*torch.pi
    return right_angle

def derive_sectors(x_data, sector_names = None):
    """ from x_data derive existing sectors in the phase plane

    Args:
        x_data (_type_): _description_
        sector_names (_type_, optional): _description_. Defaults to None.
    """
    assert len(sector_names) == len(set(sector_names)), "there exists duplicates in sector_names!"
    angles = to_right_range(np.arctan2(x_data[:,1],x_data[:,0]) )
    start_angles = to_right_range(angles - math.pi/2)
    end_angles = to_right_range(angles + math.pi/2)
    sector_boundaries, _ = torch.sort(torch.cat((start_angles, end_angles)))

    ## add 0, and 2pi to boundaries as needed
    if sector_boundaries[0] != 0:
        sector_boundaries = torch.cat((torch.tensor([0]), sector_boundaries))
    if sector_boundaries[-1] != 2*torch.pi:
        sector_boundaries = torch.cat((sector_boundaries, torch.tensor([2*torch.pi])))
    sectors = []
    for i in range(sector_boundaries.shape[0]-1):
        left = float(sector_boundaries[i])
        right = float(sector_boundaries[i+1])
        if i < sector_boundaries.shape[0] - 2:
            sectors.append(pd.Interval(left,right,closed = 'left'))
        else:
            ## close = 'both' makes sure that 2pi is included in the last sector
            sectors.append(pd.Interval(left, right, closed = 'both'))

    if sector_names is not None:
        assert len(sectors) == len(sector_names), "number of derived sectors is different from number of given sector names"
        return sector_boundaries, sectors, dict(zip(sector_names, sectors))
    else:
        return sector_boundaries, sectors

def in_which_sector(sectors, angle):
    """return a True/False list indicating which sector angle is in.

    Args:
        sectors (_type_): _description_
        angle (_type_): _description_

    Returns:
        _type_: _description_
    """
    flags = []
    for sector in sectors:
        flags.append(angle in sector) 
    return flags


def get_perturb_params(model, width = 50, not_perturb_dead = True, dead_ids = None ,default_min_param_abs = 1):
    """get the parameters requried for get_perturbed_model function
    """
# obtain the min abs of the parameters, so the perturbation extent can be determined
# and count the number of parameters
    min_param_abs = default_min_param_abs
    # parameter_number = 0
    for layer in model.parameters():
        if not not_perturb_dead:
            examined_layer = layer 
        else: # choose smallest parameters in alive neurons
            effective_layer = layer.squeeze()#TODO: notice, this will squeeze away all the dimensions with size 1, because for the 1d output case, the output weight is of size [1, width], which is very annoying.
            assert effective_layer.shape[0] == width
            # generate the mask indicate the positions of alive neurons
            alive_ids = [i for i in range(width) if i not in dead_ids]
            examined_layer = effective_layer[alive_ids]
        layer_min = torch.min(torch.abs(examined_layer))
        if layer_min < min_param_abs: min_param_abs = layer_min
    perturb_amplitude = min_param_abs/10
    print(f"perturb_amplitude: {perturb_amplitude}")
    return perturb_amplitude

def get_perturbed_model(model, perturb_amplitude,  width = 50, not_perturb_dead = True, dead_ids = None,  Gaussian = False):
    """perturb all the parameters with independently sampled noise from a uniform distribution whose range is
    between negative and positive perturb_amplitude. If Gaussian is True, then the noise is sampled independently 
    from a Gaussian distribution.
    """
    with torch.no_grad():
        noise_ls = []
        for layer in model.parameters():
            layer.squeeze_() # squeeze away the dimensions with size 1, the first dimension for each layer should be width
            assert layer.shape[0] == width
            for id, parameter in enumerate(layer):
                noise = generate_noise(parameter.shape, perturb_amplitude, Gaussian = Gaussian)
                if id in dead_ids and not_perturb_dead: # the first dimension of layer is always the width
                    noise = torch.zeros(parameter.shape)
                parameter.add_(noise)
                # concatenate noise together:
                noise_ls.append(noise.reshape(-1))
                # noise_ls contains the parameters with this sequence: first: 50 * 2 parameters for the input 
                # weights, both components of the same neuron are next to each other. Then, 50 parameters for the 
                # output weights.   
        noise_arr = torch.cat(noise_ls)
    model.eval()
    return model, noise_arr

def generate_noise(shape_tuple, perturb_amplitude, Gaussian = False):
    if not Gaussian:
        noise = perturb_amplitude*(torch.rand(shape_tuple)*2 - 1)
    else:
        noise = torch.normal(mean = torch.zeros(shape_tuple), std = perturb_amplitude*torch.ones(shape_tuple))
    return noise

def perturbation_at_critical_point(mu, mu_p, width, random_seed, savefolder, dataset_generation_func,\
    perturb_times = 5000, bins = 50, histogram_y_max = 200, Gaussian = False,
    not_perturb_dead = True, dead_ids = None, explanation=False):
    """ This function will perturb the parameters stored in a checkpoint and track the changes in loss 
    given the dataset. The results will be visualized in a histogram, and a dataframe containing the 
    perturbations leading to the negative loss changes will be returned

    Args:
        dataset_generation_func: the function generate the training data points for computing the loss
        bins (int, optional): histogram bin numbers. Defaults to 50.
        histogram_y_max (int, optional): histogram y max. Defaults to 200.

    Returns: a dataframe containing perturbation noises and resulting negative paths.
    """
    config = f"{mu:.3f}_{mu_p:.3f}_w{width}_rs{random_seed}"
    model_savepath =  os.path.join(savefolder,config,'model_optimizer','model_and_optimizer.pt')
    checkpoint = torch.load(model_savepath)

    net = two_layer_net(2, width, 1, 0, 0) # beta_1 and beta_2 are set to zero cuz it does not matter here
    net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"loss = {loss}")
    net.eval()
    x_data,y_data,dataset_name = dataset_generation_func(plot = False)
    

    print("perturbing")
    perturb_df = pd.DataFrame()
    assert (not_perturb_dead and dead_ids is not None) or not not_perturb_dead
    perturb_amplitude = get_perturb_params(net, width = width, not_perturb_dead = not_perturb_dead, dead_ids = dead_ids)
    print("perturb amplitude: ", perturb_amplitude)
    for i in tqdm(range(perturb_times)):
        perturbed_net, noise = get_perturbed_model(net,perturb_amplitude, width = width,\
            not_perturb_dead = not_perturb_dead, dead_ids=dead_ids ,Gaussian = Gaussian)
        output = perturbed_net(x_data)
        perturbed_loss = F.mse_loss(output.reshape(y_data.shape), y_data)
        loss_change = perturbed_loss - loss
        one_perturb_df = pd.DataFrame({'noise':[noise], 'loss_change': loss_change.detach().numpy()})
        perturb_df = perturb_df.append(one_perturb_df)

    perturb_df.hist(column = 'loss_change',bins=bins)
    loss_change_min = perturb_df['loss_change'].min()
    # plt.ylim([0,histogram_y_max])
    negative_perturb_df = perturb_df[perturb_df['loss_change'] < 0]
    negative_perturb_df_savedir = os.path.join(savefolder, config, 'perturb_retrain')
    os.makedirs(negative_perturb_df_savedir, exist_ok=True)
    if loss_change_min < 0:
        negative_perturb_df.to_pickle(os.path.join(negative_perturb_df_savedir,'negative_perturb_df.pkl'))
    negative_escape_path_number = (negative_perturb_df.shape)[0]
    if explanation:
        plt.figtext(0.1, -0.1,f"y_max is set to {histogram_y_max}"
                            "\n" f"minimal loss change is {loss_change_min}"
                            "\n"f"total perturbation attempt of {perturb_times} times"
                            "\n" f"acquired {negative_escape_path_number} path(s) to go down the loss surface")
    print(f'min loss change: {loss_change_min}')
    histogram_savepath = os.path.join(savefolder, config,'pics','loss change VS perturbation.jpg')
    plt.grid(False)
    plt.title('')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel('loss changes')
    plt.ylabel('occurences')
    plt.savefig(histogram_savepath, dpi = 300,bbox_inches = 'tight')
    return negative_perturb_df

    