


import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import sys
import random
import itertools
import matplotlib.pyplot as plt
import gc

sys.path.append('../')
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference, test_accuracy
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_lay_shape(net):
    lay_shape = {}
    for k in net:
        lay_shape[k] = torch.ones(1)
        for i in range(len(net[k].shape)):
            lay_shape[k] *= net[k].shape[i]
    return lay_shape

def get_pearson_lay(data, net, lay_shape):
    # div = np.zeros_like(data[0])
    div = {}
    global_d_weight = {}
    pearson = {}

    count = 0
    for k in lay_shape:
        lay_count = int(lay_shape[k].item())
        # global_d_weight = []
        for l in range(len(data)):
            weight = data[l][count:count + lay_count]
            if l == 0:
                div[k] = np.zeros_like(data[l][count:count + lay_count])
                global_d_weight[k] = []
                pearson[k] = []
            global_d_weight[k] = np.concatenate([global_d_weight[k], weight])
            div[k] += (data[l][count:count + lay_count] != 0)

            if l == len(data) - 1:
                global_d_weight[k] = global_d_weight[k].reshape(m, -1)
                pearson[k].append(np.corrcoef(global_d_weight[k]))
                div[k] = (torch.clamp(torch.tensor(div[k]), 1., len(data))).numpy()
        count += lay_count
    # div = (torch.clamp(torch.tensor(div), 1., len(data))).numpy()
    # global_d_weight = global_d_weight.reshape(m, -1)

    # pearson = np.corrcoef(global_d_weight)
    return pearson, div, global_d_weight

def Aggregate_lay(pearson, global_d_weight, lay_shape, net):
    new_global_np_dgc = {}
    data = {}
    weight = {}
    for k in pearson:
        new_global_np_dgc[k] = np.dot(pearson[k], global_d_weight[k]).reshape(m, -1)
        data[k] = np.zeros_like(new_global_np_dgc[k][0])
        for l in range(len(new_global_np_dgc[k])):
            data[k] += new_global_np_dgc[k][l]
        data[k] = data[k] / div[k]
        weight[k] = torch.tensor(data[k].reshape(net[k].shape))
        net[k] += weight[k]
    # new_global_np_dgc = np.dot(np.corrcoef(global_d_weight), global_d_weight)
    # data = np.zeros_like(new_global_np_dgc[0])
    # for l in range(len(pearson)):
    #     data += new_global_np_dgc[l]
    # data = data / div
    # weight = {}
    # count = 0
    # for k in lay_shape:
    #     lay_count = int(lay_shape[k].item())
    #     weight[k] = torch.from_numpy(data[count:count + lay_count]).reshape(net[k].shape)
    #     count += lay_count
    #     net[k] += weight[k]
    return net

def get_pearson(data):
    global_d_weight = []
    div = np.zeros_like(data[0])
    for l in range(len(data)):
        global_d_weight = np.concatenate([global_d_weight, data[l]])
        div += (data[l] != 0)
    div = (torch.clamp(torch.tensor(div), 1., len(data))).numpy()
    global_d_weight = global_d_weight.reshape(m, -1)
    pearson = np.corrcoef(global_d_weight)
    return pearson, div, global_d_weight

def Aggregate(pearson, global_d_weight, lay_shape, net):
    new_global_np_dgc = np.dot(np.corrcoef(global_d_weight), global_d_weight)
    data = np.zeros_like(new_global_np_dgc[0])
    for l in range(len(pearson)):
        data += new_global_np_dgc[l]
    data = data / div
    weight = {}
    count = 0
    for k in lay_shape:
        lay_count = int(lay_shape[k].item())
        weight[k] = torch.from_numpy(data[count:count + lay_count]).reshape(net[k].shape)
        count += lay_count
        net[k] += weight[k]
    return net

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    sample_cell = random.sample(list(selected_cells), 1)
    sample_cell = sample_cell[0]

    device = 'cuda' if args.gpu else 'cpu'

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)

    #Local update count
    c_local_epoch = {}

    #Initialize global model
    global_model = LSTM(args).to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    #Initialize control variables
    control_global = LSTM(args).to(device)
    control_weights = control_global.state_dict()
    c_local = {}
    delta_c = copy.deepcopy(global_model.state_dict())
    for idx in selected_cells:
        c_local[idx] = LSTM(args).to(device)
        c_local[idx].load_state_dict(control_weights)
        c_local_epoch[idx] = 0

    #Initialize local gradient accumulation
    w_locals = {}
    weight = copy.deepcopy(global_weights)
    for w in global_weights:
        weight[w] = global_weights[w] - global_weights[w]
    for idx in selected_cells:
        w_locals[idx] = copy.deepcopy(weight)

    #network shape
    lay_shape = get_lay_shape(weight)

    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []
    local_traffic = 0
    best_epoch = 0
    best_mse = 0.0
    accuracy_hist = []


    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_d_weight = []
        for k in delta_c:
            delta_c[k] = 0.0
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            global_model.load_state_dict(global_weights)
            global_model.train()

            w, loss, epoch_loss, local_delta_c, control_local_w, ww, d_weight, c_local_epoch[
                cell], traffic = local_model.update_weights_GCAU_TPS_Scaffold(model=copy.deepcopy(global_model),
                                                                      global_round=epoch, control_local=c_local[cell],
                                                                      control_global=copy.deepcopy(control_global),
                                                                      w_loc=w_locals[cell],
                                                                      c_local_epoch=c_local_epoch[cell] + 1)

            if epoch != 0:
                c_local[cell].load_state_dict(control_local_w)

            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)
            local_d_weight.append(d_weight)
            local_traffic += traffic

            #Collect changes of local control variables, update local gradient accumulation
            for k in delta_c:
                delta_c[k] += local_delta_c[k]
                w_locals[cell][k] = ww[k]
        loss_hist.append(sum(cell_loss) / len(cell_loss))

        # Calculate Pearson correlation coefficient
        pearson, div, global_d_weight = get_pearson(local_d_weight)

        # Aggregate
        new_global_weight = Aggregate(pearson, global_d_weight, lay_shape, copy.deepcopy(global_weights))

        # Update global model
        global_model.load_state_dict(new_global_weight)

        #Update global control variables
        for k in control_weights:
            delta_c[k] /= m
            control_weights[k] += (m / len(selected_cells)) * delta_c[k]
        control_global.load_state_dict(control_weights)


        #Test
        pred, truth = {}, {}
        test_loss_list = []
        test_mse_list = []
        histogram_value = []
        nrmse = 0.0
        avg_pred, avg_truth = 0, 0

        for cell in selected_cells:
            cell_test = test[cell]
            test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
            nrmse += test_nrmse
            avg_pred += pred[cell]
            avg_truth += truth[cell]

            test_loss_list.append(test_loss)
            test_mse_list.append(test_mse)

            histogram_value.append(truth[cell] - pred[cell])

            gc.collect()
            torch.cuda.empty_cache()

        df_pred = pd.DataFrame.from_dict(pred)
        df_truth = pd.DataFrame.from_dict(truth)
        avg_pred /= len(selected_cells)
        avg_truth /= len(selected_cells)




        mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
        mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
        nrmse = nrmse / len(selected_cells)
        accuracy_hist.append(mse)
        if epoch == 0:
            best_mse = mse
        if best_mse > mse:
            best_mse = mse
            best_epoch = epoch

        print('Epoch:{:} Best_epoch:{:} Best_mse:{:.4f} Current_mse:{:.4f} Current_mae:{:.4f} Traffic:{:.4f}'.format(epoch, best_epoch, best_mse, mse, mae, local_traffic))

    # np.savetxt('FedGC-error_trento.txt', np.array(histogram_value).reshape(-1, 1), fmt='%.4f')
    # # 计算绝对误差累积分布函数
    # fed_errors = truth[sample_cell] - pred[sample_cell]
    # np.savetxt('FedGC-absolute_error_trento.txt', fed_errors, fmt='%.4f')
    # print('GCAU_TPS_Scaffold File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse,
    #                                                                                           mae, nrmse))
    # picture loss
    plt.figure()
    plt.plot(range(len(loss_hist)), loss_hist, label='GCAU_TPS_Scaffold')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/GCAU_TPS_Scaffold_{}_{}.png'.format(args.file, 'train_loss'))

    # picture mse
    # plt.figure()
    # plt.plot(range(len(accuracy_hist)), accuracy_hist, label='GCAU_TPS_Scaffold')
    # plt.ylabel('test_mse')
    # plt.xlabel('epochs')
    # plt.legend(loc='best')
    # plt.savefig(
    #     './save/GCAU_TPS_Scaffold_{}.png'.format('train_acc'))

    # picture Forecast comparison
    # plt.figure()
    # plt.plot(range(len(pred[278])), pred[278], label='Predicted value')
    # plt.plot(range(len(truth[278])), truth[278], label='True value')
    # plt.plot(range(len(avg_pred)), avg_pred, label='Predicted value')
    # plt.plot(range(len(avg_truth)), avg_truth, label='True value')
    # plt.ylabel('Traffic')
    # plt.xlabel('Time index')
    # plt.legend(loc='best')
    # plt.savefig(
    #     './save/GCAU_TPS_Scaffold_{}.png'.format('Forecast comparison'))

