

import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import sys
import gc
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import init

sys.path.append('../')
# DualFedAtt.
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated, initialize_parameters_zeros
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def map(x, min, max):
    x_min = np.min(x)
    x_max = np.max(x)
    return min + (max - min) / (x_max - x_min) * (x - x_min)

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    sample_cell = random.sample(list(selected_cells), 1)
    sample_cell = sample_cell[0]
    # sample_cell = 1557

    device = 'cuda' if args.gpu else 'cpu'

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)



    #初始化模型
    fed_global_model = LSTM(args).to(device)
    s_global_model = LSTM(args).to(device)
    GCAU_global_model = LSTM(args).to(device)
    sg_global_model = LSTM(args).to(device)
    global_model = LSTM(args).to(device)

    fed_global_model.train()
    s_global_model.train()
    GCAU_global_model.train()
    sg_global_model.train()
    global_model.train()

    fed_global_weights = fed_global_model.state_dict()
    s_global_weights = s_global_model.state_dict()
    GCAU_global_weights = GCAU_global_model.state_dict()
    sg_global_weights = sg_global_model.state_dict()
    global_weights = global_model.state_dict()


    #初始化控制变量
    s_control_global = LSTM(args).to(device)
    GCAU_control_global = LSTM(args).to(device)
    sg_control_global = LSTM(args).to(device)
    control_global = LSTM(args).to(device)

    s_control_weights = initialize_parameters_zeros(s_control_global)
    GCAU_control_weights = initialize_parameters_zeros(GCAU_control_global)
    sg_control_weights = initialize_parameters_zeros(sg_control_global)
    control_weights = control_global.state_dict()

    s_c_local = {}
    GCAU_c_local = {}
    GCAU_c_local_epoch = {}
    GCAU_w_locals = {}
    sg_c_local = {}
    c_local = {}
    c_local_epoch = {}
    w_locals = {}

    GCAU_weights = copy.deepcopy(GCAU_global_weights)
    weight = copy.deepcopy(global_weights)
    for w in GCAU_weights:
        GCAU_weights[w] = GCAU_global_weights[w] - GCAU_global_weights[w]
        weight[w] = global_weights[w] - global_weights[w]

    for idx in selected_cells:
        s_c_local[idx] = LSTM(args).to(device)
        s_c_local[idx].load_state_dict(s_control_weights)
        GCAU_c_local[idx] = LSTM(args).to(device)
        GCAU_c_local[idx].load_state_dict(GCAU_control_weights)
        GCAU_c_local_epoch[idx] = 0
        GCAU_w_locals[idx] = copy.deepcopy(GCAU_weights)
        sg_c_local[idx] = LSTM(args).to(device)
        sg_c_local[idx].load_state_dict(sg_control_weights)
        c_local[idx] = LSTM(args).to(device)
        c_local[idx].load_state_dict(control_weights)
        c_local_epoch[idx] = 0
        w_locals[idx] = copy.deepcopy(weight)

    s_delta_c = copy.deepcopy(s_control_global.state_dict())
    GCAU_delta_c = copy.deepcopy(GCAU_control_global.state_dict())
    sg_delta_c = copy.deepcopy(sg_control_global.state_dict())
    delta_c = copy.deepcopy(control_global.state_dict())


    fed_best_val_loss = None
    fed_val_loss = []
    fed_val_acc = []
    fed_cell_loss = []
    fed_loss_hist = []
    fed_acc_hist = []
    best_fed_epoch = 0
    best_fed_mse = 0.0
    fed_local_traffic = 0.0

    s_best_val_loss = None
    s_val_loss = []
    s_val_acc = []
    s_cell_loss = []
    s_loss_hist = []
    s_acc_hist = []
    best_s_epoch = 0
    best_s_mse = 0.0
    s_local_traffic = 0.0

    GCAU_best_val_loss = None
    GCAU_val_loss = []
    GCAU_val_acc = []
    GCAU_cell_loss = []
    GCAU_loss_hist = []
    GCAU_acc_hist = []
    best_GCAU_epoch = 0
    best_GCAU_mse = 0.0

    sg_best_val_loss = None
    sg_val_loss = []
    sg_val_acc = []
    sg_cell_loss = []
    sg_loss_hist = []
    sg_acc_hist = []
    best_sg_epoch = 0
    best_sg_mse = 0.0

    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []
    epoch_pearson_dgc = []
    best_epoch = 0
    best_mse = 0.0
    accuracy_hist = []


    for epoch in tqdm.tqdm(range(args.epochs)):
        fed_local_weights, fed_local_losses = [], []
        s_local_weights, s_local_losses = [], []
        GCAU_local_weights, GCAU_local_losses = [], []
        GCAU_local_pear_data = []
        sg_local_weights, sg_local_losses = [], []
        sg_local_pear_data = []
        local_weights, local_losses = [], []
        local_grad_np_dgc = []
        local_indices = []
        local_values = []
        local_shape = []

        for i in s_delta_c:
            s_delta_c[i] = 0.0
            GCAU_delta_c[i] = 0.0
            sg_delta_c[i] = 0.0
            delta_c[i] = 0.0

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            fed_global_model.load_state_dict(fed_global_weights)
            fed_global_model.train()
            s_global_model.load_state_dict(s_global_weights)
            s_global_model.train()
            GCAU_global_model.load_state_dict(GCAU_global_weights)
            GCAU_global_model.train()
            sg_global_model.load_state_dict(sg_global_weights)
            sg_global_model.train()
            global_model.load_state_dict(global_weights)
            global_model.train()

            fed_w, fed_loss, fed_epoch_loss = local_model.update_weights_fed(model=copy.deepcopy(fed_global_model),
                                                                             global_round=epoch)

            s_w, s_loss, s_epoch_loss, s_local_delta_c, s_local_delta, s_control_local_w, = local_model.update_weights_scaffold(
                model=copy.deepcopy(s_global_model), control_local=copy.deepcopy(s_c_local[cell]),
                control_global=copy.deepcopy(s_control_global))

            GCAU_w, GCAU_loss, GCAU_epoch_loss, GCAU_local_delta_c, GCAU_local_delta, GCAU_control_local_w, GCAU_ww, \
            GCAU_c_local_epoch[cell] = local_model.update_weights_SGCAU(model=copy.deepcopy(GCAU_global_model),
                                                                       global_round=epoch,
                                                                       control_local=GCAU_c_local[cell],
                                                                       control_global=GCAU_control_global,
                                                                       w_loc=GCAU_w_locals[cell],
                                                                       c_local_epoch=GCAU_c_local_epoch[cell] + 1)

            sg_w, sg_loss, sg_epoch_loss, sg_local_delta_c, sg_local_delta, sg_control_local_w, sg_pear_data = local_model.update_weights_sg(
                model=copy.deepcopy(sg_global_model), global_round=epoch, control_local=sg_c_local[cell],
                control_global=sg_control_global, )

            w, loss, epoch_loss, local_delta_c, local_delta, control_local_w, ww, grad_np_dgc, c_local_epoch[
                cell], traffic, values, indices, shape = local_model.update_weights_test(
                model=copy.deepcopy(global_model),
                global_round=epoch, control_local=c_local[cell],
                control_global=copy.deepcopy(control_global),
                w_loc=w_locals[cell],
                c_local_epoch=c_local_epoch[cell] + 1)

            s_local_traffic += 1

            s_c_local[cell].load_state_dict(s_control_local_w)
            GCAU_c_local[cell].load_state_dict(GCAU_control_local_w)
            sg_c_local[cell].load_state_dict(sg_control_local_w)
            c_local[cell].load_state_dict(control_local_w)


            # local_weights.append(copy.deepcopy(w))
            fed_local_weights.append(copy.deepcopy(fed_w))
            fed_local_losses.append(copy.deepcopy(fed_loss))
            fed_cell_loss.append(fed_loss)
            s_local_weights.append(copy.deepcopy(s_w))
            s_local_losses.append(copy.deepcopy(s_loss))
            s_cell_loss.append(s_loss)
            GCAU_local_weights.append(copy.deepcopy(GCAU_local_delta))
            GCAU_local_losses.append(copy.deepcopy(GCAU_loss))
            GCAU_cell_loss.append(GCAU_loss)
            sg_local_weights.append(copy.deepcopy(sg_w))
            sg_local_losses.append(copy.deepcopy(sg_loss))
            sg_cell_loss.append(sg_loss)
            sg_local_pear_data.append(sg_pear_data)
            local_weights.append(copy.deepcopy(local_delta))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)
            local_grad_np_dgc.append(grad_np_dgc)
            local_values.append(values)
            local_indices.append(indices)
            local_shape.append(shape)

            for i in s_delta_c:
                s_delta_c[i] += s_local_delta_c[i]
                GCAU_delta_c[i] += GCAU_local_delta_c[i]
                sg_delta_c[i] += sg_local_delta_c[i]
                delta_c[i] += local_delta_c[i]
            for k in GCAU_ww:
                GCAU_w_locals[cell][k] = GCAU_ww[k]
                w_locals[cell][k] = ww[k]

        fed_loss_hist.append(sum(fed_cell_loss) / len(fed_cell_loss))
        s_loss_hist.append(sum(s_cell_loss) / len(s_cell_loss))
        GCAU_loss_hist.append(sum(GCAU_cell_loss) / len(GCAU_cell_loss))
        sg_loss_hist.append(sum(sg_cell_loss) / len(sg_cell_loss))
        loss_hist.append(sum(cell_loss) / len(cell_loss))

        for i in s_delta_c:
            s_delta_c[i] /= m
            GCAU_delta_c[i] /= m
            sg_delta_c[i] /= m
            delta_c[i] /= m


        #pearson
        sg_pearson = []
        global_grad_np_dgc = []
        # agg_weight = []
        div = np.zeros_like(grad_np_dgc)
        for l in range(len(sg_local_pear_data)):
            data = sg_local_pear_data[l]
            sg_pearson = np.concatenate([sg_pearson, sg_local_pear_data[l]])
            global_grad_np_dgc = np.concatenate([global_grad_np_dgc, local_grad_np_dgc[l]])
            div += (local_grad_np_dgc[l] != 0)
        sg_pearson = sg_pearson.reshape(m, -1)
        sg_pearson = np.corrcoef(sg_pearson)
        # global_grad_np_dgc = np.corrcoef(global_grad_np_dgc.reshape(m, -1))
        div = (torch.clamp(torch.tensor(div), 1., len(local_grad_np_dgc))).numpy()
        global_grad_np_dgc = global_grad_np_dgc.reshape(m, -1)
        # for l in range(len(global_grad_np_dgc)):
        #     agg_weight.append(sum(global_grad_np_dgc[l]))
        # agg_weight = torch.sigmoid(torch.tensor(agg_weight))
        #weighted


        #########################################
        # # new_global_np_dgc = np.dot(np.corrcoef(global_grad_np_dgc), global_grad_np_dgc)
        # new_pearson = np.dot(global_grad_np_dgc, local_grad_np_dgc)
        # # new_pearson = map(new_pearson, np.min(local_values), np.max(local_values))
        # d_weight = copy.deepcopy(global_weights)
        # for w in d_weight:
        #     d_weight[w] -= d_weight[w]
        # div = copy.deepcopy(d_weight)
        # tally = {}
        # for l in range(len(local_values)):
        #     tally[l] = 0
        #     for w in d_weight:
        #         a = torch.zeros_like(d_weight[w]).view(-1)
        #         indice = torch.tensor(local_indices[l][tally[l]:tally[l]+local_shape[l][w]]).type(torch.int64)
        #         value = torch.tensor(new_pearson[l][indice]).type(torch.float32)
        #         a[indice] = value
        #         tally[l] += local_shape[l][w]
        #         a = torch.reshape(a, (d_weight[w].shape))
        #         div[w] += (a != 0)
        #         d_weight[w] += a
        # for w in d_weight:
        #     div[w] = (torch.clamp(div[w], 1., len(local_values)))
        #     d_weight[w] = (d_weight[w] / div[w])
        #########################################

        sg_global = np.dot(sg_pearson, sg_local_pear_data)
        sg_a = np.zeros_like(sg_pear_data)
        new_global_np_dgc = np.dot(np.corrcoef(global_grad_np_dgc), global_grad_np_dgc)
        # new_global_np_dgc = []
        # for l in range(len(agg_weight)):
        #     new_global_np_dgc.append(local_grad_np_dgc[l] * agg_weight[l].numpy())
        a = np.zeros_like(grad_np_dgc)
        for i in range(len(sg_global)):
            sg_a += sg_global[i]
            a += new_global_np_dgc[l]
        sg_a /= m
        a = a / div
        # a = a/m
        new_sg_global_weights = {}
        new_global_weight = {}
        sg_b = []
        b = []
        count = 0
        lay_shape = {}
        for k in sg_global_weights:
            lay_shape[k] = torch.ones(1)
            for i in range(len(sg_global_weights[k].shape)):
                lay_shape[k] *= sg_global_weights[k].shape[i]
        for k in lay_shape:
            sg_c = int(lay_shape[k].item())
            sg_b = sg_a[count:count + sg_c]
            sg_d = torch.from_numpy(sg_b)
            new_sg_global_weights[k] = sg_d.reshape(sg_global_weights[k].shape)
            c = int(lay_shape[k].item())
            b = a[count:count + c]
            d = torch.from_numpy(b)
            new_global_weight[k] = d.reshape(weight[k].shape)
            count += sg_c
        ##############################################



        # Update global model
        fed_global_weights = average_weights(fed_local_weights)
        s_global_weights = average_weights(s_local_weights)
        GCAU_global_weights = copy.deepcopy(GCAU_global_model.state_dict())
        for k in new_sg_global_weights:
            sg_global_weights[k] = new_sg_global_weights[k] + sg_global_weights[k]
            GCAU_global_weights[k] += average_weights(GCAU_local_weights)[k]
            global_weights[k] += new_global_weight[k]

        fed_global_model.load_state_dict(fed_global_weights)
        s_global_model.load_state_dict(s_global_weights)
        GCAU_global_model.load_state_dict(GCAU_global_weights)
        sg_global_model.load_state_dict(sg_global_weights)
        global_model.load_state_dict(global_weights)


        #Update global c
        s_control_global_w = copy.deepcopy(s_control_global.state_dict())
        GCAU_control_global_w = copy.deepcopy(GCAU_control_global.state_dict())
        sg_control_global_w = copy.deepcopy(sg_control_global.state_dict())
        control_global_w = s_control_global.state_dict()
        for i in control_global_w:
            s_control_global_w[i] += (m / len(selected_cells)) * s_delta_c[i]
            GCAU_control_global_w[i] += (m / len(selected_cells)) * GCAU_delta_c[i]
            sg_control_global_w[i] += (m / len(selected_cells)) * sg_delta_c[i]
            control_weights[i] += (m / len(selected_cells)) * delta_c[i]
            # control_global_w[i] += (m / args.num_users) * delta_c[i]
        s_control_global.load_state_dict(s_control_global_w)
        GCAU_control_global.load_state_dict(GCAU_control_global_w)
        sg_control_global.load_state_dict(sg_control_global_w)
        control_global.load_state_dict(control_weights)

        # Test
        fed_pred, fed_truth = {}, {}
        fed_test_loss_list = []
        fed_test_mse_list = []
        fed_nrmse = 0.0
        s_pred, s_truth = {}, {}
        s_test_loss_list = []
        s_test_mse_list = []
        s_nrmse = 0.0
        GCAU_pred, GCAU_truth = {}, {}
        GCAU_test_loss_list = []
        GCAU_test_mse_list = []
        GCAU_nrmse = 0.0
        sg_pred, sg_truth = {}, {}
        sg_test_loss_list = []
        sg_test_mse_list = []
        sg_nrmse = 0.0
        pred, truth = {}, {}
        test_loss_list = []
        test_mse_list = []
        nrmse = 0.0

        for cell in selected_cells:
            cell_test = test[cell]
            fed_test_loss, fed_test_mse, fed_test_nrmse, fed_pred[cell], fed_truth[cell] = test_inference(args,
                                                                                                          fed_global_model,
                                                                                                          cell_test)
            fed_nrmse += fed_test_nrmse

            s_test_loss, s_test_mse, s_test_nrmse, s_pred[cell], s_truth[cell] = test_inference(args, s_global_model,
                                                                                                cell_test)
            s_nrmse += s_test_nrmse

            GCAU_test_loss, GCAU_test_mse, GCAU_test_nrmse, GCAU_pred[cell], GCAU_truth[cell] = test_inference(args,
                                                                                                               GCAU_global_model,
                                                                                                               cell_test)
            GCAU_nrmse += GCAU_test_nrmse

            sg_test_loss, sg_test_mse, sg_test_nrmse, sg_pred[cell], sg_truth[cell] = test_inference(args,
                                                                                                     sg_global_model,
                                                                                                     cell_test)
            sg_nrmse += sg_test_nrmse

            test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
            nrmse += test_nrmse

            gc.collect()
            torch.cuda.empty_cache()



        fed_df_pred = pd.DataFrame.from_dict(fed_pred)
        fed_df_truth = pd.DataFrame.from_dict(fed_truth)
        s_df_pred = pd.DataFrame.from_dict(s_pred)
        s_df_truth = pd.DataFrame.from_dict(s_truth)
        GCAU_df_pred = pd.DataFrame.from_dict(GCAU_pred)
        GCAU_df_truth = pd.DataFrame.from_dict(GCAU_truth)
        sg_df_pred = pd.DataFrame.from_dict(sg_pred)
        sg_df_truth = pd.DataFrame.from_dict(sg_truth)
        df_pred = pd.DataFrame.from_dict(pred)
        df_truth = pd.DataFrame.from_dict(truth)

        fed_mse = metrics.mean_squared_error(fed_df_pred.values.ravel(), fed_df_truth.values.ravel())
        fed_mae = metrics.mean_absolute_error(fed_df_pred.values.ravel(), fed_df_truth.values.ravel())
        fed_nrmse = fed_nrmse / len(selected_cells)
        s_mse = metrics.mean_squared_error(s_df_pred.values.ravel(), s_df_truth.values.ravel())
        s_mae = metrics.mean_absolute_error(s_df_pred.values.ravel(), s_df_truth.values.ravel())
        s_nrmse = s_nrmse / len(selected_cells)
        GCAU_mse = metrics.mean_squared_error(GCAU_df_pred.values.ravel(), GCAU_df_truth.values.ravel())
        GCAU_mae = metrics.mean_absolute_error(GCAU_df_pred.values.ravel(), GCAU_df_truth.values.ravel())
        GCAU_nrmse = GCAU_nrmse / len(selected_cells)
        sg_mse = metrics.mean_squared_error(sg_df_pred.values.ravel(), sg_df_truth.values.ravel())
        sg_mae = metrics.mean_absolute_error(sg_df_pred.values.ravel(), sg_df_truth.values.ravel())
        sg_nrmse = sg_nrmse / len(selected_cells)
        mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
        mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
        nrmse = nrmse / len(selected_cells)

        fed_acc_hist.append(fed_mse)
        s_acc_hist.append(s_mse)
        sg_acc_hist.append(sg_mse)
        GCAU_acc_hist.append(GCAU_mse)
        accuracy_hist.append(mse)
        if epoch == 0:
            best_fed_mse = fed_mse
            best_s_mse = s_mse
            best_GCAU_mse = GCAU_mse
            best_sg_mse = sg_mse
            best_mse = mse
        if best_fed_mse > fed_mse:
            best_fed_mse = fed_mse
            best_fed_epoch = epoch
        if best_s_mse > s_mse:
            best_s_mse = s_mse
            best_s_epoch = epoch
        if best_GCAU_mse > GCAU_mse:
            best_GCAU_mse = GCAU_mse
            best_GCAU_epoch = epoch
        if best_sg_mse > sg_mse:
            best_sg_mse = sg_mse
            best_sg_epoch = epoch
        if best_mse > mse:
            best_mse = mse
            best_epoch = epoch

        print('\nEpoch:{:}\n'.format(epoch))
        print(
            'FedAvg:  Best_fed_epoch:{:} Best_fed_mse:{:.4f} Current_fed_mse:{:.4f} Current_fed_mae:{:.4f} Current_fed_nrmse:{:.4f}\n'.format(
                best_fed_epoch,
                best_fed_mse, fed_mse, fed_mae, fed_nrmse))
        print(
            'Scaffold:  Best_s_epoch:{:} Best_s_mse:{:.4f} Current_s_mse:{:.4f} Current_s_mae:{:.4f} Current_s_nrmse:{:.4f} Traffic:{:.4f}\n'.format(
                best_s_epoch, best_s_mse,
                s_mse, s_mae, s_nrmse, s_local_traffic))
        print(
            'Scaffold-GCAU:  Best_GCAU_epoch:{:} Best_GCAU_mse:{:.4f} Current_GCAU_mse:{:.4f} Current_GCAU_mae:{:.4f} Current_GCAU_nrmse:{:.4f}\n'.format(
                best_GCAU_epoch, best_GCAU_mse, GCAU_mse, GCAU_mae, GCAU_nrmse))
        print(
            'Scaffold-G:  Best_sg_epoch:{:} Best_sg_mse:{:.4f} Current_sg_mse:{:.4f} Current_sg_mae:{:.4f} Current_sg_nrmse:{:.4f}\n'.format(
                best_sg_epoch, best_sg_mse,
                sg_mse, sg_mae, sg_nrmse))
        print(
            'GCAU_TPS_Scaffold:  Best_epoch:{:} Best_mse:{:.4f} Current_mse:{:.4f} Current_mae:{:.4f} Current_nrmse:{:.4f}\n'.format(
                best_epoch,
                best_mse,
                mse, mae, nrmse))


    # Test model accuracy
    fed_pred, fed_truth = {}, {}
    fed_test_loss_list = []
    fed_test_mse_list = []
    fed_nrmse = 0.0
    fed_histogram_value = []
    fed_absolute_error = []

    s_pred, s_truth = {}, {}
    s_test_loss_list = []
    s_test_mse_list = []
    s_nrmse = 0.0
    s_histogram_value = []
    s_absolute_error = []

    GCAU_pred, GCAU_truth = {}, {}
    GCAU_test_loss_list = []
    GCAU_test_mse_list = []
    GCAU_nrmse = 0.0

    sg_pred, sg_truth = {}, {}
    sg_test_loss_list = []
    sg_test_mse_list = []
    sg_nrmse = 0.0

    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0



    for cell in selected_cells:
        cell_test = test[cell]
        fed_test_loss, fed_test_mse, fed_test_nrmse, fed_pred[cell], fed_truth[cell] = test_inference(args, fed_global_model, cell_test)
        fed_nrmse += fed_test_nrmse

        s_test_loss, s_test_mse, s_test_nrmse, s_pred[cell], s_truth[cell] = test_inference(args, s_global_model, cell_test)
        s_nrmse += s_test_nrmse

        GCAU_test_loss, GCAU_test_mse, GCAU_test_nrmse, GCAU_pred[cell], GCAU_truth[cell] = test_inference(args, GCAU_global_model, cell_test)
        GCAU_nrmse += GCAU_test_nrmse

        sg_test_loss, sg_test_mse, sg_test_nrmse, sg_pred[cell], sg_truth[cell] = test_inference(args, sg_global_model,cell_test)
        sg_nrmse += sg_test_nrmse

        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
        nrmse += test_nrmse

        fed_test_loss_list.append(fed_test_loss)
        fed_test_mse_list.append(fed_test_mse)
        fed_histogram_value.append(fed_truth[cell] - fed_pred[cell])
        s_test_loss_list.append(s_test_loss)
        s_test_mse_list.append(s_test_mse)
        s_histogram_value.append(s_truth[cell] - s_pred[cell])
        GCAU_test_loss_list.append(GCAU_test_loss)
        GCAU_test_mse_list.append(GCAU_test_mse)
        sg_test_loss_list.append(sg_test_loss)
        sg_test_mse_list.append(sg_test_mse)
        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

        gc.collect()
        torch.cuda.empty_cache()

    fed_df_pred = pd.DataFrame.from_dict(fed_pred)
    fed_df_truth = pd.DataFrame.from_dict(fed_truth)
    s_df_pred = pd.DataFrame.from_dict(s_pred)
    s_df_truth = pd.DataFrame.from_dict(s_truth)
    GCAU_df_pred = pd.DataFrame.from_dict(GCAU_pred)
    GCAU_df_truth = pd.DataFrame.from_dict(GCAU_truth)
    sg_df_pred = pd.DataFrame.from_dict(sg_pred)
    sg_df_truth = pd.DataFrame.from_dict(sg_truth)
    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    # # 计算绝对误差累积分布函数
    # fed_errors = fed_truth[sample_cell] - fed_pred[sample_cell]
    # np.savetxt('Fedavg-absolute_error_milano.txt', fed_errors, fmt='%.4f')
    # fed_ecdf = np.cumsum(np.histogram(fed_errors, bins=100, density=True)[0])
    # fed_absolute_error.append(fed_errors)
    # fed_absolute_error.append(fed_ecdf)
    # np.savetxt('Fedavg-absolute_error.txt', np.array(fed_absolute_error))
    #
    # s_errors = s_truth[sample_cell] - s_pred[sample_cell]
    # np.savetxt('Scaffold-absolute_error_milano.txt', s_errors, fmt='%.4f')
    # s_ecdf = np.cumsum(np.histogram(s_errors, bins=100, density=True)[0])
    # s_absolute_error.append(s_errors)
    # s_absolute_error.append(s_ecdf)
    # np.savetxt('Scaffold-absolute_error.txt', np.array(s_absolute_error))


    # np.savetxt('Scaffold-error_milano.txt', np.array(s_histogram_value).reshape(-1, 1), fmt='%.4f')

    # np.savetxt('Fedavg-error_milano.txt', np.array(fed_histogram_value).reshape(-1, 1), fmt='%.4f')

    fed_mse = metrics.mean_squared_error(fed_df_pred.values.ravel(), fed_df_truth.values.ravel())
    fed_mae = metrics.mean_absolute_error(fed_df_pred.values.ravel(), fed_df_truth.values.ravel())
    fed_nrmse = fed_nrmse / len(selected_cells)
    print('FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, fed_mse,
                                                                                     fed_mae,
                                                                                     fed_nrmse))
    s_mse = metrics.mean_squared_error(s_df_pred.values.ravel(), s_df_truth.values.ravel())
    s_mae = metrics.mean_absolute_error(s_df_pred.values.ravel(), s_df_truth.values.ravel())
    s_nrmse = s_nrmse / len(selected_cells)
    print(
        'Scaffold File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, s_mse, s_mae,
                                                                                     s_nrmse))
    GCAU_mse = metrics.mean_squared_error(GCAU_df_pred.values.ravel(), GCAU_df_truth.values.ravel())
    GCAU_mae = metrics.mean_absolute_error(GCAU_df_pred.values.ravel(), GCAU_df_truth.values.ravel())
    GCAU_nrmse = GCAU_nrmse / len(selected_cells)
    print(
        'Scaffold_GCAU File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type,
                                                                                           GCAU_mse, GCAU_mae,
                                                                                           GCAU_nrmse))
    sg_mse = metrics.mean_squared_error(sg_df_pred.values.ravel(), sg_df_truth.values.ravel())
    sg_mae = metrics.mean_absolute_error(sg_df_pred.values.ravel(), sg_df_truth.values.ravel())
    sg_nrmse = sg_nrmse / len(selected_cells)
    print(
        'Scaffold_G File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, sg_mse, sg_mae,
                                                                                     sg_nrmse))
    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print(
        'GCAU_TPS_Scaffold File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse,
                                                                                              mae, nrmse))
    # picture loss
    plt.figure()
    plt.plot(range(len(fed_loss_hist)), fed_loss_hist, label='FedAvg')
    plt.plot(range(len(s_loss_hist)), s_loss_hist, label='Scaffold')
    plt.plot(range(len(GCAU_loss_hist)), GCAU_loss_hist, label='Scaffold-GCAU')
    plt.plot(range(len(sg_loss_hist)), sg_loss_hist, label='Scaffold-G')
    # plt.plot(range(len(loss_hist)), loss_hist, label='GCAU_TPS_Scaffold')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/FedAvg_Scaffold_GCAU_ScaffoldG_GCAUTPS_{}_{}.png'.format(args.file, 'train_loss'))
    #
    # # picture acc
    plt.figure()
    plt.plot(range(len(fed_acc_hist)), fed_acc_hist, label='FedAvg')
    plt.plot(range(len(s_acc_hist)), s_acc_hist, label='Scaffold')
    plt.plot(range(len(GCAU_acc_hist)), GCAU_acc_hist, label='Scaffold-GCAU')
    plt.plot(range(len(sg_acc_hist)), sg_acc_hist, label='Scaffold-G')
    # plt.plot(range(len(accuracy_hist)), accuracy_hist, label='GCAU_TPS_Scaffold')
    plt.ylabel('train_acc')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/FedAvg_Scaffold_GCAU_ScaffoldG_GCAUTPS_{}_{}.png'.format(args.file, 'train_acc'))