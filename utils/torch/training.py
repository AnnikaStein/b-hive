import math
import time
import gc

import numpy as np
import torch
import torch.nn as nn
from utils.torch.ranger import Ranger
from utils.torch.definitions_ParT import epsilons_per_feature, vars_per_candidate, defaults_per_variable
from utils.torch.attacks_ParT import *
from utils.torch.Cosine_LR import CosineAnnealingWarmupRestarts
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def perform_training(model, training_data, validation_data, directory, device, **kwargs):
    print("Warming up the training :")
    compiled = kwargs["compiled"]
    if compiled:
        model_c = torch.compile(model)
    else:
        model_c = model

    best_loss_val = math.inf
    #optimizer = torch.optim.AdamW(model_c.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.001)
    optimizer = torch.optim.RAdam(model_c.parameters(), lr=1e-3, betas=(0.95, 0.999), eps=1e-06)
    #optimizer = Ranger(model_c.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    nepochs = kwargs["nepochs"]
    FP16 = kwargs["FP16"]
    adv = kwargs["adv"]
    ParT = kwargs["ParT"]
    scaler = kwargs["scaler"]
    scheduling = kwargs["scheduler"]
    feature_edges = kwargs["feature_edges"]
    train_metrics = np.zeros((nepochs, 2))
    validation_metrics = np.zeros((nepochs, 2))
    scheduler = None
    batch_lr = False

    epsilon_factors = {'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(device),
                       'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(device),
                       'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(device),
                       'cpf_pts' : torch.Tensor(np.load(epsilons_per_feature['cpf_pts']).transpose()).to(device),
                       'npf_pts' : torch.Tensor(np.load(epsilons_per_feature['npf_pts']).transpose()).to(device),
                       'vtx_pts' : torch.Tensor(np.load(epsilons_per_feature['vtx_pts']).transpose()).to(device)
    }

    defaults_device = defaults_per_variable

    if scheduling == 'epoch_lin_decay':
        lr_epochs = max(1, int(nepochs * 0.3))
        lr_rate = 0.01 ** (1.0 / lr_epochs)
        mil = list(range(nepochs - lr_epochs, nepochs))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = mil, gamma = lr_rate)

    if scheduling == 'batch_lin_decay':
        batch_lr = True
        nsteps = get_num_steps(training_data)*nepochs
        lr_epochs = max(1, int(nsteps * 0.3))
        lr_rate = 0.01 ** (1.0 / lr_epochs)
        mil = list(range(nsteps - lr_epochs, nsteps))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = mil, gamma = lr_rate)

    if scheduling == 'batch_cosine_warmup':
        batch_lr = True
        nsteps = get_num_steps(training_data)*nepochs
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=nsteps, max_lr = 1e-3, min_lr = 1e-5, warmup_steps = int(nsteps*(1/nepochs)))


    for t in range(nepochs):
        print("Epoch", t + 1, "of", nepochs)
        loss_train, acc_train = train_model(
            training_data,
            model_c,
            loss_fn,
            optimizer,
            device,
            FP16,
            adv,
            ParT,
            scaler,
            scheduler,
            batch_lr,
            feature_edges,
            epsilon_factors,
            defaults_device            
        )

        if (not batch_lr) and (scheduler is not None):
            scheduler.step()

        train_metrics[t, :] = np.array([loss_train, acc_train])
        loss_val, acc_val = validate_model(validation_data, model_c, loss_fn, device, feature_edges = feature_edges, ParT = ParT)
        validation_metrics[t, :] = np.array([loss_val, acc_val])

        torch.save(
            {
                "epoch": t+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_train": loss_train,
                "acc_train": acc_train,
                "loss_val": loss_val,
                "acc_val": acc_val,
            },
            "{}/model_{}.pt".format(directory, t),
        )

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save(
                {
                    "epoch": t+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss_train,
                    "acc_train": acc_train,
                    "loss_val": loss_val,
                    "acc_val": acc_val,
                },
                "{}/best_model.pt".format(directory),
            )

    return train_metrics, validation_metrics


def train_model(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device="cpu",
    FP16=False,
    adv=False,
    ParT=False,
    scaler=None,
    scheduler=None,
    batch_lr=False,
    feature_edges=None,
    epsilons=None,
    default_device=None
):
    losses = 0
    losses2 = 0
    accuracy = 0.0
    model.train()

    with Progress(
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("0/? its"),
        expand=True,
    ) as progress:
        N = 0
        task = progress.add_task("Training...", total=dataloader.nits_expected)
        it = 0
        timer = 0
        start = time.time()
        for b, (x, w, y) in enumerate(dataloader):

            y = y.type(torch.LongTensor).to(device)

            start2 = time.time()
            if ParT:
                inpt = get_inpt(x.float().to(device), feature_edges, ret_glob = False)
            else:
                inpt = get_inpt(x.float().to(device), feature_edges)
                
            if FP16:
                with torch.cuda.amp.autocast():
                    if adv and (b % 3 == 0):
                        model.zero_grad()
                        minv = 0.1
                        maxv = 0.2
                        eps_v = minv + (maxv - minv)*torch.rand(1).item()
                        inpt_a = first_order_attack(sample=inpt, 
                                                    epsilon=0.1,
                                                    dev=device,
                                                    targets=y,
                                                    thismodel=model,
                                                    thiscriterion=loss_fn,
                                                    restrict_impact=-1,
                                                    epsilon_factors=epsilons,
                                                    defaults_per_variable = default_device,
                                                    do_sign_or_normed_grad = "NGM")
                        
                        pred_a = model(inpt_a)

                    pred = model(inpt)
                    if adv and (b % 3 == 0):
                        adv_probs = F.softmax(pred_a, dim=1)
                        nat_probs = F.softmax(pred, dim=1)
                        true_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()
                        #rob_loss = 1*(F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1)).mean() #* (1. - true_probs)).mean()
                        sup_loss = loss_fn(pred_a, y).mean()
                        rob_loss = 1*(F.kl_div(F.log_softmax(model(inpt_a), dim=1), F.softmax(model(inpt), dim=1), reduction='none').sum(dim=1)).mean()
                        loss = sup_loss + rob_loss
                    elif adv and (b % 3 != 0):
                        sup_loss = loss_fn(pred, y).mean()
                        loss = sup_loss
                    else:
                        loss = loss_fn(pred, y).mean()
        
                model.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                pred = model(inpt)
                sup_loss = loss_fn(pred, y).mean()
                loss = sup_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if batch_lr and (scheduler is not None):
                scheduler.step()

            if adv:
                losses += sup_loss.item()
                losses2 += rob_loss.item()
            else:
                losses += loss.item()
                losses2 += 0

            accuracy += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
            N += x.shape[0]
            avg_loss = losses / (it + 1)
            avg_loss2 = losses2 / (it + 1)
            curr_lr = optimizer.param_groups[0]['lr']            
            progress.update(task, advance=1, description=f"Training | CE Loss: {avg_loss:.4f}, Adv Loss: {avg_loss2:.4f}, lr: {curr_lr:.5f}")
            progress.columns[-1].text_format = "{}/{} its".format(
                N // dataloader.batch_size,
                "?"
                if dataloader.nits_expected == len(dataloader)
                else f"~{dataloader.nits_expected}",
            )
            timer += time.time() - start2
            it += 1

        progress.update(task, completed=dataloader.nits_expected)
        end = time.time()
        full_time = end - start
        print("Numb batch = "+str(it))
        print("Time for full batch = "+str(full_time))
        print("Time for ML training only  = "+str(timer))
        print("Time for Dataloader only  = "+str(full_time - timer))

    dataloader.nits_expected = N // dataloader.batch_size
    accuracy /= N

    print("  ", f"Average loss: {avg_loss:.4f}")
    print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
    return np.array(losses).mean(), float(accuracy)


def validate_model(dataloader, model, loss_fn, device="cpu", feature_edges = None, ParT = False):
    losses = []
    accuracy = 0.0
    model.eval()
    with Progress(
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("0/? its"),
        expand=True,
    ) as progress:
        N = 0
        task = progress.add_task("Validation...", total=dataloader.nits_expected)
        for x, w, y in dataloader:

            if ParT:
                inpt = get_inpt(x.float().to(device), feature_edges, ret_glob = False)
            else:
                inpt = get_inpt(x.float().to(device), feature_edges)

            with torch.no_grad():
                pred = model(inpt)
                loss = loss_fn(pred, y.type(torch.LongTensor).to(device)).mean()
                losses.append(loss.item())

                accuracy += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
            N += x.shape[0]
            progress.update(task, advance=1, description=f"Validation... | Loss: {np.array(losses).mean():.4f}")
            progress.columns[-1].text_format = "{}/{} its".format(
                N // dataloader.batch_size,
                "?"
                if dataloader.nits_expected == len(dataloader)
                else f"~{dataloader.nits_expected}",
            )
        progress.update(task, completed=dataloader.nits_expected)
    dataloader.nits_expected = N // dataloader.batch_size
    accuracy /= N
    print("  ", f"Average loss: {np.array(losses).mean():.4f}")
    print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
    return np.array(losses).mean(), float(accuracy)

def get_num_steps(dataloader):
    N = 0
    print("Batch-wise scheduler selected... estimating the number of steps per epochs ongoing...")
    for x, w, y in dataloader:
        if(N % 2000 == 0):
            print(N)
        N += 1
    print("Expected number of steps per epochs :"+str(N))
    return N

def get_inpt(x, feature_edges, ret_glob = True):

    feature_edges = torch.Tensor(feature_edges).int()

    feature_lengths = feature_edges[1:] - feature_edges[:-1]
    feature_lengths = torch.cat((feature_edges[:1], feature_lengths))
    glob, cpf, npf, vtx = x.split(feature_lengths.tolist(), dim=1)
    cpf = cpf.reshape(cpf.shape[0], 26, 16+4)
    npf = npf.reshape(npf.shape[0], 25, 6+4)
    vtx = vtx.reshape(vtx.shape[0], 5, 12+3)
    cpf_4v, npf_4v, vtx_4v = cpf[:,:,-4:], npf[:,:,-4:], vtx[:,:,-4:]
    cpf, npf, vtx = cpf[:,:,:-4], npf[:,:,:-4], vtx[:,:,:-3]

    if ret_glob:
        return (glob.detach(), cpf.detach(), npf.detach(), vtx.detach(),
                cpf_4v.detach(), npf_4v.detach(), vtx_4v.detach())
    else:
        return (cpf[:,:].detach(), npf.detach(), vtx.detach(),
                cpf_4v[:,:].detach(), npf_4v.detach(), vtx_4v.detach())
