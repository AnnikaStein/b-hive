import math
import os

import luigi
import numpy as np
import torch
import torch.nn as nn
import uproot
from rich.progress import track
from torch.utils.data import DataLoader

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import DatasetDependency, TrainingDependency
from utils.models.deepjet import DeepJet
from utils.models.DeepJetTransformer import DeepJetTransformer
from utils.models.ParT import ParticleTransformer
from utils.models.Better_ParT import BetterParticleTransformer
from utils.models.PartRet import ParticleRetention
from utils.torch.datasets import DeepJetDataset
from utils.torch.training import perform_training

from utils.torch.definitions_ParT import epsilons_per_feature, vars_per_candidate, defaults_per_variable
from utils.torch.attacks_ParT import *

class TrainingTask(TrainingDependency, DatasetDependency, BaseTask):
    loss_weighting = luigi.BoolParameter(
        False,
        description="Whether to weight the loss or use weighted sampling from the dataset",
    )

    n_threads = luigi.IntParameter(
        default=12, description="Number of threads to use for dataloader."
    )

    def requires(self):
        return DatasetConstructorTask.req(self)

    def output(self):
        return {
            "training_metrics": self.local_target("training_metrics.npz"),
            "validation_metrics": self.local_target("validation_metrics.npz"),
            "model": self.local_target(f"model_{self.epochs-1}.pt"),
            "best_model": self.local_target("best_model.pt"),
        }

    def run(self):
        # Loading config
        config_dict = np.load(self.input()["config_dict"].path, allow_pickle=True).item()
        os.makedirs(self.local_path(), exist_ok=True)
        print("Loading Dataset")
        files = np.array(self.input()["file_list"].load().split("\n")[:-1])

        training_mask = ~(np.char.find(files, "train") == -1)
        validation_mask = ~(np.char.find(files, "validation") == -1)

        training_files = files[training_mask]
        validation_files = files[validation_mask]

        histogram_training = np.load(
            self.input()["histogram_training"].path,
            allow_pickle=True,
        )

        # Define the training and validation datasets
        training_data = DeepJetDataset(
            training_files,
            "training",
            weighted_sampling=not (self.loss_weighting),
            device=self.device,
            histogram_training=histogram_training,
            compression = self.compression,
        )
        validation_data = DeepJetDataset(
            validation_files,
            "validation",
            weighted_sampling=not (self.loss_weighting),
            device=self.device,
            histogram_training=histogram_training,
            compression = self.compression,
        )

        batch_size = 512

        # Define the corresponding dataloaders
        training_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=True,  # Pin Memory for faster CPU/GPU memory load
            num_workers=self.n_threads,
        )
        # Expected number of iterations
        training_dataloader.nits_expected = len(training_dataloader)

        validation_dataloader = DataLoader(
            validation_data,
            batch_size=batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.n_threads,
        )
        validation_dataloader.nits_expected = len(validation_dataloader)

        # Model Defintion
        print("Model definition")
        ParT = False
        if self.model == 'DeepJet':
            model = DeepJet(config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'DeepJetTransformer':
            model = DeepJetTransformer(config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'ParticleTransformer':
            ParT = True
            model = ParticleTransformer(num_classes = 6,
                                        num_enc = 3,
                                        num_head = 8,
                                        embed_dim = 128,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'ParticleRetention':
            ParT = True
            model = ParticleRetention(num_classes = 6,
                                        num_enc = 3,
                                        num_head = 8,
                                        embed_dim = 128,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'BetterParticleTransformer':
            ParT = True
            model = BetterParticleTransformer(num_classes = 6,
                                        num_enc = 6,
                                        num_head = 8,
                                        embed_dim = 128,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'ParticleTransformerHuge':
            ParT = True
            model = ParticleTransformer(num_classes = 6,
                                        num_enc = 12,
                                        num_head = 12,
                                        embed_dim = 12*24,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)


        scaler = torch.cuda.amp.GradScaler()

        # Training
        print("Start training on " + self.device)
        train_metrics, validation_metrics = perform_training(
            model,
            training_dataloader,
            validation_dataloader,
            self.local_path(),
            self.device,
            nepochs=self.epochs,
            FP16 = self.FP16,
            adv = self.adv,
            ParT = ParT,
            scaler = scaler,
            scheduler = self.scheduling,
            feature_edges = config_dict["model"]["feature_edges"],
            compiled = self.compiled
        )

        print("Training finished. Saving data...")

        np.savez(
            self.output()["training_metrics"].path,
            loss=train_metrics[:, 0],
            acc=train_metrics[:, 1],
            allow_pickle=True,
        )
        np.savez(
            self.output()["validation_metrics"].path,
            loss=validation_metrics[:, 0],
            acc=validation_metrics[:, 1],
            allow_pickle=True,
        )


class InferenceTask(TrainingDependency, DatasetDependency, BaseTask):
    def requires(self):
        return {"training": TrainingTask.req(self), "dataset": DatasetConstructorTask.req(self)}

    def output(self):
        return {
            "output_root": self.local_target("output.root"),
            "prediction": self.local_target("prediction.npy"),
            "truth": self.local_target("truth.npy"),
            "kinematics": self.local_target("kinematics.npy"),
        }

    def run(self):
        os.makedirs(self.local_path(), exist_ok=True)
        config_dict = np.load(self.input()["dataset"]["config_dict"].path, allow_pickle=True).item()

        ParT = False
        if self.model == 'DeepJet':
            model = DeepJet(config_dict["model"]["feature_edges"], for_inference = True).to(self.device)
        if self.model == 'DeepJetTransformer':
            model = DeepJetTransformer(config_dict["model"]["feature_edges"], for_inference = True).to(self.device)
        if self.model == 'ParticleTransformer':
            ParT = True
            model = ParticleTransformer(num_classes = 6,
                                        num_enc = 3,
                                        num_head = 8,
                                        embed_dim = 128,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'ParticleRetention':
            ParT = True
            model = ParticleRetention(num_classes = 6,
                                        num_enc = 3,
                                        num_head = 8,
                                        embed_dim = 128,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'ParticleTransformerBig':
            ParT = True
            model = ParticleTransformer(num_classes = 6,
                                        num_enc = 6,
                                        num_head = 8,
                                        embed_dim = 192,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'ParticleTransformerHuge':
            ParT = True
            model = ParticleTransformer(num_classes = 6,
                                        num_enc = 12,
                                        num_head = 12,
                                        embed_dim = 12*24,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)
        if self.model == 'BetterParticleTransformer':
            ParT = True
            model = BetterParticleTransformer(num_classes = 6,
                                        num_enc = 6,
                                        num_head = 8,
                                        embed_dim = 128,
                                        cpf_dim = 16,
                                        npf_dim = 6,
                                        vtx_dim = 12,
                                        for_inference = False,
                                        small = True,
                                        feature_edges = config_dict["model"]["feature_edges"]).to(self.device)

        best_model = torch.load(
            self.input()["training"]["best_model"].path,
            map_location=torch.device(self.device),
        )
        model.load_state_dict(best_model["model_state_dict"])

        if self.adv:
            loss_fn = nn.CrossEntropyLoss(reduction="none")
            epsilons = {'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(self.device),
                        'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(self.device),
                        'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(self.device),
                        'cpf_pts' : torch.Tensor(np.load(epsilons_per_feature['cpf_pts']).transpose()).to(self.device),
                        'npf_pts' : torch.Tensor(np.load(epsilons_per_feature['npf_pts']).transpose()).to(self.device),
                        'vtx_pts' : torch.Tensor(np.load(epsilons_per_feature['vtx_pts']).transpose()).to(self.device)
            }
            default_device = defaults_per_variable
            

        print("Loading Dataset")
        files = np.array(
            open(self.input()["dataset"]["file_list"].path, "r").read().split("\n")[:-1]
        )
        test_mask = ~(np.char.find(files, "test") == -1)
        test_files = files[test_mask]

        histogram_test = np.load(
            self.input()["dataset"]["histogram_test"].path,
            allow_pickle=True,
        )
        test_data = DeepJetDataset(test_files, "test", histogram_training=histogram_test, compression = self.compression)
        test_dataloader = DataLoader(test_data, batch_size=1000, num_workers=4)

        model.eval()
        kinematics = []
        truth = []
        prediction = []
        output = []
        if self.adv:
            a_nom = []
            a_adv = []
            IP2D_nom = []
            IP2D_adv = []
            sv_mass_nom = []
            sv_mass_adv = []
        for x, _, y in track(test_dataloader, "Inference..."):
            x = x.float().to(device=self.device)

            if ParT:
                inpt = get_inpt(x, config_dict["model"]["feature_edges"], ret_glob = False)
            else:
                inpt = get_inpt(x, config_dict["model"]["feature_edges"])

            if self.adv:
                model.zero_grad()
                inpt_a = first_order_attack(sample=inpt,
                                            epsilon=0.2,
                                            dev=self.device,
                                            targets=y.type(torch.LongTensor).to(self.device),
                                            thismodel=model,
                                            reduced=True,
                                            thiscriterion=loss_fn,
                                            restrict_impact=-1,
                                            epsilon_factors=epsilons,
                                            defaults_per_variable = default_device,
                                            do_sign_or_normed_grad = "NGM")

                IP2D_nom.append(inpt[0][:,24,5])
                IP2D_adv.append(inpt_a[0][:,24,5])
                sv_mass_nom.append(inpt[2][:,4,2])
                sv_mass_adv.append(inpt_a[2][:,4,2])
                a_nom.append(inpt[0][:,:,:].reshape(-1,16))
                a_adv.append(inpt_a[0][:,:,:].reshape(-1,16))

            kinematics.append(x[:, :2, 0])
            truth.append(y)
            with torch.no_grad():
                if self.adv:
                    pred = model(inpt_a)
                else:
                    pred = model(inpt)
                pred = torch.softmax(pred, dim=1)
                prediction.append(pred)
                if len(output) == 0:
                    output = pred.cpu().numpy()
                else:
                    output = np.append(output, pred.cpu().numpy(), axis=0)

        prediction = torch.cat(prediction, dim=0).cpu().numpy()
        kinematics = torch.cat(kinematics, dim=0).cpu().numpy()
        truth = torch.cat(truth, dim=0).cpu().numpy().astype(int)
        one_hot_truth = np.zeros((len(truth), np.max(truth) + 1))
        one_hot_truth[np.arange(len(truth)), truth] = 1
#        if self.adv:
 #           import matplotlib.pyplot as plt
  #          a_nom = torch.cat(a_nom, dim=0).cpu().numpy()
   #         a_adv = torch.cat(a_adv, dim=0).cpu().numpy()
    #        a_adv = a_adv[a_nom[:,1] > 0.0]
     #       a_nom = a_nom[a_nom[:,1] > 0.0]
      #      IP2D_nom = torch.cat(IP2D_nom, dim=0).cpu().numpy()
       #     IP2D_adv = torch.cat(IP2D_adv, dim=0).cpu().numpy()
        #    sv_mass_nom = torch.cat(sv_mass_nom, dim=0).cpu().numpy()
         #   sv_mass_adv = torch.cat(sv_mass_adv, dim=0).cpu().numpy()
          #  print(a_nom.shape)
           # print(a_adv.shape)

#            for i in range(16):
 #               fig,ax = plt.subplots(figsize=[12,12])
  #              ax.scatter(a_nom[:,i], a_adv[:,i], alpha = 0.05, s=1)
   #             if i == 5:
    #                ax.set_xlim(-0.001,0.001)
     #               ax.set_ylim(-0.001,0.001)
      #          ax.set_ylabel('Population')
       #         ax.set_xlabel(str(i)+'cpf_fts')
        #        ax.grid(True)

#                fig.savefig(str(i)+'cpf_fts.png')

#                fig,ax = plt.subplots(figsize=[12,12])
 #               bins = np.arange(0.0,100,0.5)
  #              a1 = a_nom[a_nom[:,i] != 0]
   #             a2 = a_adv[a_nom[:,i] != 0]
    #            delta = a1[:,i] - a2[:,i] #get delta value
     #           delta = 100*abs(delta) / (abs(a1[:,i])+0.00000000000000001) #get the abs and have the percentage
      #          print(np.median(delta))
       #         ax.hist(delta, bins)
        #        ax.set_ylabel('Population')
         #       ax.set_xlabel(str(i)+'_cpf_fts')
          #      ax.grid(True)

#                fig.savefig(str(i)+'cpf_delta.png')
 #               fig,ax = plt.subplots(figsize=[12,12])
  #              lower_value = np.percentile(a_nom[:,i], 2)
   #             higher_value = np.percentile(a_nom[:,i], 98)
    #            if i == 5:
     #               binning = np.arange(-0.1, 0.1, 0.001)
      #          elif i == 6:
       #             binning = np.arange(0, 50, 0.5)
        #        elif i == 7:
         #           binning = np.arange(-0.2, 0.2, 0.001)
          #      elif i == 8:
           #         binning = np.arange(0, 50, 0.5)
            #    elif higher_value != lower_value:
             #       binning = np.arange(lower_value, higher_value, abs(higher_value-lower_value)/100)
              #  else:
               #     binning = 100
                    
#                ax.hist(a_nom[:,i], binning, histtype = 'step', alpha = 0.5, label = 'nominal')
 #               ax.hist(a_adv[:,i], binning, histtype = 'step', alpha = 0.3, label = 'blue')
  #              ax.set_ylabel('Distribution')
   #             ax.set_xlabel(str(i)+'_cpf_fts')
    #            ax.legend()
     #           ax.grid(True)

      #          fig.savefig(str(i)+'cpf_distr.png')

        np.save(self.output()["prediction"].path, prediction)
        np.save(self.output()["kinematics"].path, kinematics)
        np.save(self.output()["truth"].path, truth)

        output = np.concatenate((kinematics, prediction, one_hot_truth), axis=1)
        with uproot.recreate(self.output()["output_root"].path) as root_file:
            root_file["tree"] = {
                "Jet_pt": output[:, 0],
                "Jet_eta": output[:, 1],
                "prob_isB": output[:, 2],
                "prob_isBB": output[:, 3],
                "prob_isLeptB": output[:, 4],
                "prob_isC": output[:, 5],
                "prob_isUDS": output[:, 6],
                "prob_isG": output[:, 7],
                "isB": output[:, 8],
                "isBB": output[:, 9],
                "isLeptB": output[:, 10],
                "isC": output[:, 11],
                "isUDS": output[:, 12],
                "isG": output[:, 13],
            }


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
