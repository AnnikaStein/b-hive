# Tasks

All basic LAW tasks are placed inside the *tasks* directory, whereas all workflows are placed inside thw *workflows* directory.

## Detailed description of the tasks
This sections prevides a describtion of the tasks. For a more detailed information of the individual function, please have a look at the comments in the code.

### DatasetConstructorTask
This task relies on [dataset/dataset.py]() to read in [PFNano](https://github.com/cms-jet/PFNano) or [DeepNTuple]() files and store them in the numpy file format.

After specifying in the `sample_dict`what files to process, a coffea processor is started to run over the input files. Losse cuts are applied to the data (10<= p_T <=2000 and -2.5<= eta <=2.5), before the features needed for a training of the DeepJet model are extracted and the truth information is set to
```
0 for b jets,
1 for bb jets,
2 for leptonic b jets,
3 for c jets,
4 for uds jets and 
5 for g jets
```
in the output.

To not exhaust the available RAM of your machine, the files are processed and saved in chunks. The chunk size is a parameter you can adjust to the resources and capabilities of your system. The 
The processed files are stored in the directory defined in the configuration chapter. Furthermore, a text file is stored in the same directory containing all paths to the newly generated files for easer loading later.

For reweighting the inputs of the model later in the training step, one histogram per flavour mentioned above is filed and save in the same directory as well.
The respective binning is
```
p_t = [10, 25, 30, 35, 40, 45, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 2000] and 
eta = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1, 1.5, 2.0, 2.5] 
```
as in the original DeepJet implementation.

### TrainingTask
The training task will train the model based on the previously generated files and relies on [training/training.py](). Before a training starts, a weight for every jet is calculated based on its bin in the the p_t / eta space and flavour, according the method called `referenceclass="isB"` in the original [DeepJetCore implementation](https://github.com/DL4Jets/DeepJetCore/blob/master/Weighter.py#L145-L260).

For the training you will be able to choose between reweighting the loss itself by incorporating the aforementioned weights in the loss calculated or alternatively use PyTorch's WeightedRandomSampler to draw a jet collection.

For every trained epoch, a model checkpoint will be saved, including training and validation loss, in the format `model_$EPOCH.pt`. In addition, the best performing model according to the validation loss it saved in the format `best_model.pt`. Model checkpoints give you a safety net in case the job ends unexpectedly.

_WIP:_
- _Early stopping_

### InferenceTask
The inference task will calculate a predition using the previously trained model and same the output as a numpy and ROOT file. The numpy files includes the ouput of the network, while the root file contains additionally p_T, eta and the truth in one-hot-encoding according to the following keys
```
Jet_pt       : transverse momentum of the jet
Jet_eta      : eta of the jet
prob_isB     : predicted probability that it is a b jet
prob_isBB    : predicted probability that it is a bb jet
prob_isLeptB : predicted probability that it is a leptonic b jet
prob_isC     : predicted probability that it is a c jet
prob_isUDS   : predicted probability that it is a uds jet
prob_isG     : predicted probability that it is a g jet
isB          : 1 if it is a b jet, 0 otherwise
isBB         : 1 if it is a bb jet, 0 otherwise
isLeptB      : 1 if it is a leptonic b jet, 0 otherwise
isC          : 1 if it is a c jet, 0 otherwise
isUDS        : 1 if it is a uds jet, 0 otherwise
isG          : 1 if it is a g jet, 0 otherwise
``` 
in the tree.

It also relies on [training/training.py]() and includes the same methods and functions as the training task, if applicable.

### PlottingTask
The plotting task will evaluate and visualise the results from the training and prediction. It relies in [plotting/plotting.py]().

It plots and save the training and validation loss against the trained epochs. Futhermore, the discriminators `B vs L`, `C vs L` and `C vs B` are saved and plotted using ROC curves. The resulting files are splitted between TT (30GeV < p_T < 1000GeV) and QCD (300GeV < p_T < 1000GeV).