# Welcome to b-hive (work in progress)
This framework is a modernised version of [DeepJet](https://github.com/DL4Jets/DeepJet) and [DeepJetCore](https://github.com/DL4Jets/DeepJetCore), taking advantage of modern packages like [PyTorch](https://pytorch.org), [numpy](https://numpy.org), [awkward](https://awkward-array.org/doc/main/), [coffea](https://coffeateam.github.io/coffea/), [uproot](https://uproot.readthedocs.io/en/latest/) and [law](https://law.readthedocs.io/en/latest/).
You will be able to read in ROOT files, extract features needed for a training of the DeepJet model, perform a training, make predictions using a trained model and evaluate the output/performance.


## Setup

For software setup, conda is used. This ensures portability to most machines but is not mandatory for running the framework.

### Quick-Setup

```bash
# clone repository
git clone ssh://git@gitlab.cern.ch:7999/cms-btv/b-hive.git
# set up conda env
conda env create -n b_hive -f env.yml
# activate eny
conda activate b_hive
# set up environment variables 
source setup.sh

```


## Configuration

### Set up Local Configurations

In the last step of the `setup.sh`, a local script is sourced, called `local_setup.sh`.
This should be created by the user and specifies the working directory, where results should be placed.
For example:

```bash
#!/bin/bash
export DATA_PATH=/net/scratch/YOURDIRECTORY/BTV/training/
```

if this file is not created or `$DATA_PATH` is not set otherwise, everything will be placed in the *results* directory.

1) Everytime you want to use the framework, you need to source `setup.sh` by executing
```
source setup.sh
```
in the shell.

## Running the Framework

To get familiar with the possibilities, running the basic law index command
```
law index --verbose
```
will print the availabel commands.

### LAW Parameters

Every task has specific parameters in order to steer its behaviour, for example the number of training epochs or a debug flag.

### Examplatory Workflow

```bash
ToDo

```

## Usage
To peform a task simply execute
```
law run $TASK_NAME
```
in the shell. The currently available tasks are
- `DatasetConstructorTask`: reads in ROOT files and stores the relevant branches in numpy files,
- `TrainingTask`: performes a training with the previously generated numpy files,
- `InferenceTask`: performes a prediction using the previously trained model and
- `PlottingTask`: generates ROC curves using the output of the prediction
  
Due to the usage of law, the framework will check if previous steps in the chain have already been completed and automatically execute them if necessary or fall back on intermediate results to execute the requested task.


