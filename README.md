# Graph Neural Networks for State Estimation in Water Distribution systems

This repository contains code implementation for Graph Neural Networks (GNN) for State Estimation in Water Distribution systems.
The paper has been submitted to Jornual of Water Resources Planning and Management

The proposed GNN architecture learns to estimate the hydraulic states by exploiting the topology
of the WDS and computing hydraulic dynamics via learned message passing.
Two learning structures are formulated and investigated:
(1) a supervised scheme that is trained using complete information provided by
a hydraulic solver using many simulations of different network topologies and demands, and
(2) a semi-supervised approach that receives only a limited amount of information of
measurements at given locations and simultaneously explores the physical laws of mass and energy conversation.

## Virtual environment

It is recommended to use a [virtual environment](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html), as it helps manage in a clean way the code dependencies. 
```virtualenv
pip install virtualenv
```
Go to the folder /GNN-StateEstimation-WDS, and create a new virtualenv, with python 3
```virtualenv
virtualenv ENV -p python3
```
The command above should have created a folder `ENV/`. Now you need to activate your virtual environment.
```virtualenv
source ENV/bin/activate
```
You should now see (ENV) before your username.
If you want to deactivate your virtualenv (for instance to work on another project), use the following command line
```virtualenv
deactivate
```
But for now, keep you virtual environment activated!

## Requirements

If you have a GPU and want to use it, install the following requirements:

```setup
pip install -r requirements-gpu.txt
```

Otherwise, if you do not have a GPU, or do not want to use it, install the following requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the models used in the paper, here are the exact commands that were used:

- Supervised learning

```
python main.py --data_dir=datasets/asnet2_1 --learning_rate=1e-3 --minibatch_size=500 --alpha=1e-2 --hidden_layers=2 --latent_dimension=20 --correction_updates=20 --track_validation=1000 --proxy
```

- Semi-supervised learning with 1 measurement location

```
python main.py --data_dir=datasets/asnet2_1 --learning_rate=1e-3 --minibatch_size=500 --alpha=1e-2 --hidden_layers=2 --latent_dimension=20 --correction_updates=20 --track_validation=1000
```

- Semi-supervised learning with 5 measurement location


```
python main.py --data_dir=datasets/asnet2_5 --learning_rate=1e-3 --minibatch_size=500 --alpha=1e-2 --hidden_layers=2 --latent_dimension=20 --correction_updates=20 --track_validation=1000
```

## Evaluation & Visualization

To evaluate the models, codes are provided under the `visualize/` folder to reload trained model, perform inference on the test set, and also some visualizations.

