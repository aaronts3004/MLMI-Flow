# MLMI-Flow

A Capability Study of Flow Matching on Structured Dynamics:
Rigid and Deformable Motion in a Surgical Simulator


### Install 
Create a virtual environment, e.g. 
```bash 
python -m venv .mlmi_venv
```
and install the dependencies as 
```bash 
source .mlmi_venv/bin/activate 
pip install -r requirements.txt
```
### Create dataset
From the root of the repo (MLMI-Flow) run: 
```bash 
python -m scripts.generate_dataset 
```
This will generate an HDF5 file with trajectories categorized for each rigid motion primitive inside the (new) `data` folder.   
You can inspect the dataset a bit with `python -m scripts.inspect_dataset`  

### Run Train Script 
After generating the dataset, you can launch a training with  
```bash 
python -m src.train
```
This script handles the training parameters and orchestration. The loop over epochs and batches etc. happens in `trainer.py`  
At the top of `train.py` you can choose the `TRAINING_OBJECTIVE` - can be either "autoregressive" or "flow_matching"   
Make sure that your HDF5_PATH points to the .h5 file  

#### Autoregressive Objective (Overfitting Experiment)
In this setting for each primitive (line, circle, etc.) we over-fit a single trajectory per primitive  
A single trajectory = 100 time-steps with pose changes  
For this we train a simple PoseMLP: given pose(t) predict pose(t+1)  
Note: this branch still needs fixing  


#### Flow-Matching Objective
In this setting for each primitive (line, circle, etc.) we have 100, 1000, ... trajectories 
For each primitive we train a new TrajectoryMLP()
Inside `prepare_flow_matching_batch(...)` we define source, t, and target for the FM objective  
In this case the MLP learns to de-noise a noisy trajectory