# 06-run-pytorch-data.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.dataset import Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core import PipelineRun, StepRun, PortDataReference
from azureml.pipeline.steps import PythonScriptStep
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from azureml.core import Run
from azureml.core.runconfig import RunConfiguration

# +


if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/cifar10'))
# -

aml_compute_target = "clusterdemo"
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
    print("found existing compute target.")
except ComputeTargetException:
    print("creating new compute target")
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                min_nodes = 1, 
                                                                max_nodes = 4, 
                                                                idle_seconds_before_scaledown=1000)
    aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)

# +

aml_compute.wait_for_completion(show_output=True)
experiment = Experiment(workspace=ws, name='DSP_test')

print("Azure Machine Learning Compute attached")   

config = ScriptRunConfig(
    source_directory='./src',
    script='trainDSP.py',
    compute_target='clusterdemo',
    arguments=[
        '--data_path', dataset.as_named_input('input').as_mount(),
        '--learning_rate', 0.003,
        '--momentum', 0.92,
        '--output_dir', './outputs']
)
# set up pytorch environment
env = Environment.from_conda_specification(
    name='pytorch-env',
    file_path='.azureml/pytorch-env.yml'
)
config.run_config.environment = env

aml_run_config = RunConfiguration()
aml_run_config.environment.docker.enabled = True
aml_run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/base:latest"
aml_run_config.environment.python.user_managed_dependencies = False
aml_run_config.environment = env

#train step
train_step = PythonScriptStep(script_name="trainDSP.py",
                         source_directory='./src',
                         compute_target='clusterdemo',
                         runconfig=aml_run_config,
                        arguments=[
                        '--data_path', dataset.as_named_input('input').as_mount(),
                        '--learning_rate', 0.003,
                        '--momentum', 0.92,
                        '--output_dir', './outputs'])

#test step
test_step = PythonScriptStep(script_name="testDSP.py",
                         source_directory='./src',
                         compute_target='clusterdemo',
                         runconfig=aml_run_config,
                        arguments=[
                        '--data_path', dataset.as_named_input('input').as_mount(),
                        '--learning_rate', 0.003,
                        '--momentum', 0.92,
                        '--output_dir', './outputs']) 


steps = [train_step,test_step]
pipeline1 = Pipeline(workspace=ws, steps=steps)
pipeline_run1 = Experiment(ws, 'DSP_pipeline').submit(pipeline1)
#run = experiment.submit(config)

# +
#model = run.register_model(model_name='DSP_demo',model_path='outputs/DSP_demo.pt', tags={'area': "experiment", 'type': "torch"})
# -

# model = run.register_model(model_name='DSP_demo',model_path='outputs/DSP_demo.pt',
#                            tags={'area': "experiment", 'type': "torch"})

# pipeline_run1.id

# +
#from azureml.core.model import Model
#print(Model.get_model_path(model_name="pytorch_model_1",_workspace=ws))

# +
#from azureml.core.model import Model
#print(Model.get_model_path(model_name="DSP_train",_workspace=ws))

# +
#from azureml.core import Workspace
#ws = Workspace.from_config()
#print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

# +
#model = run.register_model(model_name='DSP_train',model_path='outputs/DSP_demo.pt', tags={'area': "experiment", 'type': "torch"})
# -

# model = torch.load('azureml-models/pytorch_model_1/2/DSP_demo.pt')




