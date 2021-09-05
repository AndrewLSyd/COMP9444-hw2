#!/bin/bash
# https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments
# conda create -y --name bws-mta-test --file spec-file.txt
# run this first! conda config --append channels conda-forge
echo "INFO: creating conda env and installing conda packages first"
conda create -y --name pytorch-scoring --file requirements-conda.txt
echo "INFO: installing pip packages"
pip install -r requirements-pip.txt