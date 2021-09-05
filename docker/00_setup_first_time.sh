#!bin/bash
# conda update -n base conda

conda create -n pytorch-scoring ipykernel

source activate pytorch-scoring
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -y numpy matplotlib pandas scikit-learn seaborn pandas

echo "INFO: exporting conda requirements"
conda list --explicit > requirements-conda.txt

pip install progressbar2
echo "INFO: exporting pip requirements"
pip freeze > requirements-pip.txt
pip freeze | egrep "progressbar2" > requirements-pip.txt