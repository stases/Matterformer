conda create --yes --name eqstudy python=3.12 numpy
conda activate eqstudy

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
pip install torch_geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install tqdm rdkit pandas pytorch_lightning wandb chamferdist matplotlib loguru gdown
