# HFTCRNet

Accurately assessing and forecasting bank credit ratings and systemic risk at an early stage is vitally important for a healthy financial environment and sustainable economic development. 
In this repository, we contribute the datasets and codes for our paper "HFTCRNet: Hierarchical Fusion Transformer for Interbank Credit Rating and Risk Assessment" which is under review by TNNLS.

## Requirements

```
python                       3.7.16
torch                        1.13.1
torch-cluster                1.6.1
pyg                          2.3.0
torch-scatter                2.1.1
torch-sparse                 0.6.17
tqdm                         4.42.1
scikit-learn                 1.0.2
pandas                       1.2.3
numpy                        1.21.5
powerlaw                     1.3.4
```

To install all these packages, please first install miniconda and then try this command:

```
conda install tqdm scikit-learn pandas numpy pyg pytorch-scatter pytorch-sparse pytorch-cluster pytorch-cluster pytorch-spline-conv pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -c pyg
```

## Dataset

We collected data spanning 29 quarters, covering a diverse array of banks worldwide, from the first quarter of 2016 to the first quarter of 2023. The included bank categories encompass commercial banks, savings banks, cooperative banks, real estate and mortgage banks, investment banks, Islamic banks, and central banks. In particular, the data set includes Silicon Valley Bank, Signature Bank, First Republic Bank and Credit Suisse Group during the 2023 financial crisis, as well as banks speculated to be closely related to them, so that they can be used to analyze this crisis event in the future.

For each quarter, there is an Edge table, a Feature table, and a Target table. 

+ For the Edge table, we used our temporal-consistency improved minimum density method to generate Interbank loan networks. 

+ For the Feature table, we collected over 300 features related to various bank finance. After undergoing data cleaning, feature selection, dimensionality reduction, and normalization, we finally identified 70 features. 

+ For the Target table, we have credit rating and SRISK:
  
  + For data ratings, we collect existing ranks given by Moody's Analytics, and transfer original bank ranks of each bank to relative ranks with ``A``, ``B``, ``C``, and ``D`` to facilitate further analysis on the systemic risk.
  
  + For the SRISK value and ratio, we collect them from the [vLab](https://vlab.stern.nyu.edu).

## Usage

### Contagion Chain Generation

To generate the risk contagion chain from the features, run
```
cd ./src
python generate_contagionlist.py
```

### Model Training

To train the HFTCRNet, please first configurate the ``./src/run.sh`` and then run the ``./src/run.sh`` with ``bash``.

* Configurate the ``./src/run.sh``
    ```
    start_year=2018
    start_quarter=1
    end_year=2023
    end_quarter=1
    time_steps=7

    CUDA_VISIBLE_DEVICES=0 python train.py ...
    ```

* Run the ``./src/run.sh`` with ``bash``
    ```
    cd src
    bash run.sh
    ```

## Results

The results of different credit rating method with regards to different metrics are shown as follows:

|  Method  |         Accuracy          |    Macro-$\text{F}_1$     |
| -------- | ------------------------- | ------------------------- |
| GCN      |   0.537   $\pm$   0.060   |   0.435   $\pm$   0.043   |
| GAT      |   0.540   $\pm$   0.087   |   0.429   $\pm$   0.123   |
| SAGNN    |   0.546   $\pm$   0.059   |   0.436   $\pm$   0.046   |
| TGAR     |   0.628   $\pm$   0.084   |   0.569   $\pm$   0.094   |
| STGCN    |   0.562   $\pm$   0.129   |   0.469   $\pm$   0.126   |
| STMGCN   |   0.614   $\pm$   0.081   |   0.553   $\pm$   0.086   |
| HSTGCNT  |   0.659   $\pm$   0.099   |   0.598   $\pm$   0.106   |
| TESTAM   |   0.647   $\pm$   0.095   |   0.603   $\pm$   0.102   |
| FreeDyG  |   0.615   $\pm$   0.075   |   0.557   $\pm$   0.089   |
| MDGNN_bs |   0.648   $\pm$   0.089   |   0.590   $\pm$   0.102   |
| BiSTAT   |   0.622   $\pm$   0.088   |   0.544   $\pm$   0.082   |
| HFTCRNet | **0.678** $\pm$ **0.104** | **0.630** $\pm$ **0.110** |

The **best** performance are highlighted **in bolded**.

## License

The use of the source code of interbank complies with the GNU GENERAL PUBLIC LICENSE.
Please contact us if you find any potential violations. 