# MSCGrapher

## Usage

- Train and evaluate MSCGrapher
  - You can use the following command:`sh ./scripts/ETTh1.sh`.

- Train your model
  - Add model file in the folder `./models/your_model.py`.
  - Add model in the ***class*** Exp_Main.

- Datasets
  - You can obtain the Flight dataset from [Google Drive](https://drive.google.com/drive/folders/1JSZByfM0Ghat3g_D3a-puTZ2JsfebNWL?usp=sharing). Then please place it in the folder `./dataset`.

## Model

The main components of MSCGrapher: Embedding layer, Multi-scale correlation learning block, Multi-head attention layer, Multi-scale aggregation layer, and Projection layer. Embedding layer maps the raw time series into a high-dimensional representation, which facilitates model input. Multi-scale correlation learning module divides the time series into different time scales and learns the correlation relationships between series. Multi-head attention layer captures temporal correlations within series. Finally, Multi-scale aggregation layer and Projection layer integrate information from different scales to produce the final prediction.

## Note

MSCGrapher is modified from MSGNet, which is the source code of the AAAI'2024 Paper [MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting](https://arxiv.org/abs/2401.00423). 

## Acknowledgement

We appreciate the valuable contributions of the following GitHub.

- LTSF-Linear (https://github.com/cure-lab/LTSF-Linear)
- TimesNet (https://github.com/thuml/TimesNet)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Autoformer (https://github.com/thuml/Autoformer)
- Informer (https://github.com/zhouhaoyi/Informer2020)
- FourierGNN (https://github.com/aikunyi/FourierGNN)
- StemGNN (https://github.com/microsoft/StemGNN)
- MSGNet(https://github.com/YoZhibo/MSGNet)
