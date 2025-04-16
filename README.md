# Human Activity Recognition in Older Adults using Deep Learning
This repository provides code for applying deep learning techniques to recognize and classify physical activities performed by older adults using time-series sensor data. Human Activity Recognition (HAR) plays a critical role in studying the relationship between physical activity and various health outcomes. 

We base our work on the [Human Activity Recognition 70+ (HAR70+)](https://archive.ics.uci.edu/dataset/780/har70) dataset[^1]. This dataset contains sensor data from 18 fit-to-frail older adult subjects (70-95 years old) wearing two 3-axial accelerometers (attached to the right thigh and lower back) for around 40 minutes during a semi-structured free-living protocol. We selected the HAR70+ dataset for its rich and professionally annotated sensor data. Notably, datasets centered on older adults remain underrepresented in human activity recognition research, despite their growing importance. As populations in Singapore and across Asia continue to age rapidly, there is a pressing need for robust, data-driven methods to support health monitoring and encourage independent living among older adults.

[^1]: Ustad, A., Logacjov, A., Trollebø, S.Ø., Thingstad, P., Vereijken, B., Bach, K., Maroni, N.S., 2023. Validation of an Activity Type Recognition Model Classifying Daily Physical Behavior in Older Adults: The HAR70+ Model. Sensors 23, 2368. https://doi.org/10.3390/s23052368

## Table of Contents
- [Project Overview](#project-overview)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Project Overview
```
root/
│
├── data/
├── models/
├── notebooks/                   
│   ├── eda.ipynb
│   ├── kfold_cv.ipynb
│   └── model_training.ipynb
├── results/
├── saved_components/
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── models.py
│   └── train_eval.py
├── .gitignore
├── README.md
└── requirements.txt                   
```
- `data/`: Dataset files will be downloaded to this directory by default.
- `models/`: Model parameters (`.pth` files) will be saved to this directory by default.
- `notebooks/`: Contains Jupyter notebooks (`.ipynb` files) for exploratory data analysis, model training, and k-fold cross validation.
- `results/`: Training and validation results in the form of graphs (`.png` files) will be saved to this directory by default.
- `saved_components/`: Processed dataset information (e.g. split indices) will be saved to this directory by default.
- `src/`:
    - `data_preparation.py`: Contains code for downloading and processing the dataset.
    - `models.py`: Contains code for deep learning model classes.
    - `train_eval.py`: Contains code for training and evaluation of models.

## Usage
This code was developed using Python 3.12.8.

1. Clone the repository:
    ```bash
    git clone https://github.com/thenoobychocobo/human-activity-recognition-70plus.git
    cd human-activity-recognition-70plus
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Open and run the notebooks in the `notebooks/` folder.

## Acknowledgements
WIP