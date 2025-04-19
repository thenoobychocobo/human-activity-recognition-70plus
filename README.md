# Human Activity Recognition in Older Adults using Deep Learning
Human Activity Recognition (HAR) plays a critical role in understanding the relationship between physical activity and health outcomes. With aging populations in Singapore and across Asia, there is a growing need for robust, data-driven tools to monitor physical activity in older adults and support independent living.

In this project, we apply deep learning techniques to classify physical activities performed by older adults using time-series data from wearable sensors. We base our work on two HAR datasets: the [Human Activity Recognition Trondheim (HARTH) dataset](https://archive.ics.uci.edu/dataset/779/harth) and the [Human Activity Recognition 70+ (HAR70+)](https://archive.ics.uci.edu/dataset/780/har70) datasets.

## Table of Contents
- [Project Overview](#project-overview)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Project Overview
```
root/
│
├── models/
├── notebooks/                   
│   ├── eda.ipynb
│   ├── inference_demo.ipynb
│   └── model_training.ipynb
├── results/
├── results_demo/
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── models.py
│   └── train_eval.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt                   
```
- `models/`: Model parameters (`.pth` files) will be saved to this directory by default.
- `notebooks/`: Contains Jupyter notebooks (`.ipynb` files) for exploratory data analysis, model training, and a demo notebook if you wish to skip training and want to try out our trained models.
- `results/`: Training and validation results will be saved to this directory by default. Also saves the results of the trained model (based on last epoch parameters) on the test set.
- `results_demo/`: Results on test set from `inference_demo.ipynb` are saved here. Currently contains the test results of the model with the best F1 during training for each architecture.
- `src/`:
    - `data_preparation.py`: Contains code for downloading and processing the dataset.
    - `models.py`: Contains code for deep learning model classes.
    - `train_eval.py`: Contains code for training and evaluation of models.
    - `utils.py`: Helper functions.

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
3. Dataset exploration: `notebooks/eda.ipynb`.
4. Model training: `notebooks/model_training.ipynb`. You can skip training if you wish as we already provide trained model parameters in `models/`. 
5. Model inference demo (load model parameters and evaluate on test set): `notebooks/inference_demo.ipynb`.

## Acknowledgements
The github repository for the HARTH and HAR70+ datasets can be found [here](https://github.com/ntnu-ai-lab/harth-ml-experiments).
### HARTH
Aleksej Logacjov, Kerstin Bach, Atle Kongsvold, Hilde Bremseth Bardstu, and Paul Jarle Mork. HARTH: A Human Activity Recognition Dataset for Machine Learning. Sensors, 21(23):7853, November 2021.
```bibtext
@article{logacjovHARTHHumanActivity2021,
  title = {{{HARTH}}: {{A Human Activity Recognition Dataset}} for {{Machine Learning}}},
  shorttitle = {{{HARTH}}},
  author = {Logacjov, Aleksej and Bach, Kerstin and Kongsvold, Atle and B{\aa}rdstu, Hilde Bremseth and Mork, Paul Jarle},
  year = {2021},
  month = nov,
  journal = {Sensors},
  volume = {21},
  number = {23},
  pages = {7853},
  publisher = {{Multidisciplinary Digital Publishing Institute}},
  doi = {10.3390/s21237853}
}
```

Kerstin Bach, Atle Kongsvold, Hilde Bardstu, Ellen Marie Bardal, Hakon S.Kjærnli, Sverre Herland, Aleksej Logacjov, and Paul Jarle Mork. A Machine Learning Classifier for Detection of Physical Activity Types and Postures During Free-Living. Journal for the Measurement of Physical Behaviour, pages 1–8, December 2021.
```bibtext
@article{bachMachineLearningClassifier2021,
  title = {A {{Machine Learning Classifier}} for {{Detection}} of {{Physical Activity Types}} and {{Postures During Free-Living}}},
  author = {Bach, Kerstin and Kongsvold, Atle and B{\aa}rdstu, Hilde and Bardal, Ellen Marie and Kj{\ae}rnli, H{\aa}kon S. and Herland, Sverre and Logacjov, Aleksej and Mork, Paul Jarle},
  year = {2021},
  month = dec,
  journal = {Journal for the Measurement of Physical Behaviour},
  pages = {1--8},
  publisher = {{Human Kinetics}},
  doi = {10.1123/jmpb.2021-0015},
}
```
### HAR70+
Astrid Ustad, Aleksej Logacjov, Stine Øverengen Trollebø, Pernille
Thingstad, Beatrix Vereijken, Kerstin Bach, and Nina Skjæret Maroni. Validation of an Activity Type Recognition Model Classifying Daily Physical Behavior in Older Adults: The HAR70+ Model. Sensors, 23(5):2368, January 2023.

```bibtext
@article{ustadValidationActivityType2023,
  title = {Validation of an {{Activity Type Recognition Model Classifying Daily Physical Behavior}} in {{Older Adults}}: {{The HAR70}}+ {{Model}}},
  shorttitle = {Validation of an {{Activity Type Recognition Model Classifying Daily Physical Behavior}} in {{Older Adults}}},
  author = {Ustad, Astrid and Logacjov, Aleksej and Trolleb{\o}, Stine {\O}verengen and Thingstad, Pernille and Vereijken, Beatrix and Bach, Kerstin and Maroni, Nina Skj{\ae}ret},
  year = {2023},
  month = jan,
  journal = {Sensors},
  volume = {23},
  number = {5},
  pages = {2368},
  publisher = {{Multidisciplinary Digital Publishing Institute}},
  issn = {1424-8220},
  doi = {10.3390/s23052368},
  copyright = {http://creativecommons.org/licenses/by/3.0/}
}
```
