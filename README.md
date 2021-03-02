# Natural Language Processing - Funny Headlines
### 2020-2021 Natural Language Processing Coursework

This repository is intended to be a submission for a university coursework based on the
[Assessing the Funniness of Edited News Headlines](https://competitions.codalab.org/competitions/20970) competition. 
The problem spec, datasets and other relevant information can be found on the competition page.

## Repository Structure
To reproduce the experiments, you can simply open and run the `main.ipynb` Jupyter Notebook. 

Feel free to create a Python virtual environment, activate and then run the following to install the dependencies.
```
pip install -r requirements.txt
```
Then, open the jupyter notebook `main.ipynb`.
```
jupyter notebook main.ipynb
```
Each cell in the notebook corresponds to a particular experiment. The corresponding code can be found in the 
.py files in the `approach_1` and `approach_2` folders.

The files relevant for perusal are shown below.
```
.
├── README.md
├── main.ipynb
├── approach_1
│   ├── A1_BERT.py
│   ├── A1_CNN_Concat_GloVe.py
│   ├── A1_CNN_GloVe.py
│   ├── A1_FFNN_GloVe.py
│   └── preprocessing_experiment.py
├── approach_2
│   ├── A2_CNN_Concat_Word2Index.py
│   ├── A2_CNN_Word2Index.py
│   ├── A2_Experiments.py
│   ├── A2_FFNN_Word2Index.py
```