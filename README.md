# nn-handwriting
Neural network implmentation trained on MNIST data set written in Python 3.6. The network achieves around around 97-98% accuracy on test data.

## Installation / Setup
#### Linux/OSX
```
python -m venv env
cd env\scripts\activate
activate
``` 
``` 
pip install -r requirements.txt
```
#### Windows
Create a new folder called 'wheels' in the root of the repository

Download the following whl files from http://www.lfd.uci.edu/~gohlke/pythonlibs/ into the wheels folder
- numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl
- matplotlib-2.0.2-cp36-cp36m-win_amd64.whl
- scipy-0.19.1-cp36-cp36m-win_amd64.whl
```
python -m venv env
cd env\scripts\activate
activate
``` 
```
pip install -r requirements_win.txt
```
#### Data
Because of file size limits the MNIST dataset is not included in the repository (other than the two small example sets). 
Download the following csv files into the data folder (in the root of repo)
- https://pjreddie.com/media/files/mnist_train.csv
- https://pjreddie.com/media/files/mnist_test.csv
## Usage
```
python main.py [training.csv] [test.csv]
```
Ex:
```
python main.py ./../data/mnist_train_100.csv ./../data/mnist_test_10.csv
```
```
python main.py ./../data/mnist_train.csv ./../data/mnist_test.csv
```

## References
Make Your Own Neural Network - Tariq Rashid

MNIST in CSV - https://pjreddie.com/projects/mnist-in-csv/

## License
See LICENSE file
