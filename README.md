# nn-handwriting
Neural network implmentation trained on MNIST data set written in Python 3.6. The network achieves around around 97-98% accuracy on the full MNIST dataset.

## Build
```
docker build -t nn-handwriting .
```

## Run
Containerized/Docker:
```
docker run -it --rm --name nn-handwriting nn-handwriting
```

Direct:
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

### Data
Because of file size limits the MNIST dataset is not included in the repository (other than the two small example sets). To perform a complete training regimen download the following CSV files and replace the train/test files in ```data```.
- https://pjreddie.com/media/files/mnist_train.csv
- https://pjreddie.com/media/files/mnist_test.csv

## References
Make Your Own Neural Network - Tariq Rashid

MNIST in CSV - https://pjreddie.com/projects/mnist-in-csv/
