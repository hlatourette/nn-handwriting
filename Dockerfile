FROM python:3
COPY . /usr/local/src/nn-handwriting
WORKDIR /usr/local/src/nn-handwriting
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./src/main.py", "./data/mnist_train.csv", "./data/mnist_test.csv"]