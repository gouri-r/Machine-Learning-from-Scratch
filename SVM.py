import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score

def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    input_np = np.array(input_data)
    return input_np

training_data = './datasets/DT_train.txt'
testing_data = './datasets/DT_test.txt'

train_np = readFile(training_data)
test_np = readFile(testing_data)

Y = train_np[:,-1]
X = train_np[:,:-1]

X_test = test_np[:,-1]
Y_test= test_np[:,:-1]

unique_labels = np.unique(Y)

for i in range (0,Y.size):
    for j in range (0,unique_labels.size):
        if Y[i] == unique_labels[j]:
            Y[i] = j
            break
            
for i in range (0,Y_test.size):
    for j in range (0,unique_labels.size):
        if Y_test[i] == unique_labels[j]:
            Y_test[i] = j
            break

Y = np.reshape(Y, (-1, 1))
Y_test = np.reshape(Y_test, (-1, 1))

class SVM:

    def __init__(self, learning_rate=0.05, lambda_param=0.01, n_iters=100):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


classifier = SVM()
classifier.fit(X,Y)
predictions = classifier.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)

print(accuracy)