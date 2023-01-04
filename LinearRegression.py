import numpy as np
import matplotlib.pyplot as plt 

def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def readFile(path):
    input_data = fetch_data(path)
    input_np = np.array(input_data)
    return input_np

training_data = './datasets/LR_train.txt'
train_np = readFile(training_data)

train = train_np.astype(np.float32)

testing_data = './datasets/LR_test.txt'
test_np = readFile(testing_data)

#print(train_np)

test = test_np.astype(np.float32)

#print(train)

Y = train[:,-1]
X= train[:,:-1]

Y= Y.reshape(Y.size,1)

Y_test = test[:,-1]
X_test= test[:,:-1]

Y_test= Y_test.reshape(Y_test.size,1)
#print(X.shape)
#print(Y.shape)

plt.scatter(X,Y)
plt.show()

def model(X, Y, learning_rate, iteration):
    m = Y.size
    theta = np.zeros((1,1))
    cost = []
    for i in range(iteration):
        Y_pred = np.dot(X,theta)
        cost_value = (1/(2*m))*np.sum(np.square(Y_pred-Y))
        d_theta = (1/m)*np.dot(X.T,Y_pred-Y)
        theta = theta - learning_rate*d_theta
        cost.append(cost_value)

    return theta,cost

iteration = 500
theta, cost_list = model(X, Y, 0.5,iteration=iteration)

Y_test_pred = np.dot(X_test,theta)
error = (np.sum(np.abs(Y_test - Y_test_pred))) / Y_test.size

print(error)

temp = np.arange(0, iteration)
plt.plot(cost_list, temp)
plt.show()