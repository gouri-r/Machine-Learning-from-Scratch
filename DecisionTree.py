from sklearn.metrics import accuracy_score
import numpy as np
import math
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

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

labels = train_np[:,-1]
train_data = train_np[:,:-1]

test_labels = test_np[:,-1]
test_data = test_np[:,:-1]

train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)

unique_labels = np.unique(labels)

for i in range (0,labels.size):
    for j in range (0,unique_labels.size):
        if labels[i] == unique_labels[j]:
            labels[i] = j
            break
            
for i in range (0,test_labels.size):
    for j in range (0,unique_labels.size):
        if test_labels[i] == unique_labels[j]:
            test_labels[i] = j
            break
            
labels = labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

labels = np.reshape(labels, (-1, 1))
test_labels = np.reshape(test_labels, (-1, 1))

#print(train_data)
#print(test_data)
#print(labels)
#print(test_labels)

class Node():

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, information_gain=None, value=None):
        
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.value = value

class DecisionTreeClassifier():
    
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["information_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["information_gain"])
        
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        
        best_split = {}
        max_information_gain = -float("inf")
        
        for feature_index in range(num_features):
            
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_information_gain = self.information_gain_method(y, left_y, right_y, "gini")
                    if curr_information_gain>max_information_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["information_gain"] = curr_information_gain
                        max_information_gain = curr_information_gain
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain_method(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        #class_labels = np.unique(y)
        entropy = 0
        for cls in unique_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        
        #class_labels = np.unique(y)
        gini = 0
        for cls in unique_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
                
        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        preditions = [self.single_prediction(x, self.root) for x in X]
        return preditions

    
    def single_prediction(self, x, tree):
       
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.single_prediction(x, tree.left)
        else:
            return self.single_prediction(x, tree.right)


    
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(train_data,labels)

Y_pred = classifier.predict(test_data) 
accuracy_score(test_labels, Y_pred)
print(accuracy_score)
