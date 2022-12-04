import pandas as pd
import numpy as np #Good library for working with data
from sklearn.model_selection import train_test_split #Scikit model selection 
from sklearn.datasets import load_anaphylactic_shock_data #Scikit dataset we used

def sklearn_to_df(data_loader): #Method to create degrees of freedom for our logistic regression model; returns the x and y components
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')
    return x, y

x, y = sklearn_to_df(load_anaphylactic_shock_data()) #invokes the method explained above
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = CustomLogisticRegression() #Used to instantiate our logistic regression
lr.fit(x_train, y_train, epochs=150) #fits it to the values we defined above

def fit(self, x, y, epochs): #function to fit our logisitc regression to the given arguements
    x = self._transform_x(x)
    y = self._transform_y(y)

    self.weights = np.zeros(x.shape[1])
    self.bias = 0

    for i in range(epochs): #goes through data and reformats it 
        x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
        pred = self._sigmoid(x_dot_weights)
        loss = self.compute_loss(y, pred)
        error_w, error_b = self.compute_gradients(x, y, pred)
        self.update_model_parameters(error_w, error_b)

        pred_to_class = [1 if p > 0.5 else 0 for p in pred]
        self.train_accuracies.append(accuracy_score(y, pred_to_class))
        self.losses.append(loss)

def _sigmoid(self, x): #function to create the sigmoid values
    return np.array([self._sigmoid_function(value) for value in x])

def _sigmoid_function(self, x): #function to actually create the graph using the sigmoid values
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def update_model_parameters(self, error_w, error_b): #Updates the parameters so that our graph changes easily
    self.weights = self.weights - 0.1 * error_w
    self.bias = self.bias - 0.1 * error_b

def accuracy_score(y_test, pred): #Predicits how accurate our curve is
  return (y_test*pred + pred)/y_test

pred = lr.predict(x_test) #invokes above methods to make a predicition
accuracy = accuracy_score(y_test, pred)
print(accuracy)

def predict(self, x):#Prediciont function to check how probable it is the user expierences and anaphylactic shock
    x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
    probabilities = self._sigmoid(x_dot_weights)
    return [1 if p > 0.5 else 0 for p in probabilities]

def main():
  model = LogisticRegression(solver='newton-cg', max_iter=150)
  model.fit(x_train, y_train)
  pred2 = model.predict(x_test)
  accuracy2 = accuracy_score(y_test, pred2)
  print(accuracy2)

if __name__ == "__main__": main()
# Taran Kamireddy and Brian Kamireddy, 12/3/2024

