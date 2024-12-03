import numpy as np
import json

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize a neural network with random weights and biases."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # initialize weights and biases with He initialization for ReLU activation
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
        
    def backward(self, X, y, output, learning_rate):
        # compute loss (cross-entropy)
        error = output - y
        d_W2 = np.dot(self.a1.T, error)
        d_b2 = np.sum(error, axis=0, keepdims=True)
        
        d_a1 = np.dot(error, self.W2.T)
        d_z1 = d_a1 * self.relu_derivative(self.a1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # update weights and biases
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
    
    def train(self, X, y, learning_rate, epochs):
        """Train the neural network on the given data."""
        # one-hot encode the labels y
        y_one_hot = np.eye(self.output_size)[y]
        
        for epoch in range(epochs):
            # forward pass
            output = self.forward(X)
            
            # compute loss (cross-entropy)
            loss = -np.mean(np.sum(y_one_hot * np.log(output + 1e-10), axis=1))  # Cross-entropy loss
            print(f"Epoch {epoch}/{epochs}, Loss: {loss}")
            
            # backward pass
            self.backward(X, y_one_hot, output, learning_rate)
    
    def predict(self, X):
        """Predict the class labels for the given input data."""
        return np.argmax(self.forward(X), axis=1)
    
    def save_model(self, file_path):
        """Saves the model's architecture, weights, and biases to a JSON file."""
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist()
        }

        with open(file_path, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load_model(cls, file_path):
        """Loads a model stored in a JSON file and returns an instance of the neural network."""
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        
        nn = cls(
            input_size=model_data['input_size'],
            hidden_size=model_data['hidden_size'],
            output_size=model_data['output_size']
        )
        
        nn.W1 = np.array(model_data['W1'])
        nn.b1 = np.array(model_data['b1'])
        nn.W2 = np.array(model_data['W2'])
        nn.b2 = np.array(model_data['b2'])

        return nn