import torch.nn as nn
import torch.nn.init as init

class MLPModel(nn.Module):
    def __init__(self, embeddings_size, hidden_layer_sizes, dropout_rate, num_classes):
        super(MLPModel, self).__init__()

        layers = []
        input_size = embeddings_size

        # Create hidden layers
        for size in hidden_layer_sizes:
            linear_layer = nn.Linear(input_size, size)
            init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = size

        # Output layer
        layers.append(nn.Linear(input_size, num_classes))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)




class SimpleMLPModel(nn.Module):
    def __init__(self, embeddings_size, dropout_rate, num_classes):
        super(SimpleMLPModel, self).__init__()
        
        # First hidden layer
        self.layer1 = nn.Linear(embeddings_size, 768)
        init.xavier_uniform_(self.layer1.weight)
        self.batchnorm1 = nn.BatchNorm1d(768)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second hidden layer
        self.layer2 = nn.Linear(768, 128)
        init.xavier_uniform_(self.layer2.weight)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.batchnorm1(self.layer1(x))))
        x = self.dropout2(self.relu2(self.batchnorm2(self.layer2(x))))
        x = self.output_layer(x)
        return x

