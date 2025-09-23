# import csv reading library
import csv
import numpy as np
from matplotlib import pyplot as plt
import torch
import os

# create a simple Elman recurrent neural network:
class ElmanRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.activation(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def get_data_set():
    
    # create empty lists to store month and passenger data
    month = []
    passengers = []

    # check if the file exists:
    if os.path.exists('AirPassenger.csv'):
        print("File 'AirPassengers.csv' found. Using real dataset.")
        # open the CSV file
        with open('AirPassengers.csv', mode='r') as file:
            # create a CSV reader object
            csv_reader = csv.reader(file)
            
            # iterate over each row in the CSV file
            i = 0
            for row in csv_reader:
                # first column has month, second column has number of passengers
                if i > 0:  # skip the header row
                    month.append(i-1)
                    # convert row[1] to integer:
                    passengers.append(int(row[1]))
                i += 1

        # Convert the lists to numpy arrays for easier manipulation
        month = np.array(month)
        passengers = np.array(passengers)

        #passengers = passengers.astype(np.float32) / 100.0  # Normalize data
        input_tensor = torch.tensor(passengers[:-1]).unsqueeze(1)
        target_tensor = torch.tensor(passengers[1:]).unsqueeze(1)
        # make the input and target tensors float32:
        # input_tensor = input_tensor.float()
        # target_tensor = target_tensor.float()
    else:
        # make a dummy dataset, with a sine wave + linear ramp + noise
        print("File 'AirPassengers.csv' not found. Using dummy dataset.")
        t = np.linspace(0, 4 * np.pi, 144)
        passengers = ((0.4 + (t * 9.0/144.0)) * np.sin(3.0 * t) + 0.8 * t + 0.3 * np.random.randn(144)).astype(np.float32)
        month = np.arange(144)
        #passengers = (passengers - np.min(passengers)) / (np.max(passengers) - np.min(passengers))  # Normalize data
        passengers *= 40.0  # Scale to similar range as real data
        input_tensor = torch.tensor(passengers[:-1]).unsqueeze(1)
        target_tensor = torch.tensor(passengers[1:]).unsqueeze(1)

        # plt.figure()
        # plt.plot(month, passengers, label='Dummy Data')
        # plt.xlabel('Month')
        # plt.ylabel('Number of Passengers (normalized)')
        # plt.title('Dummy Air Passengers Data')
        # plt.legend()
        # plt.show()


    return month, passengers, input_tensor, target_tensor


# Custom correlation-based loss function
def correlation_loss(output, target):
    """
    Negative correlation loss - maximizes correlation between output and target
    """
    # Flatten tensors
    output_flat = output.view(-1)
    target_flat = target.view(-1)
    
    # Calculate means
    output_mean = torch.mean(output_flat)
    target_mean = torch.mean(target_flat)
    
    # Calculate correlation coefficient
    numerator = torch.sum((output_flat - output_mean) * (target_flat - target_mean))
    output_std = torch.sqrt(torch.sum((output_flat - output_mean) ** 2))
    target_std = torch.sqrt(torch.sum((target_flat - target_mean) ** 2))
    
    correlation = numerator / (output_std * target_std + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Return negative correlation (we want to maximize correlation, so minimize negative correlation)
    return -correlation

# Alternative: Covariance-based loss
def covariance_loss(output, target):
    """
    Negative covariance loss - maximizes covariance between output and target
    """
    # Stack tensors to use torch.cov
    combined = torch.stack([output.view(-1), target.view(-1)])
    cov_matrix = torch.cov(combined)
    covariance = cov_matrix[0, 1]  # Off-diagonal element
    return -covariance

# Training loop:
def train(rnn, criterion, optimizer, input_tensor, target_tensor, n_epochs=1000):
    outputs_list = []  # Store outputs for correlation calculation
    targets_list = []  # Store targets for correlation calculation
    
    for epoch in range(n_epochs):
        hidden = rnn.initHidden()
        rnn.zero_grad()
        
        outputs_list.clear()
        targets_list.clear()
        
        # Forward pass - collect all outputs
        for i in range(input_tensor.size(0)):
            output, hidden = rnn(input_tensor[i].unsqueeze(0), hidden)
            outputs_list.append(output)
            targets_list.append(target_tensor[i].unsqueeze(0))
        
        # Calculate loss based on all outputs vs targets
        all_outputs = torch.cat(outputs_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        loss = criterion(all_outputs, all_targets)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return rnn

def run_training():
    # Choose your loss function:
    # Option 1: MSE Loss (original - learns average)
    criterion = torch.nn.MSELoss()

    # Option 2: Correlation-based loss (maximizes correlation)
    # criterion = correlation_loss

    # Option 3: Covariance-based loss (maximizes covariance)
    # criterion = covariance_loss
    input_size = 1
    hidden_size = 10
    output_size = 1
    rnn = ElmanRNN(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

    # Prepare the data for training
    month, passengers, input_tensor, target_tensor = get_data_set()
    target_tensor /= 100.0
    ratio = 0.8
    rnn = train(rnn, criterion, optimizer, input_tensor[:int(ratio * len(input_tensor))], target_tensor[:int(ratio * len(target_tensor))], n_epochs=1000)
    # Make predictions
    rnn.eval()
    predictions = []
    hidden = rnn.initHidden()
    for i in range(input_tensor.size(0)):
        with torch.no_grad():
            output, hidden = rnn(input_tensor[i].unsqueeze(0), hidden)
            predictions.append(output.item())

    # Plot the results
    plt.figure()
    plt.plot(month[1:], passengers[1:], label='Actual')
    plt.plot(month[1:], predictions, label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('Number of Passengers')
    plt.title('Air Passengers Prediction using Elman RNN')
    plt.legend()
    plt.show()

    # save the neural network:
    torch.save(rnn.state_dict(), 'elman_rnn_air_passengers.pth')

if __name__ == "__main__":
    run_training()