import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # h_0 = torch.zeros(self.num_layers, x.size(
        #     0), self.hidden_dim).requires_grad_()
        # c_0 = torch.zeros(self.num_layers, x.size(
        #     0), self.hidden_dim).requires_grad_()

        # # Forward propagate LSTM
        # out, (h_out, _) = self.lstm(x, (h_0.detach(), c_0.detach()))

        # # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])

        # return out

        h_0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        x = x.squeeze(0)  # убираем размерность seq_len из x
        h_0 = h_0.unsqueeze(0)  # добавляем новую размерность в h_0
        c_0 = c_0.unsqueeze(0)  # добавляем новую размерность в c_0
        out, (h_out, _) = self.lstm(x, (h_0.detach(), c_0.detach()))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


# Prepare the data
dm = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
cd = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141,
      135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150, 178, 163, 172, 178]
an = [11, 9, 8, 9, 9, 8, 11, 8, 9, 11, 10, 11, 10, 9, 8,
      11, 8, 11, 10, 9, 11, 8, 11, 9, 9, 9, 9, 8, 9, 10]

# Set the hyperparameters
input_dim = 2
hidden_dim = 10
output_dim = 1
num_layers = 1
learning_rate = 0.01
num_epochs = 500

# Create the LSTM model
model = LSTMPredictor(input_dim, hidden_dim, output_dim, num_layers)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Convert data to tensors
    x = torch.tensor([[cd[i], an[i]]
                     for i in range(len(dm)-1)], dtype=torch.float32)
    y = torch.tensor([[cd[i+1]]
                     for i in range(len(dm)-1)], dtype=torch.float32)

    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer
