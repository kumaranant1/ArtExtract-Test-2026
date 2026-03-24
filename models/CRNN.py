from .CNN import CNNFeatureExtractor
from .RNN import RNN
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_artists, num_styles, num_genres, rnn_hidden_dim=512):
        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.rnn = RNN(
            input_dim=self.cnn.out_channels,
            num_layers=2,
            hidden_dim=rnn_hidden_dim
        )

        classifier_input_dim = rnn_hidden_dim*2

        self.artist_head = nn.Linear(classifier_input_dim, num_artists)
        self.style_head = nn.Linear(classifier_input_dim, num_styles)
        self.genre_head = nn.Linear(classifier_input_dim, num_genres)

    def forward(self, x):
        cnn_features = self.cnn(x)

        rnn_features = self.rnn(cnn_features)

        artist_output = self.artist_head(rnn_features)
        style_output = self.style_head(rnn_features)
        genre_output = self.genre_head(rnn_features)

        return artist_output, style_output, genre_output