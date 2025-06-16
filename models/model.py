import torch.nn as nn

class SAE(nn.Module):
    def __init__(self, input_dim, latent_space_dim, encoder_dims=[100, 50], decoder_dims=None):
        super(SAE, self).__init__()

        encoder_layers = [nn.Linear(input_dim, encoder_dims[0]), nn.ReLU()]
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(encoder_dims[-1], latent_space_dim))
        # encoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)

        
        if decoder_dims is None:
            decoder_dims = list(reversed(encoder_dims))

        decoder_layers = [nn.Linear(latent_space_dim, decoder_dims[0]), nn.ReLU()]
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        decoder_layers.append.append(nn.Sigmoid())          # adjust this based on the embedding range

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
