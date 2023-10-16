from torch import nn
from hw_asr.base import BaseModel

class DeepSpeechModel(BaseModel):
    def __init__(self, n_feats, n_class, rnn_layers=5, rnn_hidden_size=512, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        )
        conv_output_size = (n_feats + 2 * 5 - (11 - 1) - 1) // 2 + 1
        conv_output_size = (conv_output_size + 2 * 5 - (11 - 1) - 1) // 1 + 1

        self.rnn = nn.GRU(
            input_size=conv_output_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, n_class)

    def forward(self, spectrogram, **batch):
        #print("====start====")
        #print(spectrogram.shape)
        spectrogram = spectrogram.unsqueeze(1).transpose(2, 3) # Batch x 1(channel) x Embed x Time
        #print(spectrogram.shape)

        output = self.conv(spectrogram)
        #print("Conv", output.shape)
        batch_size, channels, dimension, seq_lengths = output.size()

        #output = output.permute(0, 3, 1, 2)
        #output = output.view(batch_size, seq_lengths, channels * dimension)
        output = output.flatten(1, 2)
        #print(output.shape)
        output, _ = self.rnn(output)
        #print("Rnn", output.shape)

        output = self.fc(output)
        #print(output.shape)
        #print("====end=====")
        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths