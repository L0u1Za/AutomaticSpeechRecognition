from torch import nn
from hw_asr.base import BaseModel

class DeepSpeechModel(BaseModel):
    def __init__(self, n_feats, n_class, rnn_layers=5, rnn_hidden_size=512, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(11, 41), stride=(1, 2), padding=(5, 20)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        )

        self.rnn = nn.GRU(
            input_size=n_feats * 32 // 2 // 2,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, n_class)

    def forward(self, spectrogram, **batch):
        #print("====start====")
        #print(spectrogram.shape)
        spectrogram = spectrogram.transpose(1, 2).unsqueeze(1) # Batch x 1(channel) x Embed x Time
        #print(spectrogram.shape)

        output = self.conv(spectrogram)
        #print("Conv", output.shape)
        output = output.transpose(1, 2).flatten(2, 3)
        #print(output.shape)
        output, _ = self.rnn(output)
        #print("Rnn", output.shape)

        output = self.fc(output)
        #print(output.shape)
        #print("====end=====")
        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths