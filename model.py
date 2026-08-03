import torch.nn as nn

from spikingjelly.activation_based import surrogate

from bilif import TwoWayLIFNode

class EvSNet(nn.Module):
    def __init__(
        self,
        hidden_channels=64,
        kernel_size=7,
        dropout=0.1,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv_in = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size, padding=pad),
            nn.ReLU(),
        )

        self.resblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=pad),
                    nn.ReLU(),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=pad),
                )
                for _ in range(3)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
        self.out_conv = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        self.output_lif = TwoWayLIFNode(
            surrogate_function=surrogate.ATan(),
            detach_reset=False,
            v_threshold=1.,
            v_reset=None,
            tau=1e3,
            step_mode='m',
            backend='cupy',
            decay_input=False,
        )

    def forward(self, x):
        x = self.conv_in(x)  # [B, C, T]

        for blk in self.resblocks:
            y = blk(x)
            residual_sum = x + y
            x = F.relu(residual_sum)
            x = self.dropout(x)
        logits = self.out_conv(x)  # [B, 1, T]
        
        output_spikes = logits.permute(2, 0, 1)
        output_spikes = self.output_lif(output_spikes)
        output_spikes = output_spikes.permute(1, 2, 0)  # [B, 2, T]
        return output_spikes