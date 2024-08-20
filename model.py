import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, LayerNorm, Dropout, MultiheadAttention
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        scaling_factor = math.sqrt(self.head_dim)
        attention = torch.nn.functional.softmax(energy / scaling_factor, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.features = nn.Sequential(*list(alexnet.features.children())[:-1])

    def forward(self, x):
        # Extract features from the last three layers
        outputs = []
        for layer in self.features:
            x = layer(x)
            outputs.append(x)
        return outputs[-3:]  # Return features from the last three layers


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(FeatureEncoder, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)
        self.positional_encoding = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.multihead_attn = MultiheadAttention(embed_dim=out_channels, num_heads=num_heads)
        self.norm = LayerNorm(out_channels)

    def forward(self, x1, x2):
        # Convolution and positional encoding
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x1 += self.positional_encoding
        x2 += self.positional_encoding

        # Flatten the feature maps for multi-head attention
        B, C, H, W = x1.size()
        x1_flat = x1.view(B, C, -1).permute(2, 0, 1)
        x2_flat = x2.view(B, C, -1).permute(2, 0, 1)

        # Multi-head attention
        attn_output, _ = self.multihead_attn(x1_flat, x2_flat, x2_flat)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

        # Normalization and residual connection
        output = self.norm(x1 + attn_output)
        return output


class FeatureDecoder(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(FeatureDecoder, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.norm1 = LayerNorm(in_channels)
        self.norm2 = LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            Linear(in_channels * 4, in_channels),
        )
        self.dropout = Dropout(0.1)

    def forward(self, x):
        # Flatten for multi-head attention
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(2, 0, 1)

        # Multi-head attention
        attn_output, _ = self.multihead_attn(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

        # Residual connection and normalization
        x = self.norm1(x + attn_output)
        x = x + self.ffn(x.view(B, C, -1).permute(2, 0, 1)).permute(1, 2, 0).view(B, C, H, W)
        x = self.norm2(x)

        return x



class ClassificationAndRegression(nn.Module):
    def __init__(self, in_channels):
        super(ClassificationAndRegression, self).__init__()
        self.cls_conv = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.reg_conv = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, x):
        cls_output = self.cls_conv(x)
        reg_output = self.reg_conv(x)
        return cls_output, reg_output


class ModulationLayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ModulationLayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)  # Channel-wise multiplication


class HiFT(nn.Module):
    def __init__(self):
        super(HiFT, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_encoder = FeatureEncoder(in_channels=256, out_channels=256)
        self.modulation_layer = ModulationLayer(in_channels=256)
        self.feature_decoder = FeatureDecoder(in_channels=256)
        self.classification_and_regression = ClassificationAndRegression(in_channels=256)

    def forward(self, z, x):
        # Feature extraction
        z_features = self.feature_extractor(z)
        x_features = self.feature_extractor(x)

        # Feature transformation
        encoded_features = []
        for i in range(3):
            encoded = self.feature_encoder(z_features[i], x_features[i])

            # Apply Modulation Layer after the first Add & Norm and FFN layers
            modulated = self.modulation_layer(encoded)

            # Process with the concatenation and convolution (if applicable in your design)
            modulated = self.concatenate_and_conv(modulated)

            decoded = self.feature_decoder(modulated)
            encoded_features.append(decoded)

        # Sum the transformed features
        final_features = sum(encoded_features)

        # Classification and regression
        cls_output, reg_output = self.classification_and_regression(final_features)

        return cls_output, reg_output


def train(model, dataloader, epochs, device):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()

    for epoch in range(epochs):
        model.train()
        for i, (template, search, cls_label, reg_label) in enumerate(dataloader):
            template, search = template.to(device), search.to(device)
            cls_label, reg_label = cls_label.to(device), reg_label.to(device)

            optimizer.zero_grad()
            cls_output, reg_output = model(template, search)

            loss_cls = criterion_cls(cls_output, cls_label)
            loss_reg = criterion_reg(reg_output, reg_label)
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")