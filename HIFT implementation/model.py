import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, Dropout, MultiheadAttention, GroupNorm
import math
from torch.utils.data import DataLoader, TensorDataset

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.features = nn.Sequential(*list(alexnet.features.children())[:-1])

    def forward(self, x):
        outputs = []
        for layer in self.features:
            x = layer(x)
            outputs.append(x)
        return outputs[-3:]

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(FeatureEncoder, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)
        self.positional_encoding = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.multihead_attn = MultiheadAttention(embed_dim=out_channels, num_heads=num_heads)
        self.norm = GroupNorm(8, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x1 += self.positional_encoding
        x2 += self.positional_encoding

        B, C, H, W = x1.size()
        x1_flat = x1.view(B, C, -1).permute(2, 0, 1)
        x2_flat = x2.view(B, C, -1).permute(2, 0, 1)

        attn_output, _ = self.multihead_attn(x1_flat, x2_flat, x2_flat)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

        output = self.norm(x1 + attn_output)
        return output

class FeatureDecoder(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(FeatureDecoder, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.norm1 = GroupNorm(8, in_channels)
        self.norm2 = GroupNorm(8, in_channels)
        self.ffn = nn.Sequential(
            Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            Linear(in_channels * 4, in_channels),
        )
        self.dropout = Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(2, 0, 1)

        attn_output, _ = self.multihead_attn(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

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
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HiFT(nn.Module):
    def __init__(self):
        super(HiFT, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_encoder = FeatureEncoder(in_channels=256, out_channels=256)
        self.modulation_layer = ModulationLayer(in_channels=256)
        self.feature_decoder = FeatureDecoder(in_channels=256)
        self.classification_and_regression = ClassificationAndRegression(in_channels=256)
        self.concat_conv = nn.Conv2d(256 * 2, 256, kernel_size=1)

    def concatenate_and_conv(self, z, x):
        concatenated = torch.cat((z, x), dim=1)
        fused = self.concat_conv(concatenated)
        return fused

    def forward(self, z, x):
        z_features = self.feature_extractor(z)
        x_features = self.feature_extractor(x)

        encoded_features = []
        for i in range(3):
            encoded = self.feature_encoder(z_features[i], x_features[i])
            modulated = self.modulation_layer(encoded)
            fused_features = self.concatenate_and_conv(z_features[i], x_features[i])
            decoded = self.feature_decoder(fused_features)
            encoded_features.append(decoded)

        final_features = sum(encoded_features)
        cls_output, reg_output = self.classification_and_regression(final_features)

        return cls_output, reg_output

def train(model, dataloader, epochs, device):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
    criterion_cls = nn.MSELoss()
    criterion_reg = nn.SmoothL1Loss()

    for epoch in range(epochs):
        model.train()
        for i, (template, search, cls_label, reg_label) in enumerate(dataloader):
            template, search = template.to(device), search.to(device)
            # Convert cls_label to float
            cls_label = cls_label.float().to(device)
            reg_label = reg_label.to(device)

            optimizer.zero_grad()
            cls_output, reg_output = model(template, search)

            # Resizing cls_label and reg_label to match the output dimensions
            cls_label_resized = F.interpolate(cls_label, size=cls_output.shape[2:], mode='nearest')
            reg_label_resized = F.interpolate(reg_label, size=reg_output.shape[2:], mode='nearest')

            loss_cls = criterion_cls(cls_output, cls_label_resized)
            loss_reg = criterion_reg(reg_output, reg_label_resized)
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# Example usage
if __name__ == "__main__":
    # Create dummy data
    batch_size = 4
    template = torch.randn(batch_size, 3, 224, 224)
    search = torch.randn(batch_size, 3, 224, 224)
    cls_label = torch.randint(0, 2, (batch_size, 2, 1, 1))
    reg_label = torch.randn(batch_size, 4, 1, 1)

    dataset = TensorDataset(template, search, cls_label, reg_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and train
    model = HiFT()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, dataloader, epochs=10, device=device)