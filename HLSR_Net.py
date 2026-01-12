import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.convnext import convnext_tiny
from timm.models.layers import DropPath

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        if self.upsample:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class FreqDecomposer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.adaptive_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        low_freq_base = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq_base = x - low_freq_base
        
        weights = self.adaptive_weight(x)
        low_weight, high_weight = weights[:, 0:1], weights[:, 1:2]
        
        low_freq = low_freq_base * (0.8 + 0.4 * low_weight)
        high_freq = high_freq_base * (0.8 + 0.4 * high_weight)
        
        return low_freq, high_freq
class DEAM(nn.Module):
    def __init__(self, in_dim, ds=8):
        super().__init__()
        self.in_dim = in_dim
        self.key_channel = in_dim // 8
        self.ds = ds
        
        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(in_dim, self.key_channel, 1)
        self.key_conv = nn.Conv2d(in_dim, self.key_channel, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        
        self.diff_processor = nn.Sequential(
            nn.Conv2d(in_dim, in_dim//8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim//8, 1, 1),
            nn.Sigmoid()
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        original_size = x.shape[2:]
        x_down = self.pool(x)
        m_batchsize, C, width, height = x_down.size()
        
        proj_query = self.query_conv(x_down).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_down).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x_down).view(m_batchsize, -1, width * height)
        
        energy = torch.bmm(proj_query, proj_key)
        energy = (self.key_channel ** -0.5) * energy
        attention = self.softmax(energy)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        diff_weight = self.diff_processor(x_down)
        out = out * (1 + diff_weight)
        
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        return self.gamma * out + x
class BAD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim//4),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, groups=4),
                nn.BatchNorm2d(dim)
            ) for _ in range(3)
        ])
        
        self.boundary_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//16, 1, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        identity = x
        features = [x]
        
        for i, block in enumerate(self.dense_blocks):
            inp = features[0] if i == 0 else sum(features)
            out = block(inp) + inp
            features.append(out)
        
        final_feature = self.fusion(features[-1])
        boundary_weight = self.boundary_detector(identity)
        enhanced = final_feature * boundary_weight + final_feature
        
        return enhanced + identity

class HLFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decomposer = FreqDecomposer(dim)
        self.low_branch = DEAM(dim)
        self.high_branch = BAD(dim)

    def forward(self, x):
        low_freq, high_freq = self.decomposer(x)
        enhanced_low = self.low_branch(low_freq)
        enhanced_high = self.high_branch(high_freq)
        fused = enhanced_low + enhanced_high
        return fused, enhanced_low, enhanced_high

class Convnext_HLFM(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)

        self.encoder_layers = nn.ModuleList([
            backbone.features[0],
            backbone.features[2],
            backbone.features[4],
            backbone.features[6],
        ])
        
        self.feature_dims = [96, 192, 384, 768]
        
        self.hlfm_blocks = nn.ModuleList([
            HLFM(dim) for dim in self.feature_dims
        ])
        
    def forward(self, x):
        features = []
        temp_x = x
        
        for i, layer in enumerate(self.encoder_layers):
            temp_x = layer(temp_x)
            temp_x, _, _ = self.hlfm_blocks[i](temp_x)
            features.append(temp_x)
            
        return features

class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
    
    def forward(self, x):
        return self.bn(self.conv(x))

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, g=dim)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = ConvBN(mlp_ratio * dim, dim, 1)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        x = self.dwconv2(x)
        return input + self.drop_path(x)

class SR(nn.Module):
    def __init__(self, dim, drop_path=0.1, dilation=3):
        super().__init__()
        self.feature_transform = Star_Block(dim, mlp_ratio=2, drop_path=drop_path)
    
    def forward(self, x):
        return self.feature_transform(x)
class HLSR(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        
        self.encoder = Convnext_HLFM(pretrained=pretrained)
        
        self.doubleCBR = nn.Sequential(
            nn.Conv2d(768, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.up3 = UpBlock(1024, 768, 384, upsample=False)
        self.up2 = UpBlock(384, 384, 192)
        self.up1 = UpBlock(192, 192, 96)
        self.up0 = UpBlock(96, 96, 64)
        
        self.decoder3 = SR(384, drop_path=0.1)
        self.decoder2 = SR(192, drop_path=0.1)
        self.decoder1 = SR(96, drop_path=0.1)
        self.decoder0 = SR(64, drop_path=0.1)
        
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        f0, f1, f2, f3 = features
        
        CDR = self.doubleCBR(f3)

        d3 = self.up3(CDR, f3)
        d3 = self.decoder3(d3)
        
        d2 = self.up2(d3, f2)
        d2 = self.decoder2(d2)
        
        d1 = self.up1(d2, f1)
        d1 = self.decoder1(d1)
        
        d0 = self.up0(d1, f0)
        d0 = self.decoder0(d0)
        
        output = self.final_conv(d0)
        
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return output


if __name__ == "__main__":
    model_new = HLSR(num_classes=2, pretrained=False)

    model_orig = HLSR(num_classes=2, pretrained=False)
    model_orig.up3 = UpBlock(1024, 768, 384, upsample=True)

    x = torch.randn(1, 3, 512, 512)

    print("=" * 50)
    print("HLSR comparison: original(UpBlock upsample=True) vs new(UpBlock upsample=False)")
    print("=" * 50)

    with torch.no_grad():
        params_new = sum(p.numel() for p in model_new.parameters())
        params_orig = sum(p.numel() for p in model_orig.parameters())
        print(f"Params - New: {params_new / 1e6:.2f} M, Original: {params_orig / 1e6:.2f} M")

        out_new = model_new(x)
        out_orig = model_orig(x)
        print(f"Input: {x.shape}")
        print(f"New output shape: {out_new.shape}")
        print(f"Original output shape: {out_orig.shape}")

    print("=" * 50)

