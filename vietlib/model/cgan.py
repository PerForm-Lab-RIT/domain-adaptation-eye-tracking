
import torch
import torch.nn as nn

class GLU(nn.Module):
  """Gated Linear Unit
  """
  def __init__(self):
      super(GLU, self).__init__()

  def forward(self, x):
      return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
  def __init__(self, in_features, glu=False):
    super(ResidualBlock, self).__init__()
    self.glu = glu
    self.block = self.res_block(in_features)
    if glu:
      self.gate_block = self.res_block(in_features)
      self.out = self.res_block(in_features)

  def res_block(self, in_features):
    return nn.Sequential(
      nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
      nn.Conv2d(in_features, in_features, 3),
      nn.InstanceNorm2d(in_features), 
      nn.ReLU(inplace=True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_features, in_features, 3),
      nn.InstanceNorm2d(in_features)
    )

  def forward(self, x):
    if self.glu:
      x1 = self.block(x)
      gate = self.gate_block(x)
      glu = x1 * torch.sigmoid(gate)
      x2 = self.out(glu)
      return x + x2
    else:
      return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, n_segment_channels, num_residual_block, use_segment=False):
        super(GeneratorResNet, self).__init__()
        
        self.use_segment = use_segment

        channels = input_shape[0]
        
        if use_segment:
          inp_channels = channels + n_segment_channels
        else:
          inp_channels = channels

        # Initial Convolution Block
        out_features = 64
        model = [
            nn.Conv2d(inp_channels, out_features, 5, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]
            
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2), # --> width*2, heigh*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        if not self.use_segment:
          # Output Layer
          model += [nn.ReflectionPad2d(1),
                    nn.Conv2d(out_features, 1, 3),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(1, 1, 3),
                    nn.Tanh()
                  ]
        else:
          model += [nn.ReflectionPad2d(1),
                    nn.Conv2d(out_features, n_segment_channels, 3),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(n_segment_channels, 1, 3),
                    nn.Tanh()
                  ]
        
        # Unpacking
        self.model = nn.Sequential(*model) 

    def forward(self, img, hot_segment=None):
        if self.use_segment:
          assert hot_segment is not None
          inp = torch.cat((img, hot_segment), dim=1)
          gen_img = self.model(inp)
          # TODO: Do we use sum or max?
          # gen_img, gen_img_indices = torch.max(gen_img * hot_segment, dim=1, keepdim=True)
          # gen_img = torch.sum(gen_img * hot_segment, dim=1, keepdim=True)
        else:
          gen_img = self.model(img)
        return gen_img

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        channels, height, width = input_shape
        
        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height//2**4, width//2**4)
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128,256),
            *discriminator_block(256,512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)