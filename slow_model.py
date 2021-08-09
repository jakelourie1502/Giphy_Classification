import torch 

class ResidualBlock(torch.nn.Module):

    '''
    Base class for residual blocks, used in _make_layer function to create residual layers for the Slow model [only]

    There are 2 block versions:
        - block_ver=1 -> SlowPath res2 and res3 in slow
        - block_ver=2 -> rest of the SlowPath
    '''

    def __init__(self,
                 in_channels,
                 intermediate_channels,
                 block_ver,
                 identity_downsample=None,
                 stride=1
                 ):
        super(ResidualBlock, self).__init__()  
        # number of channels after a block is always *4 what it was when it entered
        self.expansion = 4

        if block_ver == 1:
            self.conv1 = torch.nn.Conv3d(
                in_channels, 
                intermediate_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=True
            )
        elif block_ver == 2:
            self.conv1 = torch.nn.Conv3d(
                in_channels, 
                intermediate_channels, 
                kernel_size=(3,1,1), 
                stride=1, 
                padding=(1,0,0), 
                bias=True
            )
        self.bn1 = torch.nn.BatchNorm3d(intermediate_channels)

        self.conv2 = torch.nn.Conv3d(
            intermediate_channels,  # here the in and out channels are the same, value after first layer in the block
            intermediate_channels, 
            kernel_size=(1,3,3), 
            stride=stride,
            padding=(0,1,1),
            bias=True
        )
        self.bn2 = torch.nn.BatchNorm3d(intermediate_channels)

        self.conv3 = torch.nn.Conv3d(
            intermediate_channels, 
            intermediate_channels*self.expansion, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
        self.bn3 = torch.nn.BatchNorm3d(intermediate_channels*self.expansion)
        self.relu = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv3d(
            intermediate_channels, 
            intermediate_channels*self.expansion, 
            kernel_size=1, 
            stride=stride, 
            padding=0, 
            bias=True
        )


    def forward(self, x):
        # We enter the block with x having 'input channels shape, e.g. 256'
        out = self.conv1(x) #goes to 128 e.g. 
        out = self.bn1(out) 
        out = self.relu(out)
        identity = out # identity is at 128

        out = self.conv2(out) #stride of 2 here shrinks spatial size
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) # scales up to 512
        
        identity = self.conv4(identity) # scales up channels to 512 and shrinks spatial size by 2.
        
        out += identity         
        return self.relu(out)

class SlowNet(torch.nn.Module):

    def __init__(self, 
                 image_channels=3, 
                 number_of_classes=10
                 ):
        super(SlowNet, self).__init__()
        
        self.in_channels = 64

        # Slow layers:
        self.data_layer_slow = torch.nn.Conv3d(
            in_channels=image_channels, 
            out_channels=image_channels, 
            kernel_size=1, 
            stride=(6,1,1),
            padding=(0,0,0)
        )

        self.conv1_slow = torch.nn.Conv3d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=(1,7,7), 
            stride=(1,3,3)
        )
  
        self.max_pool_slow = torch.nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2))
        # _make_residual(number_of_blocks, block_ver, intermediate_channels, stride)
        self.res2_slow = self._make_residual_layer(3, 1, 64) 

        self.res3_slow = self._make_residual_layer(4, 1, 128, (1,2,2))
        # COMMENT: stride should not be 1 here, but can't decide what it should be, checked their implementation but its not very clear, maybe 1x2x2, as per previous one

        self.res4_slow = self._make_residual_layer(6, 2, 256, (1,2,2))
      
        self.res5_slow = self._make_residual_layer(3, 2, 512, (1,2,2))

        self.adaptavgpool_slow = torch.nn.AdaptiveAvgPool3d(1)
        self.flat_slow = torch.nn.Flatten()
        self.linear_slow = torch.nn.Linear(2048, number_of_classes)


    def forward(self, X):

        X_slow = self.data_layer_slow(X)
        X_slow = self.conv1_slow(X_slow)
        X_slow = self.max_pool_slow(X_slow)
        X_slow = self.res2_slow(X_slow)
        X_slow = self.res3_slow(X_slow)
        X_slow = self.res4_slow(X_slow)
        X_slow = self.res5_slow(X_slow)

        X_slow = self.adaptavgpool_slow(X_slow)
        X_slow = self.flat_slow(X_slow)
        X_slow = self.linear_slow(X_slow)
        return X_slow

    def _make_residual_layer(self,
                             number_of_blocks,
                             block_ver,
                             intermediate_channels,
                             stride=1
                             ): 
        """
        as per paper the first layer in block has stride of 1, and then of 2
        """
        layers = []
        
        layers.append(
            ResidualBlock(self.in_channels, intermediate_channels, block_ver, stride)
        )
        self.in_channels = intermediate_channels*4

        for i in range(number_of_blocks -1):
            layers.append(ResidualBlock(self.in_channels, intermediate_channels, block_ver))
        return torch.nn.Sequential(*layers)