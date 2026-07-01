import torch
import torch.nn as nn

class FeatureExtractorCNN(nn.Module):
    """
    Custom 2D CNN designed to extract spatial features from facial crops (eyes/mouth).
    Input shape: (N, 3, 64, 64)
    Output shape: (N, feature_dim)
    """
    def __init__(self, out_features=64):
        super(FeatureExtractorCNN, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling -> 128D
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, out_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        features = features.view(features.size(0), -1) # Flatten
        out = self.fc(features)
        return out

class GeometricFeatureExtractor1DCNN(nn.Module):
    """
    Custom 1D CNN branch to extract local temporal patterns from raw geometric sequences 
    (EAR, MAR, Head Pose angles, and relations) before spatial-temporal fusion.
    Input shape: (batch, seq_len, geom_dim)
    Output shape: (batch, seq_len, out_channels)
    """
    def __init__(self, geom_dim=10, out_channels=64):
        super(GeometricFeatureExtractor1DCNN, self).__init__()
        
        self.conv1d_block = nn.Sequential(
            # First 1D Conv layer
            nn.Conv1d(in_channels=geom_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            
            # Second 1D Conv layer to map to final dimension
            nn.Conv1d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, geom_dim)
        # Transpose to (batch, geom_dim, seq_len) for PyTorch Conv1d
        x = x.transpose(1, 2)
        
        out = self.conv1d_block(x)
        
        # Transpose back to (batch, seq_len, out_channels)
        out = out.transpose(1, 2)
        return out

class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism to calculate importance weights for each frame 
    in the temporal sequence, preventing short but critical events from being lost.
    Input shape: (batch, seq_len, hidden_dim)
    Outputs:
        - context: (batch, hidden_dim) weighted average representation
        - weights: (batch, seq_len, 1) attention weights per frame
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_linear = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim, 1))
        
    def forward(self, lstm_out):
        # lstm_out shape: (batch, seq_len, hidden_dim)
        
        # Projection
        u_t = torch.tanh(self.attn_linear(lstm_out)) # (batch, seq_len, hidden_dim)
        
        # Alignment scores
        scores = torch.matmul(u_t, self.context_vector) # (batch, seq_len, 1)
        
        # Softmax over time step dimension (seq_len)
        attention_weights = torch.softmax(scores, dim=1) # (batch, seq_len, 1)
        
        # Context vector via weighted average
        # (batch, seq_len, hidden_dim) * (batch, seq_len, 1) -> sum over seq_len
        context = torch.sum(lstm_out * attention_weights, dim=1) # (batch, hidden_dim)
        
        return context, attention_weights

class DriverStateCNNLSTM(nn.Module):
    """
    Research-grade custom CNN-LSTM model with weight-sharing Siamese Eye extractors,
    a 1D CNN branch for geometric features, and a Temporal Attention module.
    
    Inputs:
        - eye_left_seq:  (batch, seq_len, 3, 64, 64)
        - eye_right_seq: (batch, seq_len, 3, 64, 64)
        - mouth_seq:     (batch, seq_len, 3, 64, 64)
        - geom_seq:      (batch, seq_len, geom_dim)
    
    Output:
        - probability: (batch, 1) drowsiness risk probability
    """
    def __init__(self, cnn_feature_dim=64, geom_dim=10, lstm_hidden_dim=128, lstm_layers=2):
        super(DriverStateCNNLSTM, self).__init__()
        
        # Siamese Eye CNN Branch (Weight-sharing)
        self.eye_cnn = FeatureExtractorCNN(out_features=cnn_feature_dim)
        
        # Independent Mouth CNN Branch
        self.mouth_cnn = FeatureExtractorCNN(out_features=cnn_feature_dim)
        
        # 1D CNN branch for Geometric Feature sequence
        self.geom_conv_dim = 64
        self.geom_cnn1d = GeometricFeatureExtractor1DCNN(geom_dim=geom_dim, out_channels=self.geom_conv_dim)
        
        # Spatial Feature Fusion (Left Eye + Right Eye + Mouth + 1D Conv Geom)
        # 64 + 64 + 64 + 64 = 256
        self.spatial_dim = cnn_feature_dim * 3 + self.geom_conv_dim
        
        # Linear projection to feed the LSTM
        self.fuse_layer = nn.Sequential(
            nn.Linear(self.spatial_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM Temporal Layer
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0.0
        )
        
        # Temporal Attention Layer
        self.attention = TemporalAttention(hidden_dim=lstm_hidden_dim)
        
        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, eye_left_seq, eye_right_seq, mouth_seq, geom_seq):
        batch_size, seq_len, C, H, W = eye_left_seq.size()
        
        # 1. Spatial Feature Extraction (CNN)
        # Reshape to (batch_size * seq_len, C, H, W) for parallel frame-by-frame processing
        eye_left_flat = eye_left_seq.contiguous().view(-1, C, H, W)
        eye_right_flat = eye_right_seq.contiguous().view(-1, C, H, W)
        mouth_flat = mouth_seq.contiguous().view(-1, C, H, W)
        
        # Forward pass through CNNs
        eye_left_feats = self.eye_cnn(eye_left_flat).view(batch_size, seq_len, -1)
        eye_right_feats = self.eye_cnn(eye_right_flat).view(batch_size, seq_len, -1)
        mouth_feats = self.mouth_cnn(mouth_flat).view(batch_size, seq_len, -1)
        
        # 2. Geometric Feature Extraction (1D CNN)
        # Takes (batch, seq_len, geom_dim) -> Outputs (batch, seq_len, geom_conv_dim)
        geom_feats = self.geom_cnn1d(geom_seq)
        
        # 3. Spatial Fusion
        spatial_feats = torch.cat([eye_left_feats, eye_right_feats, mouth_feats, geom_feats], dim=2)
        fused_feats = self.fuse_layer(spatial_feats) # (batch, seq_len, lstm_hidden_dim)
        
        # 4. Temporal Modeling (LSTM)
        # lstm_out shape: (batch, seq_len, lstm_hidden_dim)
        lstm_out, _ = self.lstm(fused_feats)
        
        # 5. Temporal Attention
        # attention_out shape: (batch, lstm_hidden_dim)
        attention_out, attn_weights = self.attention(lstm_out)
        
        # 6. Classification
        prob = self.classifier(attention_out)
        
        return prob
