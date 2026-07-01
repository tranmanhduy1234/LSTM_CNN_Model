import torch
import torch.nn as nn

class FeatureExtractorCNN(nn.Module):
    """
    Custom CNN designed to extract spatial features from facial crops (eyes/mouth).
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
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling -> 128
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

class DriverStateCNNLSTM(nn.Module):
    """
    Combined CNN-LSTM architecture for Driver State (Drowsiness) Recognition.
    
    Inputs:
        - eye_left_seq:  (batch, seq_len, 3, 64, 64)
        - eye_right_seq: (batch, seq_len, 3, 64, 64)
        - mouth_seq:     (batch, seq_len, 3, 64, 64)
        - geom_seq:      (batch, seq_len, geom_dim) (where geom_dim = 10)
    
    Output:
        - probability: (batch, 1) probability of being drowsy
    """
    def __init__(self, cnn_feature_dim=64, geom_dim=10, lstm_hidden_dim=128, lstm_layers=2):
        super(DriverStateCNNLSTM, self).__init__()
        
        # Share weights between left and right eye feature extractors (Siamese)
        self.eye_cnn = FeatureExtractorCNN(out_features=cnn_feature_dim)
        
        # Separate extractor for the mouth (which has different patterns)
        self.mouth_cnn = FeatureExtractorCNN(out_features=cnn_feature_dim)
        
        # Fused spatial feature dimension: 
        # Left eye cnn (cnn_feature_dim) + Right eye cnn (cnn_feature_dim) + Mouth cnn (cnn_feature_dim) + Geometric feats (geom_dim)
        self.spatial_dim = cnn_feature_dim * 3 + geom_dim
        
        # Fusing layer to reduce feature size and map linear relationship before LSTM
        self.fuse_layer = nn.Sequential(
            nn.Linear(self.spatial_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM layer to model temporal sequence
        # batch_first=True -> Input shape: (batch, seq_len, lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0.0
        )
        
        # Classification head to output drowsiness probability
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid() # Drowsiness probability: [0.0, 1.0]
        )

    def forward(self, eye_left_seq, eye_right_seq, mouth_seq, geom_seq):
        batch_size, seq_len, C, H, W = eye_left_seq.size()
        
        # Reshape sequence batches for CNN processing: (batch_size * seq_len, C, H, W)
        eye_left_flat = eye_left_seq.contiguous().view(-1, C, H, W)
        eye_right_flat = eye_right_seq.contiguous().view(-1, C, H, W)
        mouth_flat = mouth_seq.contiguous().view(-1, C, H, W)
        
        # Extract features for all frames in parallel
        eye_left_feats = self.eye_cnn(eye_left_flat)
        eye_right_feats = self.eye_cnn(eye_right_flat)
        mouth_feats = self.mouth_cnn(mouth_flat)
        
        # Reshape back to sequence: (batch_size, seq_len, feature_dim)
        eye_left_feats = eye_left_feats.view(batch_size, seq_len, -1)
        eye_right_feats = eye_right_feats.view(batch_size, seq_len, -1)
        mouth_feats = mouth_feats.view(batch_size, seq_len, -1)
        
        # Concatenate spatial features from eyes, mouth, and geometric vector
        # geom_seq shape: (batch_size, seq_len, geom_dim)
        spatial_feats = torch.cat([eye_left_feats, eye_right_feats, mouth_feats, geom_seq], dim=2)
        
        # Project spatial features
        fused_feats = self.fuse_layer(spatial_feats) # (batch_size, seq_len, lstm_hidden_dim)
        
        # Pass to LSTM
        # lstm_out shape: (batch_size, seq_len, lstm_hidden_dim)
        # h_n shape: (num_layers, batch_size, lstm_hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(fused_feats)
        
        # Take the output of the last time step for classification
        last_step_out = lstm_out[:, -1, :] # (batch_size, lstm_hidden_dim)
        
        # Classify
        prob = self.classifier(last_step_out)
        return prob
