import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BotGCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) for Bot Detection.
    
    Architecture:
    - Input Layer: Graph Convolution (num_features -> hidden_channels)
    - Activation: ReLU
    - Regularization: Dropout
    - Output Layer: Graph Convolution (hidden_channels -> num_classes)
    
    Attributes:
        num_features (int): Number of input features per node (e.g., 4: Entropy, Age, Time, Heuristic).
        hidden_channels (int): Size of the hidden layer (optimized to 64).
        dropout_rate (float): Probability of zeroing elements (optimized to 0.6).
        num_classes (int): Number of output classes (2: Human, Bot).
    """
    def __init__(self, num_features, hidden_channels=64, dropout_rate=0.6, num_classes=2):
        super(BotGCN, self).__init__()
        torch.manual_seed(12345) # Ensure reproducibility
        
        self.dropout_rate = dropout_rate
        
        # Layer 1: Input to Hidden
        self.conv1 = GCNConv(num_features, hidden_channels)
        
        # Layer 2: Hidden to Output (Logits)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Node feature matrix [num_nodes, num_features]
            edge_index (LongTensor): Graph connectivity in COO format [2, num_edges]
            
        Returns:
            Tensor: Raw logits [num_nodes, num_classes]
        """
        # 1. First Graph Convolution
        x = self.conv1(x, edge_index)
        x = x.relu()
        
        # 2. Dropout for regularization (Prevent Overfitting)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 3. Second Graph Convolution
        x = self.conv2(x, edge_index)
        
        # Note: We return raw logits. CrossEntropyLoss in PyTorch expects logits, 
        # not probabilities (no Softmax needed here).
        return x