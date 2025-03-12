# rsc_module/embedding_classifier.py

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class AttentionFusion(nn.Module):
    """
    Applies an attention mechanism to a set of embeddings.
    Expects input of shape (batch, n, embed_dim) and returns a weighted sum (batch, embed_dim).
    """
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Linear(embed_dim, 1)
      
    def forward(self, x):
        # x shape: (batch, n, embed_dim)
        attn_weights = self.attn(x)  # shape: (batch, n, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # shape: (batch, n, 1)
        # Compute weighted sum of embeddings
        fused = torch.sum(attn_weights * x, dim=1)  # shape: (batch, embed_dim)
        return fused

class RSCClassifier(nn.Module):
    """
    A high-complexity classifier that takes a composite embedding as input
    and outputs a relevance score between 0 and 1.
    """
    def __init__(self, input_dim=512, hidden_dims=[256, 128], dropout=0.3):
        super(RSCClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, x):
        return self.model(x)

class RSCSystem:
    """
    The system that handles embedding extraction, attention-based fusion,
    dimensionality reduction, and relevance scoring.
    """
    def __init__(self, device='cpu'):
        self.device = device
        # Load the pretrained SentenceTransformer model.
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embed_dim = 384  # Each embedding is 384-dimensional.
        self.num_inputs = 3   # We have three inputs: query, result, and user role.
        
        # Initialize the attention-based fusion module.
        self.fusion = AttentionFusion(embed_dim=self.embed_dim).to(self.device)
        # After concatenation: 3*384 = 1152, plus fused vector (384) = 1536.
        # We reduce this composite vector to 512 dimensions.
        self.dim_reducer = nn.Linear(1536, 512).to(self.device)
        
        # Initialize the classifier.
        self.classifier = RSCClassifier(input_dim=512).to(self.device)
        # (In production, you would load trained weights.)

    def get_composite_embedding(self, query_text, query_result, user_role):
        """
        Generates embeddings for the query, query result, and user role,
        applies attention-based fusion, and reduces the dimension to a composite feature vector.
        
        Args:
            query_text (str): The SQL query text.
            query_result (str): The textual representation of the query result.
            user_role (str): The role of the user.
        
        Returns:
            torch.Tensor: Composite embedding tensor with shape (1, 512).
        """
        # Obtain embeddings for each input.
        query_embedding = self.embedder.encode(query_text, convert_to_tensor=True)  # Shape: (384,)
        result_embedding = self.embedder.encode(query_result, convert_to_tensor=True)  # Shape: (384,)
        role_embedding = self.embedder.encode(user_role, convert_to_tensor=True)  # Shape: (384,)
        
        # Create a concatenated embedding: shape (1152,)
        concatenated = torch.cat([query_embedding, result_embedding, role_embedding], dim=0)
        concatenated = concatenated.unsqueeze(0)  # Shape: (1, 1152)
        
        # Stack the embeddings for attention fusion: shape (1, 3, 384)
        embeddings_stack = torch.stack([query_embedding, result_embedding, role_embedding], dim=0)
        embeddings_stack = embeddings_stack.unsqueeze(0)  # Shape: (1, 3, 384)
        fused = self.fusion(embeddings_stack)  # Shape: (1, 384)
        
        # Combine the concatenated and fused representations: shape (1, 1152 + 384 = 1536)
        combined = torch.cat([concatenated, fused], dim=1)
        
        # Reduce dimensionality to 512.
        composite_embedding = self.dim_reducer(combined)  # Shape: (1, 512)
        return composite_embedding.to(self.device)
    
    def score_query(self, query_text, query_result, user_role):
        """
        Computes the relevance score for a given query based on the composite embedding.
        
        Args:
            query_text (str): The SQL query.
            query_result (str): The query result text.
            user_role (str): The role of the user.
        
        Returns:
            float: The relevance score (between 0 and 1).
        """
        composite_embedding = self.get_composite_embedding(query_text, query_result, user_role)
        score = self.classifier(composite_embedding)
        return score.item()
