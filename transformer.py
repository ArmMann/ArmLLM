import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoImageProcessor
import torch.nn.functional as F

# Transformer implementation from scratch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Shape of pe: [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Apply the formula
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra dimension to pe so that it can be added to the input embeddings
        pe = pe.unsqueeze(0).requires_grad_(False)  # Shape: [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # pe[:, :x.size(1)] shape: [1, seq_len, d_model]

        x = x + self.pe[:, :x.size(1)]
        #raise NotImplementedError
        # Shape remains [batch_size, seq_len, d_model]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V shapes: [batch_size, num_heads, seq_len, d_k]
        d_k = Q.size(-1)
        # Shape: [batch_size, num_heads, seq_len, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        Q = self.linear_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attention_output)
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Multi-head attention sub-layer with residual connection and layer normalization
        attn_output = self.multi_head_attention(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        # Shape: [batch_size, seq_len, d_model]
         # Feed-forward sub-layer with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

#Integrating every block in Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, img_size, patch_size, d_model, num_heads, num_layers, d_ff, num_classes):
        super(TransformerEncoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        ## added the positional encoding and encoding layers
        self.positional_encoding = PositionalEncoding(d_model, self.num_patches)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
        self.norm = nn.LayerNorm(d_model)

    def patchify(self, images):
        # images shape: [batch_size, channels, height, width]
        batch_size = images.shape[0]
        # patches shape: [batch_size, num_patches, patch_dim]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_dim)
        return patches  # Shape: [batch_size, num_patches, patch_dim]

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = self.patchify(x)  # Shape: [batch_size, num_patches, patch_dim]
        x = self.patch_embedding(x)  # Shape: [batch_size, num_patches, d_model]
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        # TODO: positional embedding, layers, norm,
        x = self.norm(x)
        x = x.mean(dim=1)  # Take the mean across patches , Global average pooling
        return self.fc(x)  # Shape: [batch_size, num_classes]

# Data loading and preprocessing

# Data loading and preprocessing
def load_and_preprocess_data():
    # Load the full dataset and split it
    dataset = load_dataset("chriamue/bird-species-dataset", split="train[:5%]")
    train_dataset = dataset.train_test_split(test_size=0.1)  # 10% for validation

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    def preprocess_image(example):
        inputs = image_processor(example["image"], return_tensors="pt")
        return {"pixel_values": inputs.pixel_values.squeeze(0), "label": example["label"]}
    
    train_dataset["train"] = train_dataset["train"].map(preprocess_image, remove_columns=["image"])
    train_dataset["train"].set_format(type="torch", columns=["pixel_values", "label"])
    
    train_dataset["test"] = train_dataset["test"].map(preprocess_image, remove_columns=["image"])
    train_dataset["test"].set_format(type="torch", columns=["pixel_values", "label"])
    
    return train_dataset["train"], train_dataset["test"]




def validate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["pixel_values"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss:.3f}, Accuracy: {accuracy:.2f}%")
    
# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total = 0
    correct = 0
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        inputs, labels = batch["pixel_values"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Print outputs for debugging
        if batch_idx == 0:
            print("Model outputs:", outputs)
            print("Labels:", labels)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    average_loss = total_loss / len(dataloader)
    print(f"Training loss: {average_loss:.3f}, Accuracy: {accuracy:.2f}%")

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    img_size = 224
    patch_size = 16
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 1024
    num_classes = 525
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
    # Load and preprocess data
    train_data, validation_data = load_and_preprocess_data()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TransformerEncoder(img_size, patch_size, d_model, num_heads, num_layers, d_ff, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training and validation loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device)
        validate(model, validation_loader, criterion, device)

if __name__ == "__main__":
    main()
