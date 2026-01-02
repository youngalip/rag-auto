import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from collections import Counter
import numpy as np

# ===== IMPROVED MODEL ARCHITECTURE =====
class ImprovedSeq2SeqRAG(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=600, dropout=0.25):
        """
        FIXED: Larger model for better generation
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder - Increased capacity
        self.encoder = nn.LSTM(
            embed_dim, 
            hidden_dim,
            num_layers=3,  # FIXED: 2 → 3 layers
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            embed_dim, 
            hidden_dim * 2,
            num_layers=3,  # FIXED: 2 → 3 layers
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def _combine_bi_hidden(self, h, c):
        num_layers = self.encoder.num_layers
        num_directions = 2 if self.encoder.bidirectional else 1
        H = h.size(2)
        B = h.size(1)

        h_resh = h.view(num_layers, num_directions, B, H)
        c_resh = c.view(num_layers, num_directions, B, H)

        if num_directions == 1:
            h_cat = h_resh[:,0]
            c_cat = c_resh[:,0]
        else:
            h_cat = torch.cat((h_resh[:,0], h_resh[:,1]), dim=2)
            c_cat = torch.cat((c_resh[:,0], c_resh[:,1]), dim=2)

        return h_cat.contiguous(), c_cat.contiguous()
        
    def forward(self, input_seq, target_seq):
        emb_input = self.dropout(self.embed(input_seq))
        _, (h, c) = self.encoder(emb_input)
        
        h, c = self._combine_bi_hidden(h, c)
        
        emb_target = self.dropout(self.embed(target_seq[:, :-1]))
        out, _ = self.decoder(emb_target, (h, c))
        
        return self.fc(out)
    
    def encode(self, input_seq):
        emb_input = self.embed(input_seq)
        _, (h, c) = self.encoder(emb_input)
        h, c = self._combine_bi_hidden(h, c)
        return h, c
    
    def decode_step(self, x, h, c):
        emb = self.embed(x)
        out, (h, c) = self.decoder(emb, (h, c))
        logits = self.fc(out)
        return logits, h, c

# ===== RAG DATASET =====
class RAGDataset(Dataset):
    def __init__(self, data, vocab=None, max_len_input=250, max_len_output=200):
        """
        FIXED: Longer sequences for better answers
        """
        self.data = data
        self.max_len_input = max_len_input  # FIXED: 200 → 250
        self.max_len_output = max_len_output  # FIXED: 150 → 200
        
        if vocab is None:
            all_text = []
            for d in self.data:
                all_text.append(d["question"] + " " + d["context"] + " " + d["answer"])
            
            all_text = " ".join(all_text).lower().split()
            counter = Counter(all_text)
            
            self.word2idx = {
                "<pad>": 0,
                "<sos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "<sep>": 4
            }
            
            # FIXED: Lower min frequency for richer vocabulary
            for w, count in counter.items():
                if count >= 1 and w not in self.word2idx:  # FIXED: 2 → 1
                    self.word2idx[w] = len(self.word2idx)
            
            self.idx2word = {i: w for w, i in self.word2idx.items()}
        else:
            self.word2idx, self.idx2word = vocab
    
    def encode(self, text, max_len, add_special=True):
        tokens = text.lower().split()
        ids = [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]
        
        if add_special:
            ids = [self.word2idx["<sos>"]] + ids + [self.word2idx["<eos>"]]
        
        ids = ids[:max_len]
        ids += [self.word2idx["<pad>"]] * (max_len - len(ids))
        return torch.tensor(ids)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = f"{item['question']} <sep> {item['context']}"
        output_text = item['answer']
        
        return (
            self.encode(input_text, self.max_len_input, add_special=False),
            self.encode(output_text, self.max_len_output, add_special=True)
        )

# ===== K-FOLD TRAINING =====
def train_kfold_rag(
    data_file="rag_training_data.json",
    k=5, 
    epochs=20,  # FIXED: 15 → 20 epochs
    batch_size=8,
    learning_rate=0.0008,  # FIXED: Lower LR for stability
    device=None
):
    """
    FIXED: Better training strategy
    """
    print("=" * 60)
    print("IMPROVED RAG MODEL TRAINING")
    print("=" * 60)
    
    with open(data_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    print(f"Total training samples: {len(all_data)}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    best_loss = float("inf")
    best_fold = -1
    fold_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{k}")
        print('='*60)
        
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        
        train_ds = RAGDataset(train_data)
        val_ds = RAGDataset(val_data, vocab=(train_ds.word2idx, train_ds.idx2word))
        
        print(f"Vocabulary size: {len(train_ds.word2idx)}")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        # FIXED: Larger model
        model = ImprovedSeq2SeqRAG(len(train_ds.word2idx)).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3,  # FIXED: More patience
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=train_ds.word2idx["<pad>"])
        
        epoch_train_losses = []
        epoch_val_losses = []
        
        for epoch in range(epochs):
            # Train
            model.train()
            total_train_loss = 0
            
            for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                
                optimizer.zero_grad()
                output = model(input_seq, target_seq)
                
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    target_seq[:, 1:].reshape(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for input_seq, target_seq in val_loader:
                    input_seq = input_seq.to(device)
                    target_seq = target_seq.to(device)
                    
                    output = model(input_seq, target_seq)
                    loss = criterion(
                        output.reshape(-1, output.size(-1)),
                        target_seq[:, 1:].reshape(-1)
                    )
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            epoch_train_losses.append(avg_train_loss)
            epoch_val_losses.append(avg_val_loss)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            
            scheduler.step(avg_val_loss)
        
        # Save best model
        final_val_loss = epoch_val_losses[-1]
        fold_losses.append(final_val_loss)
        
        if final_val_loss < best_loss:
            best_loss = final_val_loss
            best_fold = fold + 1
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "word2idx": train_ds.word2idx,
                "idx2word": train_ds.idx2word,
                "fold": fold + 1,
                "val_loss": final_val_loss
            }, "model_rag_v2.pth")
            
            print(f"\n✓ Best model saved (Fold {best_fold}, Val Loss: {best_loss:.4f})")
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs+1), epoch_train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, epochs+1), epoch_val_losses, label='Val Loss', marker='s')
        plt.title(f'Training Progress - Fold {fold+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'training_fold_{fold+1}.png')
        plt.close()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best model: Fold {best_fold}")
    print(f"Best val loss: {best_loss:.4f}")
    print(f"\nAll fold losses:")
    for i, loss in enumerate(fold_losses, 1):
        print(f"  Fold {i}: {loss:.4f}")
    print(f"\nAverage: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
    print("\nModel saved as: model_rag_v2.pth")
    print("="*60)

if __name__ == "__main__":
    train_kfold_rag(
        data_file="rag_training_data.json",
        k=5,
        epochs=20,
        batch_size=8,
        learning_rate=0.0008
    )