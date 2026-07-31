import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

class SimpleANN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleANN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, offsets):
        embedded = self.embedding(text)
        # Average pooling for varying length sentences
        pooled = torch.zeros(len(offsets), embedded.size(1), device=embedded.device)
        for i in range(len(offsets)):
            start = offsets[i]
            end = offsets[i+1] if i < len(offsets) - 1 else text.size(0)
            if start < end:
                pooled[i] = embedded[start:end].mean(dim=0)
        
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out).squeeze()

def build_vocab(datasets, vocab_size=10000):
    counter = Counter()
    for text in datasets['train']['text']:
        counter.update(text.lower().split())
    
    # 0 is padding, 1 is unk
    vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common(vocab_size - 2))}
    return vocab

def text_pipeline(text, vocab):
    return [vocab.get(token, 1) for token in text.lower().split()]

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.float32)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'model')
    results_dir = os.path.join(base_dir, 'results')
    report_dir = os.path.join(base_dir, 'report')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print("Loading IMDB dataset via Hugging Face datasets...")
    dataset = load_dataset('stanfordnlp/imdb')
    
    print("Building vocabulary...")
    vocab_size = 10000
    vocab = build_vocab(dataset, vocab_size)

    # Process dataset
    print("Processing dataset...")
    train_data = [(text_pipeline(item['text'], vocab), item['label']) for item in tqdm(dataset['train'])]
    test_data = [(text_pipeline(item['text'], vocab), item['label']) for item in tqdm(dataset['test'])]

    batch_size = 256
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleANN(vocab_size=vocab_size, embed_dim=16, hidden_dim=16, output_dim=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 10
    history = {'loss': [], 'accuracy': []}

    print("Training model...")
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for labels, text, offsets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels, text, offsets = labels.to(device), text.to(device), offsets.to(device)
            optimizer.zero_grad()
            
            outputs = model(text, offsets)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_acc += (predicted == labels).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = total_acc / len(train_data)
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    print("Evaluating model...")
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for labels, text, offsets in tqdm(test_loader, desc="Evaluating"):
            labels, text, offsets = labels.to(device), text.to(device), offsets.to(device)
            outputs = model(text, offsets)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            test_acc += (predicted == labels).sum().item()
            
    final_test_loss = test_loss / len(test_loader)
    final_test_acc = test_acc / len(test_data)
    print(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.4f}")

    # Save model
    model_path = os.path.join(model_dir, 'sentiment_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['loss'], 'b-')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['accuracy'], 'b-')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plot_path = os.path.join(results_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")

    # Report
    report_path = os.path.join(report_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write("IMDB Movie Review Sentiment Analysis - Model Report\n")
        f.write("="*60 + "\n\n")
        f.write("Architecture:\n")
        f.write("- Embedding layer (10000 vocab size, 16 dimensions)\n")
        f.write("- GlobalAveragePooling (Custom implementation)\n")
        f.write("- Dense layer (16 units, ReLU activation)\n")
        f.write("- Dense output layer (1 unit, Sigmoid activation)\n\n")
        f.write("Evaluation Results:\n")
        f.write(f"- Test Loss: {final_test_loss:.4f}\n")
        f.write(f"- Test Accuracy: {final_test_acc:.4f}\n\n")
        f.write("The model was built using PyTorch and achieved solid accuracy on the test set.\n")
    
    print(f"Report saved to {report_path}")

if __name__ == '__main__':
    main()
