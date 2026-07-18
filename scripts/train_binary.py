
import argparse
import torch
import numpy as np
from tmgl import (
    TMGL, create_tmgl_model,
    TMGLBinaryTrainer,
    load_binary_data, create_dataloaders
)


def main():
    parser = argparse.ArgumentParser(description='Train TMGL for binary classification')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, default='data/', help='Data directory')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/best_model.pt', help='Model save path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    data_path = f"{args.data_path}/{args.dataset}.csv"
    df = load_binary_data(data_path)
    train_loader, val_loader, test_loader = create_dataloaders(df, task='binary', batch_size=args.batch_size)
    
    # Create model
    model = create_tmgl_model(num_classes=2, hidden_dim=args.hidden_dim, task='binary')
    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = TMGLBinaryTrainer(model, device)
    results = trainer.train(
        train_loader, val_loader, test_loader,
        epochs=args.epochs, save_path=args.save_path
    )
    

if __name__ == "__main__":
    main()
