from model import train_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST CNN model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save-path', type=str, default='mnist_cnn.pth', help='Path to save the model')
    
    args = parser.parse_args()
    
    print(f"Training MNIST CNN with {args.epochs} epochs...")
    model, train_losses, train_accuracies, test_accuracy = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
    
    print(f"\nTraining completed!")
    print(f"Final test accuracy: {test_accuracy:.2f}%")