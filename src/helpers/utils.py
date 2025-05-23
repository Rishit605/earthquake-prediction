# Plot training and evaluation history
def plot_loss(train_losses, val_losses):
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss History')
    plt.legend()

    plt.show()