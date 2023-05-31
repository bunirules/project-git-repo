import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.utils import save_image
plt.style.use('ggplot')
def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    Note that the output and label pixels (when dealing with images) should
    be normalized as the `max_val` here is 1 and not 255.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR
def save_plot(train_loss, val_loss, train_psnr, val_psnr, save_path):
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path+'/loss.png')
    plt.close()
    # PSNR plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(save_path+'/psnr.png')
    plt.close()
def save_model_state(model, save_path):
    # save the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), save_path+'/model.pth')
def save_model(epochs, model, optimizer, criterion, save_path):
    """
    Function to save the trained model to disk.
    """
    # Remove the last model checkpoint if present.
    torch.save({
                'epoch': epochs+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_path+"/model_ckpt.pth")
def save_validation_results(image_data, outputs, epoch, batch_iter,save_path):
    """
    Function to save the validation reconstructed images.
    """
    save_image(
        image_data, 
        save_path+f"/valid_results/val_sr_{epoch}_{batch_iter}_input.png"
    )
    save_image(
        outputs, 
        save_path+f"/valid_results/val_sr_{epoch}_{batch_iter}_output.png"
    )