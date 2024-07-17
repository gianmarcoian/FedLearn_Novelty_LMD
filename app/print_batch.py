import torch
import matplotlib.pyplot as plt
import os

def save_combined_images(batch_file, output_dir, num_images=10):
    batch_data = torch.load(batch_file)
    orig_images = batch_data['orig']
    masked_images = batch_data['masked']
    recon_images = batch_data['recon']

    for i in range(orig_images.shape[0]):
        fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

        # Converti l'immagine originale e mascherata in formato numpy per matplotlib
        orig_img = orig_images[i].permute(1, 2, 0).numpy()
        masked_img = masked_images[0][i].permute(1, 2, 0).numpy()

        # Normalizzazione dei valori delle immagini
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        masked_img = (masked_img - masked_img.min()) / (masked_img.max() - masked_img.min())

        for j in range(num_images):
            # Visualizza l'immagine originale
            axs[j, 0].imshow(orig_img, cmap='gray')
            axs[j, 0].set_title('Original Image')
            axs[j, 0].axis('off')

            # Visualizza l'immagine mascherata
            axs[j, 1].imshow(masked_img, cmap='gray')
            axs[j, 1].set_title('Masked Image')
            axs[j, 1].axis('off')

            # Visualizza l'immagine ricostruita
            recon_img = recon_images[j][i].permute(1, 2, 0).numpy()
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
            axs[j, 2].imshow(recon_img, cmap='gray')
            axs[j, 2].set_title(f'Reconstructed Image {j + 1}')
            axs[j, 2].axis('off')

        plt.tight_layout()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'image_{i}.jpeg')
        plt.savefig(output_file, format='jpeg')
        plt.close(fig)
batch_file_path = 'image_out.pth'
output_directory = 'confronto/all_recon_732'

save_combined_images(batch_file_path, output_directory, num_images=10)
