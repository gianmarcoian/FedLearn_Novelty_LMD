import torch
import matplotlib.pyplot as plt

# Carica il file .pth
file_path = 'image_in.pth'
data = torch.load(file_path)
print("Chiavi nel file .pth:", data.keys())

# Stampa le forme dei tensori per ciascuna chiave
for key in data.keys():
    if isinstance(data[key], torch.Tensor):
        print(f"{key}: {data[key].shape}")
    else:
        print(f"{key}: {type(data[key])}")
# Assumi che data contenga le chiavi 'origin', 'masked', e 'recon'
origin = data['orig']
masked = data['masked']  # Deve essere una lista o un array di 5 immagini
recon = data['recon']    # Deve essere una lista o un array di 5 immagini

# Controlla che i dati siano nel formato corretto

assert len(masked) == 5, "Ci dovrebbero essere 5 immagini mascherate"
assert len(recon) == 5, "Ci dovrebbero essere 5 immagini ricostruite"

# Funzione per visualizzare le immagini
def show_images(images, title):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for img, ax in zip(images, axes):
        if img.dim() == 4:  # Batch dimension presente
            img = img[0]    # Rimuovi la dimensione batch
        ax.imshow(img.permute(1, 2, 0))  # Converti da (C, H, W) a (H, W, C)
        ax.axis('off')
    fig.suptitle(title)
    plt.show()

# Verifica se la dimensione batch è presente e rimuovila
if origin.dim() == 4:
    origin = origin[0]

show_images([origin], 'Origin')
show_images(masked, 'Masked')
show_images(recon, 'Reconstructed')
