import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
tflib.init_tf()

with open("karras2019stylegan-ffhq-1024x1024.pkl", 'rb') as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f) 


generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))



orang=np.load("latent_representations/cropped_kamala_01.npy")

smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')
gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
age_direction = np.load('ffhq_dataset/latent_directions/age.npy')

new_latent_vector = orang.copy()

coeff=2

# big question : dapet darimana elemen 1-8 mengubah direction?
new_latent_vector[:8] = (new_latent_vector + coeff*smile_direction)[:8]

new_image=generate_image(new_latent_vector)

new_image.save("gambar_generated.png")