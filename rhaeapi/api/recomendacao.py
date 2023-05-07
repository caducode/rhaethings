import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import os
import pickle
from transformers import CLIPProcessor, CLIPModel
import torch
import requests
import uuid
#import cv2
from deepface import DeepFace
#from IPython import display
from math import sqrt


#definicoes de path
emb_path = "/home/cadu/Documents/projeto_arezzo/2023/rhae_api/rhae_api/assets/embeddings_full.data"
#imgs_base = "/home/cadu/Documents/projeto_arezzo/2023/demo_recomendacao/celebridades_imdb"
#profile_base = "/home/cadu/Documents/projeto_arezzo/2023/demo_recomendacao/produtos_celebridades"
#img_src_path =  "/home/cadu/Documents/projeto_arezzo/2023/demo_recomendacao/arezzo_full"



## Carga dos embbedings para calculo de similaridade dos produtos ##
with open(emb_path, "rb") as f:
    data = pickle.load(f)
itens = data["embeddings"]
itens /= itens.norm(dim=-1, keepdim=True)
data["names"] = data["names"][: data["embeddings"].shape[0] :]


def extract_one_img(path_img):
    with torch.no_grad():
        device = "cuda"
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.to(device)
        image = Image.open(path_img)

        inputs = processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        embeddings = model.get_image_features(**inputs)
        cpu = embeddings.to("cpu")
        name_img = os.path.basename(path_img)

    return name_img, cpu


#busca cliente
def busca_cliente(url_foto):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if(url_foto):
        sym_faces_df = DeepFace.find(img_path = url_foto, db_path = imgs_base, enforce_detection=False) #enforce_detection=False
        f, axarr = plt.subplots(2, 2, figsize=(10,10))
        curr_row = 0
        for index, row in enumerate(list(sym_faces_df[0].identity[0:4])):
            print(index)

##busca_cliente("imagem_camera.png")

def busca_produtos(qtd_rec=5,img_path=""):
    qtd_rec  = int(qtd_rec)
    if(img_path):
        response = requests.get(img_path, stream = True)
        if(response.status_code == 200):
            type_img = response.headers['content-type'].split('/')[1]
            img = Image.open(response.raw)
            filename = uuid.uuid4().hex
            img.save(os.path.join("/home/cadu/Documents/projeto_arezzo/2023/rhae_api/rhae_api/assets/fotos",filename+"."+type_img))
    
    img_name, tensor = extract_one_img(os.path.join("/home/cadu/Documents/projeto_arezzo/2023/rhae_api/rhae_api/assets/fotos",filename+"."+type_img))
    item = tensor[0]
  
    similarity = np.matmul(item.numpy(), itens.numpy().T)
    topn_index = np.argpartition(similarity, -(qtd_rec+2))[-(qtd_rec+2):]
    results = [
        {"index": topn_index[i], "score": similarity[topn_index[i]]}
        for i in range(qtd_rec)
    ]

    results = sorted(results, key=lambda d: d["score"], reverse=True)
    recommendations = [data['names'][result['index']] for i,result in enumerate(results)]
    return recommendations