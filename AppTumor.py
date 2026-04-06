import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torchxrayvision as xrv
from pathlib import Path as path
import gradio as gr
from torchvision.models import densenet121, DenseNet121_Weights

#Brain Tumor detector

LR= 1e-4
BatchSize = 8
numofclasses = 4 #meningioma, no tumor,glioma, pituitary
epochs= 5
data_dir = r"C:\Users\promo\Downloads\braintumormain\Training"
val_dir = r"C:\Users\promo\Downloads\braintumormain\val"
testpic = r"meningioma.jpg"
device = "cpu"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

])

#load dataset
brain_Dataset = datasets.ImageFolder(data_dir,transform = transform)
#pass data in batches and shuffle them
datapasser =DataLoader(brain_Dataset,batch_size=BatchSize,shuffle=True)

model = densenet121(weights=DenseNet121_Weights.DEFAULT)
model.classifier = nn.Linear(model.classifier.in_features, 4)

#Training ground:
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR
#                        )
# for epoch in range(epochs):
#     print(f"Started Epoch {epoch}...")
#     model.train()
#     #fun fact, when you do .train the model dosent load weights.
#     #but you can adjust weights SIMPLY BY loading the saved weight(aka loadstatdic)
#     #and then pass it to device and then train.
#     for x, y in datapasser:
#         #pass the x and y to the model, the model is on the cpu
#         x=x.to(device)
#         y= y.to(device)
#         optimizer.zero_grad()
#         outputs = model(x)
#         loss = criterion(outputs,y)
#         loss.backward()
#         optimizer.step()
#
#     #model evalutaiton
#     """
#     val_dataset = datasets.ImageFolder(val_dir, transform = transform)
#     val_passer = DataLoader(val_dataset, batch_size= BatchSize,shuffle= False)
#
#     #inside loop
#     model.eval()
#     with torch.no_grad():
#         pass
#
#     """
#
#
#     torch.save(model.state_dict(), "brain_model.pth")
#     print(f"Model saved as brain_model.pth, Count: {epoch}")
#

model.load_state_dict(torch.load(r"brain_modelMAIN.pth", map_location=device))
model.to(device)

def preprocess(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    img= image.convert("RGB")
    imgtensor= transform(img).unsqueeze(0).to(device)
    return imgtensor

def predictimage(param):
    #evaluate model
    model.eval()

    #tensor image
    pre_processed_image = preprocess(param)
    with torch.no_grad():
        output = model(pre_processed_image)
        probs = F.softmax(output[0], dim=0)
    classes = brain_Dataset.classes
    resultss = {cls: float(proba) for cls, proba in zip(classes, probs)}
    return resultss


def generate_heatmap(img, class_index=None):

    input_tensor = preprocess(img).to(device)
    if class_index is None:
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            class_index = torch.argmax(outputs[0]).item()

    # GradCAM
    cam = GradCAM(model=model, target_layers=[model.features])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_index)])
    grayscale_cam = grayscale_cam[0, :]  # remove batch dimension

    # prepare original image for overlay
    img_np = np.array(img.resize((224, 224))) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # convert to PIL.Image
    if isinstance(cam_image, np.ndarray):
        cam_image = Image.fromarray(cam_image)

    return cam_image

#test single image in code

# results = predictimage(testpic)
# #.items gets the keys of the valuees
# top5 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
#
# print("Top predictions:")
# for label, prob in top5:
#     print(f"{label}: {prob * 100:.2f}%")

def gradio_interface(img):
    results = predictimage(img)
    classes = brain_Dataset.classes
    top_class = max(results, key=results.get)
    class_idx = classes.index(top_class)
    heatmap = generate_heatmap(img, class_index=class_idx)
    return results, heatmap

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=4), gr.Image(type="pil")],
    title="Brain Tumor Detector",
    description="Upload An Mri image of the brain and Detect if there might be a tumor or not."

)

iface.launch()

