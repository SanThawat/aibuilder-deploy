import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gdown
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from skimage.color import label2rgb
import torchvision.transforms.functional as TF

# Define model classes and other functions (as in your code)...

# Define the model classes
class res_conv(nn.Module):
    def __init__(self, input_channels, output_channels, down=True):
        super(res_conv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Dropout(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Dropout(0.1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x1
        return x2

class start_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(start_conv, self).__init__()
        self.conv = res_conv(input_channels, output_channels)

    def forward(self, x):
        x = self.conv(x)
        return x

class down_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(down_conv, self).__init__()
        self.conv = nn.Sequential(nn.MaxPool2d(2),
                                  res_conv(input_channels, output_channels))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(up_conv, self).__init__()
        self.up = nn.ConvTranspose2d(input_channels // 2, input_channels // 2, kernel_size=2, stride=2)
        self.conv = res_conv(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff1 = x2.shape[2] - x1.shape[2]
        diff2 = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, pad=(diff1 // 2, diff1 - diff1 // 2, diff2 // 2, diff2 - diff2 // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class stop_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(stop_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        return x

class ResUnet(nn.Module):
    def __init__(self):
        super(ResUnet, self).__init__()
        self.inc = start_conv(1, 64)
        self.down1 = down_conv(64, 128)
        self.down2 = down_conv(128, 256)
        self.down3 = down_conv(256, 512)
        self.down4 = down_conv(512, 512)
        self.up1 = up_conv(1024, 256)
        self.up2 = up_conv(512, 128)
        self.up3 = up_conv(256, 64)
        self.up4 = up_conv(128, 64)
        self.outc = stop_conv(64, 1)

    def forward(self, x):
        xin = self.inc(x)
        xd1 = self.down1(xin)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)
        xu1 = self.up1(xd4, xd3)
        xu2 = self.up2(xu1, xd2)
        xu3 = self.up3(xu2, xd1)
        xu4 = self.up4(xu3, xin)
        out = self.outc(xu4)
        return out

def load_model():
    model = ResUnet()
    url = 'https://drive.google.com/uc?id=16Ccgf-sS7c_4hYtXQYrI1jGXC285e3L0'
    output = 'best_unet_01062024_v2.pth'
    gdown.download(url, output, quiet=False)
    model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))
    model.eval()
    return model

def im_converterX(tensor):
    tensor = tensor.squeeze()  # Remove the extra dimensions
    if tensor.ndim == 2:  # If the image is grayscale, add a dummy channel dimension
        tensor = tensor.unsqueeze(0)
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)  # Reorder the dimensions
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

def im_converterY(tensor):
    tensor = tensor.squeeze()  # Remove the extra dimensions
    if tensor.ndim == 2:  # If the image is grayscale, add a dummy channel dimension
        tensor = tensor.unsqueeze(0)
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)  # Reorder the dimensions
    image = image * np.array((1, 1, 1))
    image = image.clip(0, 1)
    return image

def ConnectedComp(img, dim):
    kernel = np.ones((3, 3), dtype=np.float32)
    image = cv2.resize(img.astype(np.float32), dim)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayy = (gray * 255).astype(np.uint8)
    thresh = cv2.threshold(grayy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    erosion = cv2.erode(thresh, kernel, iterations=3)
    gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)

    ret, markers = cv2.connectedComponents(erosion, connectivity=8)
    new = cv2.watershed(erosion, markers, mask=thresh)
    RGB = label2rgb(new, bg_label=0)

    return erosion, gradient, RGB

def get_fnames(root):
    xs = os.listdir(os.path.join(root))
    f = lambda fname: int(fname.split('.png')[0])
    xs = sorted(xs, key=f)
    return xs

class dset(Dataset):
    def __init__(self, data, root_dir='data', transformX=None):
        self.root_dir = root_dir
        self.transformX = transformX
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fname = self.data[index]
        imx_name = os.path.join(self.root_dir, fname)
        imx = Image.open(imx_name)

        if self.transformX:
            imx = self.transformX(imx)

        sample = {'image': imx}
        return sample

# Define transformations
tx_X = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
model = load_model()

# Streamlit app
st.title('ResUNet X-ray Image Segmentation')

uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)

    # Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Save the uploaded image temporarily
    uploaded_image_path = os.path.join("data", "uploaded_image.jpg")
    image.save(uploaded_image_path)

    # Create a custom dataset and dataloader for the uploaded image
    test_data = ["uploaded_image.jpg"]
    test_set = dset(test_data, 'data', transformX=tx_X)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

    # Display the original image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        with torch.no_grad():
            pig = plt.figure(figsize=(16, 10))
            dim = 3040, 1280

            for i, sample in enumerate(test_loader):
                Xs = sample["image"]
                a = pig.add_subplot(3, 1, 1)
                imgx = im_converterX(Xs[0])
                imgx = cv2.resize(imgx, dim, interpolation=cv2.INTER_AREA)
                plt.title('Input x-ray img')
                plt.imshow(imgx)

                a = pig.add_subplot(3, 1, 2)
                output_img = im_converterY(model(Xs)[0])
                output_img = cv2.resize(output_img, dim, interpolation=cv2.INTER_AREA)
                plt.title('Predicted masked img')
                plt.imshow(output_img)
                break

            # Display the figure in Streamlit
            st.pyplot(pig)
