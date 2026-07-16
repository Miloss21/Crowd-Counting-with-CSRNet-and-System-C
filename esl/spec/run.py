# ---------- run.py (modified) ----------
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt, cm as c
from scipy.ndimage import gaussian_filter
import scipy
import torchvision.transforms.functional as F
from model import CSRNet, BIT_RANGES, compute_bitwidths
import torch
from torchvision import transforms
import cProfile
import pstats
from torch.utils.tensorboard import SummaryWriter
import time
import os


def main():

    img_path = os.path.join("..", "..", "data", "testA", "images", "IMG_100.jpg")
    weights_path = os.path.join("..", "..", "data", "weights.pth")

    # Transformacije
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Učitavanje modela
    model = CSRNet()
    checkpoint = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(checkpoint)
    model.eval()

    # Prikaz slike
    print("Original Image")
    plt.imshow(plt.imread(img_path))
    plt.savefig("output_image.png")
    plt.show()

    # Obrada slike
    img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)

    # Inference
    output = model(img)

    # Prikaz predikcije
    print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))
    temp = np.asarray(output.detach().cpu().reshape(output.shape[2], output.shape[3]))
    plt.imshow(temp, cmap=c.jet)
    plt.savefig("density_image.png")
    plt.show()

    # ================================
    #  ISPIS BIT RANGE ANALIZE
    # ================================
    print("\n===== BIT RANGE ANALYSIS =====")
    for i, layer in enumerate(BIT_RANGES["layers"]):
        print(f"\nLayer {i+1}: {layer['layer']}")
        print(f"  Input range:  {layer['in_min']:.6f} → {layer['in_max']:.6f}")
        print(f"  Weight range: {layer['w_min']:.6f} → {layer['w_max']:.6f}")
        print(f"  Output range: {layer['out_min']:.6f} → {layer['out_max']:.6f}")
        print(f"  Weight shape: {layer['weight_shape']}")
        print(f"  Output shape: {layer['output_shape']}")

    # ================================
    #  GENERISI CSV SA PREDLOZIMA BIT-WIDTH
    # ================================
    csv_path = 'bitwidths_report.csv'
    rows = compute_bitwidths(BIT_RANGES, frac_bits_act_default=10, frac_bits_w_default=8, safety_margin_acc=2, csv_path=csv_path)
    print(f"\nSaved bitwidth suggestions to {csv_path} ({len(rows)} layers).")


if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    # Snimi rezultate
    profiler.dump_stats('profile_output.prof')

    # Prikaži top 10 najsporijih funkcija
    stats = pstats.Stats('profile_output.prof')
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.print_stats(10)