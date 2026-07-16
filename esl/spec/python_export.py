"""
Eksport slike, težina i bias-a u fixed-point format za SystemC.

Konstante na vrhu MORAJU biti sinhronizovane sa esl/defines.hpp.
Ako se menja bit-width u C++ kodu, promeniti i ovde.
"""

import torch
import numpy as np
import cv2
import json
from pathlib import Path
from model import CSRNet

# ============================================================
# KONFIGURACIJA - MORA BITI SINHRONIZOVANA SA esl/defines.hpp
# ============================================================

# Fixed-point formati (bitske širine uključuju sign bit)
ACT_INT_BITS    = 3 # bilo 2
ACT_FRAC_BITS   = 10
ACT_TOTAL_BITS  = ACT_INT_BITS + ACT_FRAC_BITS          # 12

WEIGHT_INT_BITS   = 2
WEIGHT_FRAC_BITS  = 8
WEIGHT_TOTAL_BITS = WEIGHT_INT_BITS + WEIGHT_FRAC_BITS  # 10

BIAS_INT_BITS    = 2
BIAS_FRAC_BITS   = 10
BIAS_TOTAL_BITS  = BIAS_INT_BITS + BIAS_FRAC_BITS       # 12

# Arhitektura parametri
INPUT_HEIGHT    = 256
INPUT_WIDTH     = 256
INPUT_CHANNELS  = 3
OUTPUT_CHANNELS = 64

# Putanje (robustno - radi iz bilo kog foldera)
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
IMG_PATH = DATA_DIR / "testA" / "images" / "IMG_100.jpg"
WEIGHTS_PATH = DATA_DIR / "weights.pth"


# ============================================================
# HELPER: saturacija na signed opseg
# ============================================================

def signed_range(total_bits):
    """Vraca (min, max) za signed integer sa datom bitskom sirinom."""
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))
    return min_val, max_val


# ============================================================
# UCITAVANJE MODELA I SLIKE
# ============================================================

print("Učitavanje modela...")
model = CSRNet()
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

print("Učitavanje slike...")
img = cv2.imread(str(IMG_PATH))
if img is None:
    raise FileNotFoundError(f"Slika nije pronađena: {IMG_PATH}")

print(f"Originalna veličina: {img.shape}")

# Resize na ulazne dimenzije
img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
print(f"Nakon resize-a: {img.shape}")

# Normalizacija (ImageNet statistika - mora biti isto kao PyTorch transforms)
img = img.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
for c in range(3):
    img[:, :, c] = (img[:, :, c] - mean[c]) / std[c]

print(f"Opseg nakon normalizacije: [{img.min():.3f}, {img.max():.3f}]")


# ============================================================
# KONVERZIJA SLIKE U FIXED-POINT
# ============================================================

print("Konverzija slike u fixed-point...")
img_fixed = img * (1 << ACT_FRAC_BITS)

MIN_ACT, MAX_ACT = signed_range(ACT_TOTAL_BITS)
n_saturated = np.sum((img_fixed > MAX_ACT) | (img_fixed < MIN_ACT))
if n_saturated > 0:
    print(f"⚠️  {n_saturated} piksela saturirano na opseg [{MIN_ACT}, {MAX_ACT}]")

img_fixed = np.clip(img_fixed, MIN_ACT, MAX_ACT).astype(np.int16)
print(f"Opseg fixed-point: [{img_fixed.min()}, {img_fixed.max()}] (dozvoljeno: [{MIN_ACT}, {MAX_ACT}])")

# Transpose iz (H,W,C) u (C,H,W) - PyTorch/SystemC konvencija
img_fixed_chw = np.transpose(img_fixed, (2, 0, 1))
print(f"Shape: {img_fixed_chw.shape}")

img_fixed_chw.tofile(DATA_DIR / "input_image.bin")
print(f"✅ Slika: {img_fixed_chw.nbytes} bytes")


# ============================================================
# EKSPORT TEZINA
# ============================================================

print("\nEksport težina...")
first_conv = model.frontend[0]
weights = first_conv.weight.detach().cpu().numpy()
print(f"Weights shape: {weights.shape}")

MIN_W, MAX_W = signed_range(WEIGHT_TOTAL_BITS)
weights_fixed = weights * (1 << WEIGHT_FRAC_BITS)
weights_fixed = np.clip(weights_fixed, MIN_W, MAX_W).astype(np.int16)

print(f"Weights range: [{weights_fixed.min()}, {weights_fixed.max()}] (dozvoljeno: [{MIN_W}, {MAX_W}])")

weights_fixed.tofile(DATA_DIR / "weights_conv1.bin")
print(f"✅ Težine: {weights_fixed.nbytes} bytes")


# ============================================================
# EKSPORT BIAS-A
# ============================================================

print("\nEksport bias-a...")
MIN_B, MAX_B = signed_range(BIAS_TOTAL_BITS)

if first_conv.bias is not None:
    bias = first_conv.bias.detach().cpu().numpy()
    print(f"Bias shape: {bias.shape}")
    print(f"Bias range (float): [{bias.min():.6f}, {bias.max():.6f}]")

    bias_fixed = bias * (1 << BIAS_FRAC_BITS)
    bias_fixed = np.clip(bias_fixed, MIN_B, MAX_B).astype(np.int16)

    print(f"Bias range (fixed): [{bias_fixed.min()}, {bias_fixed.max()}] (dozvoljeno: [{MIN_B}, {MAX_B}])")
else:
    print("⚠️  WARNING: Prvi Conv2d sloj NEMA bias! Kreiram nule.")
    bias_fixed = np.zeros(OUTPUT_CHANNELS, dtype=np.int16)

bias_fixed.tofile(DATA_DIR / "bias_conv1.bin")
print(f"✅ Bias: {bias_fixed.nbytes} bytes ({len(bias_fixed)} elemenata)")


# ============================================================
# METADATA
# ============================================================

metadata = {
    "input_shape":    list(img_fixed_chw.shape),
    "weights_shape":  list(weights_fixed.shape),
    "bias_shape":     [OUTPUT_CHANNELS],
    "input_bytes":    int(img_fixed_chw.nbytes),
    "weights_bytes":  int(weights_fixed.nbytes),
    "bias_bytes":     int(bias_fixed.nbytes),
    "bit_formats": {
        "activations": {"int": ACT_INT_BITS,    "frac": ACT_FRAC_BITS,    "total": ACT_TOTAL_BITS},
        "weights":     {"int": WEIGHT_INT_BITS, "frac": WEIGHT_FRAC_BITS, "total": WEIGHT_TOTAL_BITS},
        "bias":        {"int": BIAS_INT_BITS,   "frac": BIAS_FRAC_BITS,   "total": BIAS_TOTAL_BITS},
    },
}

with open(DATA_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)


print("\n" + "="*50)
print("EXPORT ZAVRŠEN!")
print("="*50)
print(f"\nKreirani fajlovi u {DATA_DIR}:")
print(f"  - input_image.bin   ({img_fixed_chw.nbytes} bytes)")
print(f"  - weights_conv1.bin ({weights_fixed.nbytes} bytes)")
print(f"  - bias_conv1.bin    ({bias_fixed.nbytes} bytes)")
print(f"  - metadata.json")
print("="*50)