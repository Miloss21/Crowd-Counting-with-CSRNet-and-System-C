#!/usr/bin/env python3
"""Verifikacija SystemC vs PyTorch"""

import torch
import numpy as np
import sys
from pathlib import Path
from model import CSRNet

# Konfiguracija
BASE_DIR = Path(__file__).parent.parent.parent  # root projekta (iznad python/)
SYSTEMC_DIR = BASE_DIR / "data"

INPUT_BIN = SYSTEMC_DIR / "input_image.bin"
OUTPUT_BIN = SYSTEMC_DIR / "output.bin"
WEIGHTS_PTH = BASE_DIR / "data" / "weights.pth"

INPUT_CHANNELS = 3
INPUT_HEIGHT = 256
INPUT_WIDTH = 256
OUTPUT_CHANNELS = 64
FRAC_BITS_ACT = 10


def main():
    print("\n" + "="*60)
    print("  VERIFIKACIJA: SystemC vs PyTorch")
    print("="*60)

    # 1. Učitaj input (isti ulaz kao SystemC - kvantizovan)
    print("\n[1/4] Učitavanje input slike...")
    if not INPUT_BIN.exists():
        print(f"❌ {INPUT_BIN} ne postoji!")
        return 1

    img_fixed = np.fromfile(INPUT_BIN, dtype=np.int16)
    img_fixed = img_fixed.reshape(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    img_float = img_fixed.astype(np.float32) / (1 << FRAC_BITS_ACT)
    print(f"   ✅ Slika: {img_fixed.shape}")

    # 2. PyTorch inference (samo prvi Conv2d sloj + ReLU)
    # Napomena: model.frontend[0] je Conv2d, model.frontend[1] je ReLU.
    # SystemC primenjuje ReLU odmah nakon konvolucije (vidi conv2d_fixed.hpp),
    # pa ovde takođe moramo dodati relu() da bi poređenje bilo pravično.
    print("\n[2/4] PyTorch inference...")
    model = CSRNet()
    checkpoint = torch.load(WEIGHTS_PTH, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    img_tensor = torch.from_numpy(img_float).unsqueeze(0)

    with torch.no_grad():
        first_conv = model.frontend[0]
        output_pytorch = first_conv(img_tensor)
        output_pytorch = torch.relu(output_pytorch)

    output_pytorch = output_pytorch.squeeze(0).numpy()
    print(f"   ✅ PyTorch output: {output_pytorch.shape}")
    print(f"   📊 Range: [{output_pytorch.min():.3f}, {output_pytorch.max():.3f}]")

    # 3. Učitaj SystemC output
    print("\n[3/4] Učitavanje SystemC output...")
    if not OUTPUT_BIN.exists():
        print(f"❌ {OUTPUT_BIN} ne postoji!")
        return 1

    output_fixed = np.fromfile(OUTPUT_BIN, dtype=np.int16)
    expected_size = OUTPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH

    if output_fixed.size != expected_size:
        print(f"❌ Veličina: očekivano {expected_size}, dobijeno {output_fixed.size}")
        return 1

    output_fixed = output_fixed.reshape(OUTPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    output_systemc = output_fixed.astype(np.float32) / (1 << FRAC_BITS_ACT)
    print(f"   ✅ SystemC output: {output_systemc.shape}")
    print(f"   📊 Range: [{output_systemc.min():.3f}, {output_systemc.max():.3f}]")

    # 4. Poređenje
    print("\n[4/4] Poređenje...")

    abs_diff = np.abs(output_pytorch - output_systemc)
    mean_error = np.mean(abs_diff)
    max_error = np.max(abs_diff)

    # Globalna relativna greska (L1 norma razlike / L1 norma reference).
    # Ovo daje realan procenat za razliku od po-pikselne podele koja puca
    # na ReLU nulama (output ima puno nula, deljenje sa ~0 = veliki broj).
    epsilon = 1e-8
    total_abs_ref = np.sum(np.abs(output_pytorch))
    rel_error = np.sum(abs_diff) / (total_abs_ref + epsilon)

    # Preciznost kao 1 - prosecna_greska / prosecna_referenca
    mean_abs_ref = np.mean(np.abs(output_pytorch))
    similarity = 1.0 - (mean_error / (mean_abs_ref + epsilon))
    similarity_percent = similarity * 100

    print("\n" + "="*60)
    print("  REZULTATI")
    print("="*60)
    print(f"  Prosečna greška:      {mean_error:.6f}")
    print(f"  Maksimalna greška:    {max_error:.6f}")
    print(f"  Relativna greška:     {rel_error:.2%}")
    print(f"  Preciznost:           {similarity_percent:.2f}%")
    print("="*60)

    if similarity_percent > 99.5:
        print("\n  🎉 ODLIČNO! SystemC je veoma precizan!")
        return 0
    elif similarity_percent > 98.0:
        print("\n  ✅ DOBRO! Male razlike prihvatljive za fixed-point.")
        return 0
    elif similarity_percent > 95.0:
        print("\n  ⚠️  PRIHVATLJIVO, ali proveri implementaciju.")
        return 0
    else:
        print("\n  ❌ GREŠKA! Velike razlike!")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ GREŠKA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)