# ---------- model.py (modified) ----------
import torch
import torch.nn as nn
from torchvision import models
import math
import csv

# ============================
#  BITSKA ANALIZA – GLOBALNE STRUKTURE
# ============================

BIT_RANGES = {
    "layers": []
}


def register_bitwise_hooks(model):
    """
    Kači hook na svaki Conv2d da bi se zabeležili:
    - min/max ulaza
    - min/max težina
    - min/max izlaza
    """
    def hook_fn(layer, input, output):
        inp = input[0]
        out = output

        BIT_RANGES["layers"].append({
            "layer": layer.__class__.__name__,
            "in_min": float(inp.min().item()),
            "in_max": float(inp.max().item()),
            "w_min": float(layer.weight.min().item()),
            "w_max": float(layer.weight.max().item()),
            "out_min": float(out.min().item()),
            "out_max": float(out.max().item()),
            "weight_shape": tuple(layer.weight.shape),
            "output_shape": tuple(out.shape),
        })

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(hook_fn)


# ============================
#  TVOJ ORIGINALNI CSRNET MODEL
# ============================

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # ⭐ DODATO: Registracija hook-ova za bitsku analizu
        register_bitwise_hooks(self)

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# ============================
#  FUNKCIJE ZA IZRACUN BIT-WIDTH-OVA I CSV
# ============================

def _mag(xmin, xmax):
    return max(abs(xmin), abs(xmax))


def _bits_no_sign_for_mag(mag):
    # returns number of integer bits *excluding* sign bit needed to represent magnitude
    if mag <= 0:
        return 0
    # if mag < 1, no integer bits are required (all fractional)
    if mag < 1:
        return 0
    return math.ceil(math.log2(mag + 1e-12))


def compute_bitwidths(bit_ranges, frac_bits_act_default=10, frac_bits_w_default=8, safety_margin_acc=2, csv_path='bitwidths_report.csv'):
    """
    Izracunava minimalne int-bitove za input/weight/output po layeru, i predlaze
    fractional bits, akumulacione bitove i ukupne formate. Snima CSV.

    Povratna vrednost: lista dictova sa predlozima po layeru
    """
    rows = []

    for i, layer in enumerate(bit_ranges['layers']):
        w_shape = layer.get('weight_shape', (0,0,1,1))
        out_ch, in_ch, kh, kw = w_shape
        kernel_elements = max(1, kh * kw * in_ch)

        in_min = layer['in_min']
        in_max = layer['in_max']
        w_min = layer['w_min']
        w_max = layer['w_max']
        out_min = layer['out_min']
        out_max = layer['out_max']

        in_mag = _mag(in_min, in_max)
        w_mag = _mag(w_min, w_max)
        out_mag = _mag(out_min, out_max)

        in_int_no_sign = _bits_no_sign_for_mag(in_mag)
        w_int_no_sign = _bits_no_sign_for_mag(w_mag)
        out_int_no_sign = _bits_no_sign_for_mag(out_mag)

        # add sign bit
        in_int_with_sign = in_int_no_sign + 1
        w_int_with_sign = w_int_no_sign + 1
        out_int_with_sign = out_int_no_sign + 1

        # suggested fractional bits (defaults can be tuned)
        frac_act = frac_bits_act_default
        frac_w = frac_bits_w_default

        # total bits for activation and weight
        act_total_bits = in_int_with_sign + frac_act
        w_total_bits = w_int_with_sign + frac_w

        # accum bits: product bits + log2(kernel elements) + safety margin
        prod_bits = act_total_bits + w_total_bits
        accum_extra = math.ceil(math.log2(kernel_elements))
        accum_bits = prod_bits + accum_extra + safety_margin_acc

        row = {
            'layer_index': i+1,
            'layer_name': layer['layer'],
            'weight_shape': w_shape,
            'output_shape': layer.get('output_shape', None),

            'in_min': in_min,
            'in_max': in_max,
            'in_mag': in_mag,
            'in_int_no_sign': in_int_no_sign,
            'in_int_with_sign': in_int_with_sign,

            'w_min': w_min,
            'w_max': w_max,
            'w_mag': w_mag,
            'w_int_no_sign': w_int_no_sign,
            'w_int_with_sign': w_int_with_sign,

            'out_min': out_min,
            'out_max': out_max,
            'out_mag': out_mag,
            'out_int_no_sign': out_int_no_sign,
            'out_int_with_sign': out_int_with_sign,

            'suggested_frac_act': frac_act,
            'suggested_frac_w': frac_w,
            'suggested_act_total_bits': act_total_bits,
            'suggested_w_total_bits': w_total_bits,
            'kernel_elements': kernel_elements,
            'accum_extra_bits': accum_extra,
            'suggested_accum_bits': accum_bits,
        }

        rows.append(row)

    # write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return rows


# Eksportuj korisne funkcije/varijable
__all__ = ['CSRNet', 'make_layers', 'BIT_RANGES', 'compute_bitwidths']



