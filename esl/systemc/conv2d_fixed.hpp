#ifndef CONV2D_FIXED_HPP
#define CONV2D_FIXED_HPP

#include "defines.hpp"
#include <cstdint>
#include <iostream>
#include <cmath>

// ==========================================
// KONVERZIJA: float → fixed-point
// ==========================================

inline fixed_act_t float_to_fixed_act(float val) {
    int32_t scaled = static_cast<int32_t>(val * (1 << ACT_FRAC_BITS));
    
    const int32_t MAX_VAL = (1 << (ACT_TOTAL_BITS - 1)) - 1;  
    const int32_t MIN_VAL = -(1 << (ACT_TOTAL_BITS - 1));     
    
    if (scaled > MAX_VAL) {
        std::cerr << "WARNING: Activation overflow! " << val << " → clipped" << std::endl;
        scaled = MAX_VAL;
    }
    if (scaled < MIN_VAL) {
        std::cerr << "WARNING: Activation underflow! " << val << " → clipped" << std::endl;
        scaled = MIN_VAL;
    }
    
    return static_cast<fixed_act_t>(scaled);
}

inline fixed_weight_t float_to_fixed_weight(float val) {
    int32_t scaled = static_cast<int32_t>(val * (1 << WEIGHT_FRAC_BITS));
    
    const int32_t MAX_VAL = (1 << (WEIGHT_TOTAL_BITS - 1)) - 1;  // +511
    const int32_t MIN_VAL = -(1 << (WEIGHT_TOTAL_BITS - 1));      // -512
    
    if (scaled > MAX_VAL) scaled = MAX_VAL;
    if (scaled < MIN_VAL) scaled = MIN_VAL;
    
    return static_cast<fixed_weight_t>(scaled);
}

inline float fixed_act_to_float(fixed_act_t val) {
    return static_cast<float>(val) / (1 << ACT_FRAC_BITS);
}

inline float fixed_output_to_float(fixed_output_t val) {
    return static_cast<float>(val) / (1 << OUT_FRAC_BITS);
}

// ==========================================
// KRITIČNA FUNKCIJA: Konvolucija
// ==========================================

void conv2d_fixed_point(
    fixed_act_t*** input,          // [C][H][W]
    fixed_weight_t**** weights,    // [OutC][InC][KH][KW]
    fixed_bias_t* bias,            // [OutC]  DODATO
    fixed_output_t*** output,      // [OutC][OutH][OutW]
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int stride,
    int padding
) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    std::cout << "[CONV2D] Početak konvolucije..." << std::endl;
    std::cout << "[CONV2D] Input:  " << in_channels << "x" << in_h << "x" << in_w << std::endl;
    std::cout << "[CONV2D] Output: " << out_channels << "x" << out_h << "x" << out_w << std::endl;
    std::cout << "[CONV2D] Bitske širine:" << std::endl;
    std::cout << "[CONV2D]   - Aktivacije: " << ACT_TOTAL_BITS << " bita (" 
              << ACT_INT_BITS << " int + " << ACT_FRAC_BITS << " frac)" << std::endl;
    std::cout << "[CONV2D]   - Težine: " << WEIGHT_TOTAL_BITS << " bita (" 
              << WEIGHT_INT_BITS << " int + " << WEIGHT_FRAC_BITS << " frac)" << std::endl;
    std::cout << "[CONV2D]   - Akumulator: " << ACCUM_BITS << " bita" << std::endl;
    
    int overflow_count = 0;
    
    // Prolazak kroz sve output kanale
    for (int oc = 0; oc < out_channels; oc++) {
        
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                
                // ==========================================
                // AKUMULATOR (32-bit) - INICIJALIZUJ SA BIAS-om
                // ==========================================
                // Bias je već u pravom formatu (sa istim frac bitovima kao aktivacije)
                // pa ga samo konvertujemo u 32-bit i šiftujemo za weight frac bitove
                fixed_accum_t accum = static_cast<fixed_accum_t>(bias[oc]) << WEIGHT_FRAC_BITS;
                
                int in_y_start = oy * stride - padding;
                int in_x_start = ox * stride - padding;
                
                // Konvolucija preko svih input kanala
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            
                            int in_y = in_y_start + kh;
                            int in_x = in_x_start + kw;
                            
                            // Zero padding
                            fixed_act_t pixel;
                            if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w) {
                                pixel = 0;
                            } else {
                                pixel = input[ic][in_y][in_x];
                            }
                            
                            fixed_weight_t weight = weights[oc][ic][kh][kw];
                            
                            // ==========================================
                            // MAC OPERACIJA
                            // ==========================================
                            accum += static_cast<fixed_accum_t>(pixel) * 
                                     static_cast<fixed_accum_t>(weight);
                        }
                    }
                }
                
                // ==========================================
                // SKALIRANJE U OUTPUT FORMAT
                // ==========================================
                fixed_accum_t scaled = accum >> WEIGHT_FRAC_BITS;
                
                // Saturacija na 16 bitova output
                const fixed_accum_t MAX_OUT = (1 << (OUT_TOTAL_BITS - 1)) - 1;
                const fixed_accum_t MIN_OUT = -(1 << (OUT_TOTAL_BITS - 1));
                
                if (scaled > MAX_OUT) {
                    scaled = MAX_OUT;
                    overflow_count++;
                } else if (scaled < MIN_OUT) {
                    scaled = MIN_OUT;
                    overflow_count++;
                }
                
                // ReLU aktivacija
                if (scaled < 0) scaled = 0;
                
                output[oc][oy][ox] = static_cast<fixed_output_t>(scaled);
            }
        }
        
        // Progress indikator
        if ((oc + 1) % 8 == 0) {
            std::cout << "[CONV2D] Obrađeno " << (oc + 1) << "/" << out_channels << " kanala" << std::endl;
        }
    }
    
    if (overflow_count > 0) {
        std::cout << "[CONV2D] WARNING: " << overflow_count << " overflow-a!" << std::endl;
    }
    
    std::cout << "[CONV2D] Konvolucija završena!" << std::endl;
}

// ==========================================
// HELPER FUNKCIJE: Alokacija memorije
// ==========================================

inline fixed_bias_t* alloc_1d_bias(int size) {
    return new fixed_bias_t[size]();
}

inline void free_1d_bias(fixed_bias_t* arr) {
    delete[] arr;
}

inline fixed_act_t*** alloc_3d_act(int c, int h, int w) {
    fixed_act_t*** arr = new fixed_act_t**[c];
    for (int i = 0; i < c; i++) {
        arr[i] = new fixed_act_t*[h];
        for (int j = 0; j < h; j++) {
            arr[i][j] = new fixed_act_t[w]();
        }
    }
    return arr;
}

inline fixed_weight_t**** alloc_4d_weight(int oc, int ic, int kh, int kw) {
    fixed_weight_t**** arr = new fixed_weight_t***[oc];
    for (int i = 0; i < oc; i++) {
        arr[i] = new fixed_weight_t**[ic];
        for (int j = 0; j < ic; j++) {
            arr[i][j] = new fixed_weight_t*[kh];
            for (int k = 0; k < kh; k++) {
                arr[i][j][k] = new fixed_weight_t[kw]();
            }
        }
    }
    return arr;
}

inline fixed_output_t*** alloc_3d_output(int c, int h, int w) {
    fixed_output_t*** arr = new fixed_output_t**[c];
    for (int i = 0; i < c; i++) {
        arr[i] = new fixed_output_t*[h];
        for (int j = 0; j < h; j++) {
            arr[i][j] = new fixed_output_t[w]();
        }
    }
    return arr;
}

inline void free_3d_act(fixed_act_t*** arr, int c, int h) {
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            delete[] arr[i][j];
        }
        delete[] arr[i];
    }
    delete[] arr;
}

inline void free_4d_weight(fixed_weight_t**** arr, int oc, int ic, int kh) {
    for (int i = 0; i < oc; i++) {
        for (int j = 0; j < ic; j++) {
            for (int k = 0; k < kh; k++) {
                delete[] arr[i][j][k];
            }
            delete[] arr[i][j];
        }
        delete[] arr[i];
    }
    delete[] arr;
}

inline void free_3d_output(fixed_output_t*** arr, int c, int h) {
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            delete[] arr[i][j];
        }
        delete[] arr[i];
    }
    delete[] arr;
}

#endif // CONV2D_FIXED_HPP