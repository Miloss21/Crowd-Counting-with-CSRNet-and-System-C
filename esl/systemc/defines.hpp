#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <cstdint>

// ==========================================
// FIXED-POINT FORMATI
// ==========================================

// AKTIVACIJE (ulaz/izlaz slojeva)
constexpr int ACT_INT_BITS = 3; // ovde bilo 2
constexpr int ACT_FRAC_BITS = 10;
constexpr int ACT_TOTAL_BITS = ACT_INT_BITS + ACT_FRAC_BITS;
typedef int16_t fixed_act_t;

// TEŽINE
constexpr int WEIGHT_INT_BITS = 2;
constexpr int WEIGHT_FRAC_BITS = 8;
constexpr int WEIGHT_TOTAL_BITS = WEIGHT_INT_BITS + WEIGHT_FRAC_BITS;
typedef int16_t fixed_weight_t;

// BIAS
constexpr int BIAS_INT_BITS = 2;
constexpr int BIAS_FRAC_BITS = 10;
constexpr int BIAS_TOTAL_BITS = BIAS_INT_BITS + BIAS_FRAC_BITS;
typedef int16_t fixed_bias_t;

// OUTPUT
constexpr int OUT_INT_BITS = 6;
constexpr int OUT_FRAC_BITS = 10;
constexpr int OUT_TOTAL_BITS = OUT_INT_BITS + OUT_FRAC_BITS;
typedef int16_t fixed_output_t;

// AKUMULATOR
constexpr int ACCUM_BITS = 32;
typedef int32_t fixed_accum_t;

// ==========================================
// ARHITEKTURA PARAMETRI
// ==========================================

constexpr int INPUT_CHANNELS = 3;
constexpr int INPUT_HEIGHT = 256;
constexpr int INPUT_WIDTH = 256;
constexpr int OUTPUT_CHANNELS = 64;
constexpr int KERNEL_SIZE = 3;
constexpr int STRIDE = 1;
constexpr int PADDING = 1;

// MEMORY LAYOUT
constexpr uint32_t INPUT_SIZE = 
    INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(int16_t);

constexpr uint32_t WEIGHTS_SIZE = 
    OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(int16_t);

constexpr uint32_t BIAS_SIZE = 
    OUTPUT_CHANNELS * sizeof(int16_t);

constexpr uint32_t OUTPUT_SIZE = 
    OUTPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(int16_t);

// TOTAL MEMORY SIZE
constexpr uint32_t MEMORY_SIZE = 0x01000000;  // 16 MB

// MEMORY MAP
constexpr uint32_t VP_ADDR_MEMORY_L = 0x00000000;
constexpr uint32_t VP_ADDR_MEMORY_H = 0x00FFFFFF;
constexpr uint32_t VP_ADDR_IP_L     = 0x43c00000;
constexpr uint32_t VP_ADDR_IP_H     = 0x43c000FF;

constexpr uint32_t INPUT_ADDR     = 0x00000000;
constexpr uint32_t WEIGHTS_ADDR   = 0x00200000;
constexpr uint32_t BIAS_ADDR      = 0x00300000;
constexpr uint32_t OUTPUT_ADDR    = 0x00400000;

// IP REGISTERS
constexpr uint8_t REG_CTRL          = 0x00;
constexpr uint8_t REG_STATUS        = 0x04;
constexpr uint8_t REG_IN_ADDR       = 0x08;
constexpr uint8_t REG_WEIGHTS_ADDR  = 0x0C;
constexpr uint8_t REG_BIAS_ADDR     = 0x10;
constexpr uint8_t REG_OUT_ADDR      = 0x14;
constexpr uint8_t REG_IN_H          = 0x18;
constexpr uint8_t REG_IN_W          = 0x1C;
constexpr uint8_t REG_IN_C          = 0x20;
constexpr uint8_t REG_OUT_C         = 0x24;
constexpr uint8_t REG_KERNEL_SIZE   = 0x28;
constexpr uint8_t REG_STRIDE        = 0x2C;
constexpr uint8_t REG_PADDING       = 0x30;

// TIMING
constexpr int DELAY = 10;  // ns
constexpr double CLOCK_PERIOD_NS = 10.0;  // 100 MHz

#endif // DEFINES_HPP