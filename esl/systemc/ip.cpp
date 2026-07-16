#include "ip.hpp"
#include "conv2d_fixed.hpp"
#include <iostream>

using namespace sc_core;
using namespace std;


IP::IP(sc_module_name name)
    : sc_module(name),
      ip_socket("ip_socket"),
      mem_socket("mem_socket"),
      offset(SC_ZERO_TIME),
      busy(false)
{
    ip_socket.register_b_transport(this, &IP::b_transport);
    SC_THREAD(ip_main_thread);

    reg_ctrl = 0;
    reg_status = 0x01;

    reg_in_addr = 0;
    reg_weights_addr = 0;
    reg_bias_addr = 0;          // ✅ DODATO
    reg_out_addr = 0;
    
    reg_in_h = 0;
    reg_in_w = 0;
    reg_in_c = 0;
    reg_out_c = 0;
    reg_kernel_size = 0;
    reg_stride = 0;
    reg_padding = 0;

    SC_REPORT_INFO("IP", "Constructed.");
}

IP::~IP() {
    SC_REPORT_INFO("IP", "Destroyed.");
}

// ======================================================
// REGISTER ACCESS
// ======================================================
void IP::b_transport(pl_t &pl, sc_time &delay)
{
    uint32_t addr = pl.get_address() - VP_ADDR_IP_L;
    uint8_t* data = pl.get_data_ptr();

    if (pl.get_command() == tlm::TLM_WRITE_COMMAND) {
        uint32_t val =
            (data[3] << 24) |
            (data[2] << 16) |
            (data[1] << 8)  |
             data[0];

        switch (addr) {
            case REG_CTRL:
                reg_ctrl = val;
                if ((val & 0x1) && !busy) {
                    busy = true;
                    reg_status = 0x02;
                    start_event.notify(SC_ZERO_TIME);
                }
                break;

            case REG_IN_ADDR:      reg_in_addr = val; break;
            case REG_WEIGHTS_ADDR: reg_weights_addr = val; break;
            case REG_BIAS_ADDR:    reg_bias_addr = val; break;   // ✅
            case REG_OUT_ADDR:     reg_out_addr = val; break;

            case REG_IN_H:         reg_in_h = val; break;
            case REG_IN_W:         reg_in_w = val; break;
            case REG_IN_C:         reg_in_c = val; break;
            case REG_OUT_C:        reg_out_c = val; break;
            case REG_KERNEL_SIZE:  reg_kernel_size = val; break;
            case REG_STRIDE:       reg_stride = val; break;
            case REG_PADDING:      reg_padding = val; break;

            default:
                pl.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
                return;
        }
    } else {
        uint32_t val = 0;
        if (addr == REG_STATUS) val = reg_status;
        else if (addr == REG_CTRL) val = reg_ctrl;
        else {
            pl.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            return;
        }

        data[0] = val & 0xFF;
        data[1] = (val >> 8) & 0xFF;
        data[2] = (val >> 16) & 0xFF;
        data[3] = (val >> 24) & 0xFF;
    }

    pl.set_response_status(tlm::TLM_OK_RESPONSE);
    delay += sc_time(DELAY, SC_NS);
}

// ======================================================
// MAIN THREAD
// ======================================================
void IP::ip_main_thread()
{
    while (true) {
        wait(start_event);
        perform_convolution();
        busy = false;
        reg_status = 0x01;
        done_event.notify(SC_ZERO_TIME);
    }
}

// ======================================================
// CONVOLUTION
// ======================================================
void IP::perform_convolution()
{
    fixed_act_t*** input =
        alloc_3d_act(reg_in_c, reg_in_h, reg_in_w);

    fixed_weight_t**** weights =
        alloc_4d_weight(reg_out_c, reg_in_c,
                        reg_kernel_size, reg_kernel_size);

    fixed_bias_t* bias =
        alloc_1d_bias(reg_out_c);    // ✅

    uint32_t out_h =
        (reg_in_h + 2 * reg_padding - reg_kernel_size) / reg_stride + 1;
    uint32_t out_w =
        (reg_in_w + 2 * reg_padding - reg_kernel_size) / reg_stride + 1;

    fixed_output_t*** output =
        alloc_3d_output(reg_out_c, out_h, out_w);

    // INPUT
    for (uint32_t c = 0; c < reg_in_c; c++)
        for (uint32_t i = 0; i < reg_in_h; i++)
            for (uint32_t j = 0; j < reg_in_w; j++)
                input[c][i][j] =
                    read_mem16(reg_in_addr +
                        (c * reg_in_h * reg_in_w +
                         i * reg_in_w + j) * 2);

    // WEIGHTS
    for (uint32_t oc = 0; oc < reg_out_c; oc++)
        for (uint32_t ic = 0; ic < reg_in_c; ic++)
            for (uint32_t kh = 0; kh < reg_kernel_size; kh++)
                for (uint32_t kw = 0; kw < reg_kernel_size; kw++)
                    weights[oc][ic][kh][kw] =
                        read_mem16(reg_weights_addr +
                            ((oc * reg_in_c * reg_kernel_size * reg_kernel_size) +
                             (ic * reg_kernel_size * reg_kernel_size) +
                             (kh * reg_kernel_size) + kw) * 2);

    // ✅ BIAS
    for (uint32_t oc = 0; oc < reg_out_c; oc++)
        bias[oc] =
            read_mem16(reg_bias_addr + oc * 2);

    // CONV
    conv2d_fixed_point(
        input, weights, bias, output,
        reg_in_c, reg_in_h, reg_in_w,
        reg_out_c, reg_kernel_size,
        reg_stride, reg_padding
    );
        //kasnjenje 
            uint32_t num_output_pixels = reg_out_c * out_h * out_w;
            const uint32_t PIPELINE_DEPTH = 5;
            sc_time compute_time = sc_time((num_output_pixels + PIPELINE_DEPTH) * CLOCK_PERIOD_NS, SC_NS);
            wait(compute_time);



    // OUTPUT
    for (uint32_t oc = 0; oc < reg_out_c; oc++)
        for (uint32_t i = 0; i < out_h; i++)
            for (uint32_t j = 0; j < out_w; j++)
                write_mem16(reg_out_addr +
                    (oc * out_h * out_w + i * out_w + j) * 2,
                    output[oc][i][j]);

    free_3d_act(input, reg_in_c, reg_in_h);
    free_4d_weight(weights, reg_out_c, reg_in_c, reg_kernel_size);
    free_1d_bias(bias);
    free_3d_output(output, reg_out_c, out_h);
}

// ======================================================
// MEMORY ACCESS
// ======================================================
int16_t IP::read_mem16(uint32_t addr)
{
    pl_t pl;
    sc_time delay = SC_ZERO_TIME;
    uint8_t buf[2];

    pl.set_command(tlm::TLM_READ_COMMAND);
    pl.set_address(addr);
    pl.set_data_length(2);
    pl.set_data_ptr(buf);

    mem_socket->b_transport(pl, delay);
    wait(delay);  // Kašnjenje pristupa memoriji (IP->BRAM)
    return (int16_t)((buf[1] << 8) | buf[0]);
}

void IP::write_mem16(uint32_t addr, int16_t val)
{
    pl_t pl;
    sc_time delay = SC_ZERO_TIME;
    uint8_t buf[2];

    buf[0] = val & 0xFF;
    buf[1] = (val >> 8) & 0xFF;

    pl.set_command(tlm::TLM_WRITE_COMMAND);
    pl.set_address(addr);
    pl.set_data_length(2);
    pl.set_data_ptr(buf);

    mem_socket->b_transport(pl, delay);
    wait(delay);  // Kašnjenje pristupa memoriji (IP->BRAM)
}
