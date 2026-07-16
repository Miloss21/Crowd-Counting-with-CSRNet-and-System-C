#ifndef IP_HPP
#define IP_HPP

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_target_socket.h>
#include <tlm_utils/simple_initiator_socket.h>
#include "defines.hpp"

class IP : public sc_core::sc_module
{
public:
    tlm_utils::simple_target_socket<IP> ip_socket;
    tlm_utils::simple_initiator_socket<IP> mem_socket;
    
    sc_core::sc_event start_event;
    sc_core::sc_event done_event;
    
    SC_HAS_PROCESS(IP);
    
    IP(sc_core::sc_module_name name);
    ~IP();

private:
    typedef tlm::tlm_generic_payload pl_t;
    
    sc_core::sc_time offset;
    bool busy;
    
    // Registers
    uint32_t reg_ctrl;
    uint32_t reg_status;
    uint32_t reg_in_addr;
    uint32_t reg_weights_addr;
    uint32_t reg_bias_addr;      // ✅ DODATO
    uint32_t reg_out_addr;
    uint32_t reg_in_h;
    uint32_t reg_in_w;
    uint32_t reg_in_c;
    uint32_t reg_out_c;
    uint32_t reg_kernel_size;
    uint32_t reg_stride;
    uint32_t reg_padding;
    
    // Methods
    void b_transport(pl_t &pl, sc_core::sc_time &delay);
    void ip_main_thread();
    void perform_convolution();
    
    int16_t read_mem16(uint32_t addr);
    void write_mem16(uint32_t addr, int16_t val);
};

#endif // IP_HPP