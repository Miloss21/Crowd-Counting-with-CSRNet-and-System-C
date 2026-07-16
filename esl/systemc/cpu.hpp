
#ifndef CPU_HPP
#define CPU_HPP

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include "defines.hpp"
#include "ip.hpp"
#include <fstream>

class Cpu : public sc_core::sc_module
{
public:
    tlm_utils::simple_initiator_socket<Cpu> interconnect_socket;
    
    SC_HAS_PROCESS(Cpu);
    
    Cpu(sc_core::sc_module_name name, IP* ip_ptr);
    ~Cpu();

private:
    typedef tlm::tlm_generic_payload pl_t;
    
    sc_core::sc_time offset;
    IP* ip;
    
    // Data buffers
    int16_t input_fixed[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
    int16_t weights[OUTPUT_CHANNELS][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    int16_t bias[OUTPUT_CHANNELS];  // ✅ DODATO
    
    // Methods
    void execute();
    void load_preprocessed_data();
    void write_memory(uint32_t addr, const uint8_t* data, uint32_t len);
    void write_ip_reg(uint8_t reg, uint32_t val);
    uint32_t read_ip_reg(uint8_t reg);
    void save_output_to_file();
};

#endif // CPU_HPP