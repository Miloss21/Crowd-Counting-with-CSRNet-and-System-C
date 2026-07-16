#include "memory.hpp"
#include <cstdint>

using namespace sc_core;

Memory::Memory(sc_module_name name) : sc_module(name) {
    memory_socket.register_b_transport(this, &Memory::b_transport);
    memory_to_ip_socket.register_b_transport(this, &Memory::b_transport);

    mem.assign(MEMORY_SIZE, 0);

    SC_REPORT_INFO("Memory", "Constructed.");
}

Memory::~Memory() {
    SC_REPORT_INFO("Memory", "Destroyed.");
}

void Memory::b_transport(pl_t &pl, sc_time &offset) {
    tlm::tlm_command cmd = pl.get_command();
    uint32_t addr = static_cast<uint32_t>(pl.get_address());
    unsigned int len = pl.get_data_length();
    unsigned char *buf = pl.get_data_ptr();

    // Provera granica
    if (addr + len > MEMORY_SIZE) { 
        SC_REPORT_ERROR("Memory", "Nevalidan pristup memoriji!");
        pl.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
        offset += sc_time(10, SC_NS);
        return;
    }

    if (cmd == tlm::TLM_WRITE_COMMAND) {
        for (unsigned int i = 0; i < len; ++i) {
            mem[addr + i] = buf[i];
        }
        pl.set_response_status(tlm::TLM_OK_RESPONSE);
    }
    else if (cmd == tlm::TLM_READ_COMMAND) {
        for (unsigned int i = 0; i < len; ++i) {
            buf[i] = mem[addr + i];
        }
        pl.set_response_status(tlm::TLM_OK_RESPONSE);
    }
    else {
        pl.set_response_status(tlm::TLM_COMMAND_ERROR_RESPONSE);
    }

    offset += sc_time(10, SC_NS);
}
