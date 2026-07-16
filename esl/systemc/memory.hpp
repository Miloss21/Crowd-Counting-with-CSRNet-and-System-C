#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_target_socket.h>
#include <vector>
#include "defines.hpp"

class Memory : public sc_core::sc_module {
public:
    Memory(sc_core::sc_module_name name);
    ~Memory();

    tlm_utils::simple_target_socket<Memory> memory_socket;
    tlm_utils::simple_target_socket<Memory> memory_to_ip_socket;

protected:
    typedef tlm::tlm_generic_payload pl_t;  // ✅ Dodaj ovo
    void b_transport(pl_t &pl, sc_core::sc_time &offset);
    std::vector<unsigned char> mem;
    
};

#endif // MEMORY_HPP