#ifndef INTERCONNECT_HPP
#define INTERCONNECT_HPP

#include <systemc>
#include "defines.hpp"
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

class Interconnect : public sc_core::sc_module {
public:
    Interconnect(sc_core::sc_module_name name);
    ~Interconnect();

    tlm_utils::simple_initiator_socket<Interconnect> memory_socket;
    tlm_utils::simple_initiator_socket<Interconnect> ip_socket;
    tlm_utils::simple_target_socket<Interconnect> cpu_socket;
   
protected:
    typedef tlm::tlm_generic_payload pl_t;  // ✅ Dodaj ovo
    void b_transport(pl_t &pl, sc_core::sc_time &offset);
    
};

#endif // INTERCONNECT_HPP