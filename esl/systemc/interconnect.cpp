#include "interconnect.hpp"

Interconnect::Interconnect(sc_core::sc_module_name name)
    : sc_module(name), memory_socket("memory_socket"), ip_socket("ip_socket"), cpu_socket("cpu_socket") {
    cpu_socket.register_b_transport(this, &Interconnect::b_transport);
    SC_REPORT_INFO("Interconnect", "Constructed.");
}

Interconnect::~Interconnect() {
    SC_REPORT_INFO("Interconnect", "Destroyed.");
}

void Interconnect::b_transport(pl_t &pl, sc_core::sc_time &offset)
{
    sc_dt::uint64 addr = pl.get_address();

    // ================= MEMORY =================
    if (addr >= VP_ADDR_MEMORY_L && addr <= VP_ADDR_MEMORY_H) {
        sc_dt::uint64 taddr = addr - VP_ADDR_MEMORY_L;
        pl.set_address(taddr);
        memory_socket->b_transport(pl, offset);
        pl.set_address(addr);
        return;
    }

    // ================= IP =================
    if (addr >= VP_ADDR_IP_L && addr <= VP_ADDR_IP_H) {
        // ❗❗ NE DIRAJ ADRESU ❗❗
        ip_socket->b_transport(pl, offset);
        offset += sc_core::sc_time(5 * DELAY, sc_core::SC_NS);
        return;
    }

    SC_REPORT_ERROR("Interconnect", "Wrong address");
    pl.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
}