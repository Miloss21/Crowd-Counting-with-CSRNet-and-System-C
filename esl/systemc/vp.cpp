#include "vp.hpp"

Vp::Vp(sc_core::sc_module_name name)
    : sc_module(name),
      ip("IP"),
      cpu("Cpu", &ip),
      interconnect("Interconnect"),
      memory("Memory")
{
    // CPU -> Interconnect
    cpu.interconnect_socket.bind(interconnect.cpu_socket);

    // Interconnect -> Memory (CPU accesses RAM)
    interconnect.memory_socket.bind(memory.memory_socket);

    // Interconnect -> IP registers
    interconnect.ip_socket.bind(ip.ip_socket);

    // IP -> Memory (DMA)
    ip.mem_socket.bind(memory.memory_to_ip_socket);

    SC_REPORT_INFO("Virtual Platform", "Constructed.");
}

Vp::~Vp() {
    SC_REPORT_INFO("Virtual Platform", "Destroyed.");
}
