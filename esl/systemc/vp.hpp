#ifndef VP_HPP
#define VP_HPP

#include <systemc>
#include "cpu.hpp"
#include "interconnect.hpp"
#include "ip.hpp"
#include "memory.hpp"

class Vp : public sc_core::sc_module {
public:
    Vp(sc_core::sc_module_name name);
    ~Vp();

protected:
    IP ip;
    Cpu cpu;
    Interconnect interconnect;
    Memory memory;
};

#endif // VP_HPP