#include "cpu.hpp"
#include <iostream>

using namespace std;

Cpu::Cpu(sc_core::sc_module_name name, IP* ip_ptr)
    : sc_module(name)
    , interconnect_socket("interconnect_socket")
    , offset(sc_core::SC_ZERO_TIME)
    , ip(ip_ptr)
{
    SC_THREAD(execute);
}

Cpu::~Cpu() {
    cout << "CPU modul završen." << endl;
}

// ======================================================
// LOAD DATA
// ======================================================
void Cpu::load_preprocessed_data() {
    cout << "[CPU] Učitavanje preprocessovanih podataka..." << endl;

    std::ifstream img_file("../../data/input_image.bin", std::ios::binary);
    if (!img_file)
        SC_REPORT_FATAL("CPU", "Ne mogu otvoriti input_image.bin!");

    img_file.read(reinterpret_cast<char*>(input_fixed), INPUT_SIZE);
    img_file.close();

    cout << "[CPU] Slika učitana: "
         << INPUT_CHANNELS << "x" << INPUT_HEIGHT << "x" << INPUT_WIDTH
         << " (" << INPUT_SIZE << " bytes)" << endl;

    std::ifstream weights_file("../../data/weights_conv1.bin", std::ios::binary);
    if (!weights_file)
        SC_REPORT_FATAL("CPU", "Ne mogu otvoriti weights_conv1.bin!");

    weights_file.read(reinterpret_cast<char*>(weights), WEIGHTS_SIZE);
    weights_file.close();

    cout << "[CPU] Težine učitane: "
         << OUTPUT_CHANNELS << "x" << INPUT_CHANNELS << "x"
         << KERNEL_SIZE << "x" << KERNEL_SIZE
         << " (" << WEIGHTS_SIZE << " bytes)" << endl;

    // ✅ UČITAJ BIAS
    std::ifstream bias_file("../../data/bias_conv1.bin", std::ios::binary);
    if (!bias_file)
        SC_REPORT_FATAL("CPU", "Ne mogu otvoriti bias_conv1.bin!");

    bias_file.read(reinterpret_cast<char*>(bias), BIAS_SIZE);
    bias_file.close();

    cout << "[CPU] Bias učitan: "
         << OUTPUT_CHANNELS << " elemenata"
         << " (" << BIAS_SIZE << " bytes)" << endl;
}

// ======================================================
// MEMORY WRITE (FIXED)
// ======================================================
void Cpu::write_memory(uint32_t addr, const uint8_t* data, uint32_t len) {
    pl_t pl;
    offset = sc_core::SC_ZERO_TIME;

    pl.set_command(tlm::TLM_WRITE_COMMAND);
    pl.set_address(addr);
    pl.set_data_length(len);
    pl.set_data_ptr(const_cast<uint8_t*>(data));
    pl.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

    interconnect_socket->b_transport(pl, offset);

    if (pl.get_response_status() != tlm::TLM_OK_RESPONSE)
        SC_REPORT_FATAL("CPU", "Greška pri upisu u memoriju!");

    wait(offset);
}

// ======================================================
// IP REGISTER WRITE (FIXED)
// ======================================================
void Cpu::write_ip_reg(uint8_t reg, uint32_t val)
{
    pl_t pl;
    sc_core::sc_time delay = sc_core::SC_ZERO_TIME;

    uint8_t data[4];
    data[0] =  val        & 0xFF;
    data[1] = (val >> 8)  & 0xFF;
    data[2] = (val >> 16) & 0xFF;
    data[3] = (val >> 24) & 0xFF;

    pl.set_command(tlm::TLM_WRITE_COMMAND);
    pl.set_address(VP_ADDR_IP_L + reg);
    pl.set_data_ptr(data);
    pl.set_data_length(4);
    pl.set_streaming_width(4);
    pl.set_byte_enable_ptr(nullptr);
    pl.set_dmi_allowed(false);
    pl.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

    interconnect_socket->b_transport(pl, delay);

    if (pl.get_response_status() != tlm::TLM_OK_RESPONSE) {
        SC_REPORT_FATAL("CPU", "Greška pri upisu IP registra!");
    }
    wait(delay);  //kasnjenje
}

// ======================================================
// IP REGISTER READ (FIXED)
// ======================================================
uint32_t Cpu::read_ip_reg(uint8_t reg) {
    pl_t pl;
    uint8_t buf[4];
    offset = sc_core::SC_ZERO_TIME;

    pl.set_command(tlm::TLM_READ_COMMAND);
    pl.set_address(VP_ADDR_IP_L + reg);
    pl.set_data_length(4);
    pl.set_data_ptr(buf);
    pl.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

    interconnect_socket->b_transport(pl, offset);

    if (pl.get_response_status() != tlm::TLM_OK_RESPONSE)
        SC_REPORT_FATAL("CPU", "Greška pri čitanju IP registra!");

    wait(offset);

    return (buf[3] << 24) | (buf[2] << 16) | (buf[1] << 8) | buf[0];
}

// ======================================================
// MAIN CPU THREAD
// ======================================================
void Cpu::execute() {
    cout << "\n========================================" << endl;
    cout << "      CPU: Početak izvršavanja" << endl;
    cout << "========================================\n" << endl;

    sc_core::sc_time start_time = sc_core::sc_time_stamp();

    load_preprocessed_data();

    cout << "\n[CPU] Slanje ulazne slike u memoriju..." << endl;
    for (int c = 0; c < INPUT_CHANNELS; c++) {
        for (int i = 0; i < INPUT_HEIGHT; i++) {
            uint32_t addr = INPUT_ADDR + (c * INPUT_HEIGHT * INPUT_WIDTH + i * INPUT_WIDTH) * 2;
            write_memory(addr, reinterpret_cast<uint8_t*>(input_fixed[c][i]), INPUT_WIDTH * 2);
        }
        cout << "[CPU] Poslato " << (c + 1) << "/" << INPUT_CHANNELS << " kanala" << endl;
    }

    cout << "\n[CPU] Slanje težina u memoriju..." << endl;
    for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
        for (int ic = 0; ic < INPUT_CHANNELS; ic++) {
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                uint32_t addr = WEIGHTS_ADDR +
                    ((oc * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE) +
                     (ic * KERNEL_SIZE * KERNEL_SIZE) +
                     (kh * KERNEL_SIZE)) * 2;
                write_memory(addr, reinterpret_cast<uint8_t*>(weights[oc][ic][kh]), KERNEL_SIZE * 2);
            }
        }
        if ((oc + 1) % 16 == 0)
            cout << "[CPU] Poslato " << (oc + 1) << "/" << OUTPUT_CHANNELS << endl;
    }

    wait(sc_core::SC_ZERO_TIME);

    // ✅ SLANJE BIAS-a U MEMORIJU
    cout << "\n[CPU] Slanje bias-a u memoriju..." << endl;
    write_memory(BIAS_ADDR, reinterpret_cast<uint8_t*>(bias), BIAS_SIZE);
    cout << "[CPU] Bias poslat: " << OUTPUT_CHANNELS << " elemenata" << endl;

    cout << "\n[CPU] Konfiguracija IP bloka..." << endl;
    write_ip_reg(REG_IN_ADDR, INPUT_ADDR);
    write_ip_reg(REG_WEIGHTS_ADDR, WEIGHTS_ADDR);
    write_ip_reg(REG_BIAS_ADDR, BIAS_ADDR);
    write_ip_reg(REG_OUT_ADDR, OUTPUT_ADDR);
    write_ip_reg(REG_IN_H, INPUT_HEIGHT);
    write_ip_reg(REG_IN_W, INPUT_WIDTH);
    write_ip_reg(REG_IN_C, INPUT_CHANNELS);
    write_ip_reg(REG_OUT_C, OUTPUT_CHANNELS);
    write_ip_reg(REG_KERNEL_SIZE, KERNEL_SIZE);
    write_ip_reg(REG_STRIDE, STRIDE);
    write_ip_reg(REG_PADDING, PADDING);

    cout << "[CPU] Pokretanje IP bloka..." << endl;
    write_ip_reg(REG_CTRL, 0x01);

    cout << "[CPU] Čekanje da IP završi konvoluciju..." << endl;
    wait(ip->done_event);

    cout << "[CPU] IP blok završio konvoluciju!" << endl;
    save_output_to_file();

    sc_core::sc_time end_time = sc_core::sc_time_stamp();
    double duration = (end_time - start_time).to_seconds();

    cout << "\n========================================" << endl;
    cout << "      KRAJ CPU IZVRŠAVANJA" << endl;
    cout << "========================================" << endl;
    printf("⏱  Ukupno vreme: %.6f sekundi\n", duration);
    printf("📊 Throughput: %.2f FPS\n", 1.0 / duration);
    cout << "========================================\n" << endl;
}

// ======================================================
// SAVE OUTPUT TO FILE
// ======================================================
void Cpu::save_output_to_file() {
    cout << "\n[CPU] Snimanje output-a u fajl..." << endl;

    uint32_t out_h = (INPUT_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    uint32_t out_w = (INPUT_WIDTH  + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    uint32_t total_elems = OUTPUT_CHANNELS * out_h * out_w;

    int16_t* output_buffer = new int16_t[total_elems];

    cout << "[CPU] Čitanje output-a iz memorije..." << endl;

    for (uint32_t c = 0; c < OUTPUT_CHANNELS; c++) {
        for (uint32_t i = 0; i < out_h; i++) {
            for (uint32_t j = 0; j < out_w; j++) {
                uint32_t addr = OUTPUT_ADDR + (c * out_h * out_w + i * out_w + j) * 2;

                pl_t pl;
                uint8_t buf[2];

                pl.set_command(tlm::TLM_READ_COMMAND);
                pl.set_address(addr);
                pl.set_data_length(2);
                pl.set_data_ptr(buf);
                pl.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

                interconnect_socket->b_transport(pl, offset);
                wait(sc_core::sc_time(DELAY, sc_core::SC_NS));

                int16_t val = static_cast<int16_t>((buf[1] << 8) | buf[0]);
                output_buffer[c * out_h * out_w + i * out_w + j] = val;
            }
        }

        if ((c + 1) % 8 == 0) {
            cout << "[CPU] Pročitano " << (c + 1) << "/" << OUTPUT_CHANNELS << " kanala" << endl;
        }
    }

    std::ofstream file("../../data/output.bin", std::ios::binary);
    if (!file) {
        SC_REPORT_ERROR("CPU", "Ne mogu da kreiram output.bin!");
        delete[] output_buffer;
        return;
    }

    file.write(reinterpret_cast<char*>(output_buffer), total_elems * sizeof(int16_t));
    file.close();

    delete[] output_buffer;

    cout << "[CPU] ✅ Output sačuvan u ../../data/output.bin" << endl;
    cout << "[CPU]    Dimenzije: " << OUTPUT_CHANNELS << "x" << out_h << "x" << out_w << endl;
}