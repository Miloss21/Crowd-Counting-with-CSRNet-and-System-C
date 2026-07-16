# ==========================================
# Makefile za SystemC VP (ESL)
# ==========================================
# SystemC se traži u standardnim include
# i library putanjama sistema.
# ==========================================

TARGET = vp

# Izvorni fajlovi
SRCS = main.cpp cpu.cpp ip.cpp interconnect.cpp memory.cpp vp.cpp
OBJS = $(SRCS:.cpp=.o)

# Kompajler
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
LDFLAGS  = -lsystemc -lpthread

# ==========================================
# Pravila
# ==========================================

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
