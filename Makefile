KOKKOS_PATH = ${HOME}/kokkos-master
KOKKOS_KERNELS_PATH = ${HOME}/kokkos-kernels-master
KOKKOS_DEVICES = "Cuda, OpenMP"
EXE_NAME = "gnnAlgorithm"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX_LIST = gnnAlgorithm.cpp getGraphSize.cpp readGraphData.cpp initialiseWeights.cpp accuracyErrorUtil.cpp generateNodeEmbeddings.cpp forwardPropogate.cpp backPropogate.cpp
 
CXXFLAGS += -DUSING_THRUST
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "Ampere80"
KOKKOS_CUDA_OPTIONS=enable_lambda
CXXFLAGS += -I/dcs/large/u2145461/CUDA/include -L/dcs/large/u2145461/CUDA/lib64 -lcudart -lcublas -lcuda -lcudnn

LINK += -I/dcs/large/u2145461/CUDA/include -L/dcs/large/u2145461/CUDA/lib64 -lcudart -lcublas -lcuda -lcudnn

else
CXX_LIST = gnnAlgorithm.cpp
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "BDW"
endif

ifdef SILO
SILO_DIR = /modules/cs257/silo-4.11
CXXFLAGS += -DUSING_SILO -I$(SILO_DIR)/include
LIB_PATHS += -L$(SILO_DIR)/lib -lsilo
CXX_LIST += writeTimestep.cpp
endif

LINK = ${CXX}

CXXFLAGS += -O3 -march=native -funroll-loops -fopenmp-simd

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

ifdef TINY
TINY_DIR = ./tiny-dnn-master/tiny_dnn
CXXFLAGS += -I$(TINY_DIR) -DCNN_NO_SERIALIZATION -DUSING_TINY
LIB_PATHS += -L$(TINY_DIR)
endif

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) $(LIB_PATHS) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(LIB_PATHS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
