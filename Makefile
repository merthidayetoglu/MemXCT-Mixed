# ----- Make Macros -----

CXX = mpicxx
CXXFLAGS = -std=c++11 -fopenmp -I /gpfs/mira-home/merth/ThetaGPU/cuda-11.2/targets/x86_64-linux/include
OPTFLAGS = -O3

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++11 -gencode arch=compute_80,code=sm_80 -ccbin=mpicxx -Xcompiler -fopenmp -Xptxas="-v"

LD_FLAGS = -ccbin=mpicxx -Xcompiler -fopenmp

TARGETS = memxct
OBJECTS = preproc.o reducemap.o main.o raytrace.o kernels.o communications.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o:    %.cpp 
	${CXX} ${CXXFLAGS} ${OPTFLAGS} $^ -c -o $@

%.o:    %.cu 
	${NVCC} ${NVCCFLAGS} $^ -c -o $@

memxct: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt core *.html *.xml
