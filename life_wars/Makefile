include ../common/common.mk

all: life_wars

life_wars: life_wars.cu
	$(NVCC) $(NVCCFLAGS) $(NCCL_INC) -o $@ $< $(NCCL_LIB)

run: all
	./run.sh

clean:
	rm -rf life_wars frames out