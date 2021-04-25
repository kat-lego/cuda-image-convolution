CFLAGS = -std=c++11 -I include/
CC = nvcc

convolution: main.o utils.o convolution.o
	$(CC) $(CFLAGS) main.o utils.o convolution.o -o convolution

main.o: src/main.cu
	$(CC) $(CFLAGS) -c src/main.cu

utils.o: src/utils.cu
	$(CC) $(CFLAGS) -c src/utils.cu

convolution.o: src/convolution.cu
	$(CC) $(CFLAGS) -c src/convolution.cu

clean:
	rm convolution *.o ../output/* ../resfiles/*
