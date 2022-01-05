CC = nvcc

CFLAGS = util.cpp -g -O3

TARGET = wcfa

$(TARGET): $(TARGET).cu
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cu

clean:
	$(RM) $(TARGET);\
	rm -f res/*;\

check:
	rm -f res/*;\
	./all_tests.sh;\
	./verify_results.sh;\
