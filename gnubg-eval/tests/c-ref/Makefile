CC ?= gcc
CFLAGS ?= -O2 -Wno-unused-result
TARGET := harness

.PHONY: all run clean

all: $(TARGET)

$(TARGET): harness.c
	$(CC) $(CFLAGS) -o $@ $< -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
