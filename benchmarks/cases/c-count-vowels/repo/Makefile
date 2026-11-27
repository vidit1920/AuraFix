CC = gcc
CFLAGS = -Wall -Iinclude

.PHONY: test
test: build/test_text
	./build/test_text

build/test_text: src/text.c tests/test_text.c include/text.h
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ src/text.c tests/test_text.c
