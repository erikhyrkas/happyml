//
// Created by Erik Hyrkas on 12/19/2022.
//

#ifndef MICROML_PORTABLE_BYTES_HPP
#define MICROML_PORTABLE_BYTES_HPP
#include <iostream>

// windows uses little endian order, but big endian is more common for other operating systems.
// I decided to implement my own byte swap for now because I didn't want to bring in
// any external libraries, not even winsock.h for hton (host to network.)
//
// Cross-platform C++ programs tend to be obnoxiously hard to read and ugly. I'm going to
// try to avoid external library dependencies to simplify compiling the application and
// an excessive amount of directives that are OS specific. We'll see if I regret this later.
//
// Really, I imagine there's a lot of work to be done to make this project
// portable across operating systems, but this was a problem I wanted to tackle
// while I had it on my mind. There are probably going to be compiling issues
// using gcc that I'll deal with later and issues around 32-bit, but I'll get to them.
// I've more-or-less tried to stay on top of it, but I haven't been testing
// with multiple operating systems or compilers. Eventually, I'll clean it up.
//
// TODO: I didn't go for memory efficient in my implementation below. I really should, but
//  I went for something that works without any thought or effort. I will streamline the
//  functions later.

#ifdef _WIN32
#define SWAP_BYTES 1
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define SWAP_BYTES 1
#else
#define SWAP_BYTES 0
#endif


void printBits(const uint64_t x) {
    for (int i = 63; i >= 0; i--) {
        cout << ((x >> i) & 1);
        if ((i%8) == 0) cout << " ";
    }
    cout << endl;
}

#if SWAP_BYTES == 1
uint64_t portableBytes(uint64_t bytes) {
    uint64_t a = (bytes & 0x00000000000000ff) << 56;
    uint64_t b = (bytes & 0x000000000000ff00) << 40;
    uint64_t c = (bytes & 0x0000000000ff0000) << 24;
    uint64_t d = (bytes & 0x00000000ff000000) << 8;
    uint64_t e = (bytes & 0x000000ff00000000) >> 8;
    uint64_t f = (bytes & 0x0000ff0000000000) >> 24;
    uint64_t g = (bytes & 0x00ff000000000000) >> 40;
    uint64_t h = (bytes & 0xff00000000000000) >> 56;
    return a | b | c | d | e | f | g | h;
}

uint32_t portableBytes(uint32_t bytes) {
    uint32_t a = (bytes & 0x000000ff) << 24;
    uint32_t b = (bytes & 0x0000ff00) << 8;
    uint32_t c = (bytes & 0x00ff0000) >> 8;
    uint32_t d = (bytes & 0xff000000) >> 24;
    return  a | b | c | d;
}

uint16_t portableBytes(uint16_t bytes) {
    return (bytes << 8) | (bytes >> 8);
}
#else
// we're all ready big endian, so just pass through. Nothing to see here.

uint64_t portableBytes(uint64_t bytes) {
    return bytes;
}

uint32_t portableBytes(uint32_t bytes) {
    return bytes;
}

uint16_t portableBytes(uint16_t bytes) {
    return bytes;
}

#endif

#endif //MICROML_PORTABLE_BYTES_HPP
