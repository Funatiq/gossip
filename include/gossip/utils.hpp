#pragma once

#include <stdexcept>
#include <iostream>

#define THROW_EXCEPTIONS 1

bool check(bool statement, const char* message) {
    if(!statement) {
#ifdef THROW_EXCEPTIONS
            throw std::invalid_argument(message);
#else
            std::cerr << message << std::endl;
            return false;
#endif
    }
    return true;
}
