//
//  io_utils.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 15/03/16.
//
//

#pragma once

#include <iostream>
#include <fstream>

template <typename char_type, typename element_type>
std::basic_ostream<char_type>& operator<<(std::basic_ostream<char_type>& sout, const std::vector<element_type>& elements) {
    sout << "{";
    for (auto it = elements.cbegin(); it != elements.cend(); ++it) {
        if (it != elements.cbegin()) {
            sout << ", ";
        }
        sout << *it;
    }
    sout << "}";
    return sout;
}
