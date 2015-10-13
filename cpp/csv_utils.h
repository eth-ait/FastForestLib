//
//  csv_reader.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 09/10/15.
//
//

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <iostream>
#include <sstream>
#include "mpl_utils.h"

#include <boost/iterator/iterator_facade.hpp>

namespace ait
{
    
    /// @brief Trim leading whitespacess from a std::string.
    inline std::string& trim_leading(std::string& str)
    {
        str.erase(str.begin(),
                  std::find_if(str.begin(), str.end(),
                               std::not1(std::function<bool(int)>([] (int c) -> bool { return std::isspace(c); }
        ))));
        return str;
    }
    
    /// @brief Trim trailing whitespacess from a std::string.
    inline std::string& trim_trailing(std::string& str)
    {
        str.erase(std::find_if(str.rbegin(), str.rend(),
                               std::not1(std::function<bool(int)>([] (int c) -> bool { return std::isspace(c); }
        ))).base(), str.end());
        return str;
    }
    
    /// @brief Trim leading and trailing whitespacess from a std::string.
    inline std::string& trim(std::string& str)
    {
        return trim_trailing(trim_leading(str));
    }

    template <typename T>
    inline T convert_from_string(const std::string& str)
    {
        std::istringstream sin(str);
        T value;
        sin >> value;
        return value;
    }
    
    template <typename T, char delimiter = ','>
    class CSVReader
    {
        class CSVIterator
        : public boost::iterator_facade<CSVIterator, const std::vector<T>, boost::forward_traversal_tag>
        {
            friend class CSVReader;
            friend class boost::iterator_core_access;

            explicit CSVIterator()
            : sin_ptr_(nullptr)
            {}

            explicit CSVIterator(std::istream* sin_ptr)
            : sin_ptr_(sin_ptr)
            {}

            void read_next_line()
            {
                row_.clear();
                
                std::string line;
                std::getline(*sin_ptr_, line);
                
                std::istringstream sline(line);
                std::string cell;
                while (std::getline(sline, cell, delimiter))
                {
                    T value;
                    std::istringstream stmp(cell);
                    stmp >> value;
                    eval_if_string<T, std::string&(std::string&)>::eval(value, trim);
                    row_.push_back(value);
                }
            }

            void increment()
            {
                read_next_line();
                if (row_.size() == 0)
                {
                    sin_ptr_ = nullptr;
                }
            }

            bool equal(const CSVIterator &other) const
            {
                return this->sin_ptr_ == other.sin_ptr_;
            }
            
            typename CSVIterator::iterator_facade_::reference& dereference() const
            {
                return row_;
            }

            std::istream* sin_ptr_;
            typename CSVIterator::iterator_facade_::value_type row_;
        };

    public:
        using iterator = CSVIterator;
        
        explicit CSVReader(std::istream& sin)
        : sin_(sin)
        {}
        
        iterator begin()
        {
            iterator it = iterator(&sin_);
            return ++it;
        }

        iterator end()
        {
            return iterator();
        }

    private:
        std::istream& sin_;
    };

}
