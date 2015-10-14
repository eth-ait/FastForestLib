//
//  serialization_utils.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 14/10/15.
//
//

#pragma once

#include <tuple>

namespace
{
    template <int index>
    struct TupleSerializer
    {
        template <typename Archive, typename ...Args>
        static void serialize(Archive& archive, std::tuple<Args...>& t, const unsigned int version)
        {
            TupleSerializer<index - 1>::template serialize(archive, t, version);
            archive & std::get<index>(t);
        }
    };
    
    template <>
    struct TupleSerializer<0>
    {
        template <typename Archive, typename ...Args>
        static void serialize(Archive& archive, std::tuple<Args...>& t, const unsigned int version)
        {
        }
    };
}

namespace boost
{
namespace serialization
{

template <typename Archive, typename ...Args>
void serialize(Archive& archive, std::tuple<Args...>& t, const unsigned int version)
{
    constexpr int tuple_size = std::tuple_size<std::tuple<Args...>>::value;
    TupleSerializer<tuple_size - 1>::template serialize(archive, t, version);
}

}
}
