//
//  mpl_utils.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 09/10/15.
//
//

#pragma once

#include <boost/utility/enable_if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/has_xxx.hpp>

namespace ait
{

    // TODO: Extract into mpl_utils.h
    BOOST_MPL_HAS_XXX_TRAIT_DEF(is_saving)
    BOOST_MPL_HAS_XXX_TRAIT_DEF(is_loading)
    
    template <typename T> using is_boost_archive = boost::mpl::or_<has_is_saving<T>, has_is_loading<T>>;
    template <typename T> using enable_if_boost_archive = boost::enable_if<is_boost_archive<T>, bool>;
    template <typename T> using disable_if_boost_archive = boost::disable_if<is_boost_archive<T>, bool>;
    
    template <typename T, typename Signature>
    struct eval_if_string;

    template <typename T, typename R, typename... Args>
    struct eval_if_string<T, R(Args...)>
    {
        static T& eval(T& t, const std::function<R(Args...)>& f)
        {
            return t;
        }
    };
    
    template <typename R, typename... Args>
    struct eval_if_string<std::string, R(Args...)>
    {
        static std::string& eval(std::string& t, const std::function<R(Args...)>& f)
        {
            return f(t);
        }
    };

}
