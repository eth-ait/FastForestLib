//
//  iterator_utils.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 07/10/15.
//
//

#pragma once

#include <boost/utility/enable_if.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace ait
{

template <typename T>
struct extract_element_type
{
    using type = typename T::element_type;
};
    
/// @brief An adapter that returns the underlying value from a iterator of a pointer.
/// @tparam BaseIterator The underlying iterator of a pointer type.
/// @tparam ValueType The underlying value type. The default will try to automatically infer the value type.
///
/// Some template magic is done to ensure that this can be used for both raw pointers as well as smart pointers.
///
template <typename BaseIterator, typename ValueType = boost::use_default>
class PointerIteratorWrapper
: public boost::iterator_adaptor<
    PointerIteratorWrapper<BaseIterator, ValueType>,
    BaseIterator,
    typename boost::mpl::eval_if<
        boost::is_same<boost::use_default, ValueType>,
        boost::mpl::eval_if<
        boost::is_pointer<typename BaseIterator::value_type>,
            boost::mpl::identity<boost::remove_pointer<typename BaseIterator::value_type>>,
            extract_element_type<typename BaseIterator::value_type>
        >,
        boost::mpl::identity<ValueType>
    >::type
>
{
public:
    explicit PointerIteratorWrapper(BaseIterator it)
    : PointerIteratorWrapper::iterator_adaptor_(it)
    {}
    
    template <typename OtherBaseIterator, typename OtherValueType>
    PointerIteratorWrapper(
        const PointerIteratorWrapper<OtherBaseIterator, OtherValueType> &other,
        typename boost::enable_if<
            boost::is_convertible<OtherBaseIterator, BaseIterator>,
            int
        >::type = 0
    )
    : PointerIteratorWrapper::iterator_adaptor_(other.base())
    {}
    
private:
    friend class boost::iterator_core_access;
    template <typename, typename> friend class PointerIteratorWrapper;

    typename PointerIteratorWrapper::iterator_adaptor_::reference dereference() const
    {
        return *(*this->base());
    }
};

/// @brief Create an adapter of a pointer iterator that directly returns the value instead of the pointer.
template <typename BaseIterator, typename ValueType = boost::use_default>
inline PointerIteratorWrapper<BaseIterator, ValueType> make_pointer_iterator_wrapper(BaseIterator it)
{
    return PointerIteratorWrapper<BaseIterator, ValueType>(it);
}

}
