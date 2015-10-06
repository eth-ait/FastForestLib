//
//  ait.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 29/09/15.
//
//

#pragma once

#include <cstdint>
#include <Eigen/Dense>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/utility/enable_if.hpp>

namespace ait
{

//typedef std::int64_t size_type;
using size_type = EIGEN_DEFAULT_DENSE_INDEX_TYPE;
using scalar_type = double;

template <typename BaseIterator>
class PointerIteratorWrapper : public boost::iterator_adaptor<PointerIteratorWrapper<BaseIterator>, BaseIterator, typename BaseIterator::value_type::element_type>
{
public:
    PointerIteratorWrapper()
    : PointerIteratorWrapper::iterator_adapter_(0)
    {}
    
    explicit PointerIteratorWrapper(BaseIterator it)
    : PointerIteratorWrapper::iterator_adaptor_(it)
    {}
    
    template <typename OtherBaseIterator>
    PointerIteratorWrapper(
                           const PointerIteratorWrapper<OtherBaseIterator> &other,
                           typename boost::enable_if<
                           boost::is_convertible<OtherBaseIterator, BaseIterator>, int>::type = 0
                           )
    : PointerIteratorWrapper::iterator_adaptor_(other.base())
    {}
    
private:
    friend class boost::iterator_core_access;
    typename PointerIteratorWrapper::iterator_adaptor_::reference dereference() const
    {
        return *(*this->base());
    }
};

template <typename BaseIterator>
inline PointerIteratorWrapper<BaseIterator> make_pointer_iterator_wrapper(BaseIterator it)
{
    return PointerIteratorWrapper<BaseIterator>(it);
}

}
