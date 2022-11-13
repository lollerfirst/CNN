#ifndef _CONSTRAINTS_HPP
#define _CONSTRAINTS_HPP

#include <type_traits>
#include <concepts>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/tensor.hpp>

namespace cnn
{
    using namespace boost::numeric;

    template <typename N>
    concept Numeric = std::is_floating_point_v<N> || std::is_integral_v<N>;

    template <typename Fn, typename RET_TYPE>
    concept Initializer = std::is_invocable_v<Fn> &&
                            std::is_same_v<RET_TYPE, std::result_of_t<Fn(void)>;
    
    template <typename U, typename NUM_TYPE>
    concept DataShape = std::is_same_v<U, typename ublas::tensor<NUM_TYPE>> ||
                        std::is_same_v<U, typename ublas::vector<NUM_TYPE>>;


}

#endif