#ifndef _CONSTRAINTS_HPP
#define _CONSTRAINTS_HPP

#include <type_traits>
#include <concepts>

namespace cnn
{
    template <typename N>
    concept Numeric = std::is_floating_point_v<N> || std::is_integral_v<N>;

    template <typename Fn, typename RET_TYPE>
    concept Initializer = std::is_callable_v<Fn> && std::same_as_v<RET_TYPE, std::result_of_t<Fn(void)>;
}

#endif