#ifndef _CONSTRAINTS_HPP
#define _CONSTRAINTS_HPP

#include <type_traits>
#include <concepts>

namespace cnn
{
    template <typename N>
    concept Numeric = std::is_floating_point_v<N> || std::is_integral_v<N>;
}

#endif