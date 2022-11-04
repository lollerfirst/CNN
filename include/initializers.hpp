#ifndef _INITIALIZERS_HPP
#define _INITIALIZERS_HPP

#include <constraints.hpp>
#include <type_traits>
#include <random>

namespace cnn{
    
    /**
     * @brief default initializer lambda returns random uniform values.
     * 
     * @tparam RET_TYPE type of the CNN 
     */
    template <Numeric RET_TYPE>
    auto default_initializer = []() -> RET_TYPE
    {
        
        if constexpr (std::is_floating_point_v<RET_TYPE>)
        {
            static std::uniform_real_distribution<RET_TYPE> distribution{-1, 1};
            return distribution(generator);
        }
        else
        {
            static std::uniform_int_distribution<RET_TYPE> distribution{0, SHRT_MAX};
            return distribution(generator);
        }

    };
}

#endif