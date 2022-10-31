#ifndef _INITIALIZERS_HPP
#define _INITIALIZERS_HPP

namespace cnn{
    template <Numeric RET_TYPE>
    auto default_initializer = []() -> RET_TYPE
    {
        constexpr if (std::is_floating_point_v<RET_TYPE>)
        {
            
        }
        else
        {

        }
    }
}

#endif