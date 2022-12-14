#ifndef _FLATTEN_HPP
#define _FLATTEN_HPP

#include <component.hpp>
#include <constraints.hpp>

namespace cnn
{
    using namespace boost::numeric;

    template <Numeric NUM_TYPE>
    class Flatten : Component
    {
        public:
            constexpr Flatten() : comptype {FLATTEN} {}
            ~Flatten() {}

            ublas::vector<NUM_TYPE> apply(const ublas::tensor<NUM_TYPE>& in_tensor) const;
            ublas::tensor<NUM_TYPE> update(const ublas::vector<NUM_TYPE>& gradient_vector);
    }

}

#endif