#ifndef _FLATTEN_HPP
#define _FLATTEN_HPP

#include <component.hpp>
#include <constraints.hpp>

namespace cnn
{
    using namespace boost::numeric;

    template <Numeric NUM_TYPE>
    class Flatten : virtual Component<NUM_TYPE>
    {
        public:
            Flatten() : comptype {FLATTEN} {}
            ~Flatten() = default;

            ublas::vector<NUM_TYPE> apply(const ublas::tensor<NUM_TYPE>& in_tensor) const override;
            ublas::tensor<NUM_TYPE> update(const ublas::vector<NUM_TYPE>& gradient_vector) const override;
    }

}

#endif