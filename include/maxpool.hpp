#ifndef _MAXPOOL_HPP
#define _MAXPOOL_HPP

#include <contraints.hpp>
#include <component.hpp>
#include <boost/numeric/ublas/tensor.hpp>

#include <utility>

namespace cnn
{
    using namespace boost::numeric;

    template <Numeric NUM_TYPE>
    class MaxPool : virtual Component<NUM_TYPE>
    {
        private:
            std::pair<std::size_t> dimensions;
            short stride;

        public:
            constexpr MaxPool(std::size_t dim1 = 3UL, std::size_t dim2 = 3UL, short strd = 3) :
            dimensions{dim1, dim2}, stride{strd} {}

            ublas::tensor<NUM_TYPE> apply(const ublas::tensor<NUM_TYPE>& in_tensor) const override;

            ublas::tensor<NUM_TYPE> update(const ublas::tensor<NUM_TYPE>& gradient_tensor) const override;
    };
}

#endif