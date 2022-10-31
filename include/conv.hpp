#ifndef _CONV_HPP
#define _CONV_HPP

#include <boost/numeric/ublas/tensor.hpp>

namespace cnn
{
    template <typename NUM_TYPE>
    class Conv
    {
        private:
            using namespace boost::numeric;

            ublas::tensor<NUM_TYPE> kernel_tensor;
        
        public:
            constexpr 
    };
}

#endif