#ifndef _CONV_HPP
#define _CONV_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <constraints.hpp>
#include <type_traits>

namespace cnn
{
    

    template <Numeric NUM_TYPE>
    class Conv : virtual Component<NUM_TYPE>
    {
        private:
            using namespace boost::numeric;

            ublas::tensor<NUM_TYPE> kernel_tensor;
            int stride;
        
        public:
            template <typename Fn>
            requires Initializer<Fn, NUM_TYPE>
            constexpr Conv(const std::initializer_list<std::size_t>& init_list, Fn initializer_lambda) : kernel_tensor{init_list} 
            {
                //... initialize the Tensor with values taken from initializer_lambda
            }

            Conv (const Conv& c) : kernel_tensor {}

            

    };
}

#endif