#ifndef _CONV_HPP
#define _CONV_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <constraints.hpp>
#include <component.hpp>
#include <type_traits>

namespace cnn
{

    /**
     * @brief Default filters for a 3x3x3 convolution
     * 
     */
    static double default_filters[] = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0,  // SHARPEN

        -1, -1, -1,
        -1, 4, -1,
        -1, -1, -1, //RIDGE DETECTION

        0, 0, 0,
        0, 1, 0,
        0, 0, 0    // IDENTITY
    };

    template <Numeric NUM_TYPE>
    class Conv : virtual Component<NUM_TYPE>
    {
        private:
            using namespace boost::numeric;

            ublas::tensor<NUM_TYPE> kernel_tensor;
            short stride;
        
        public:
            template <typename Fn>
            requires Initializer<Fn, NUM_TYPE>
            constexpr Conv(const std::initializer_list<std::size_t>& init_list, Fn initializer_lambda) : kernel_tensor{init_list};

            constexpr Conv() : kernel_tensor{{3, 3, 3}};

            Conv (const Conv& c) : kernel_tensor {c.kernel_tensor}, comptype {c.comptype} {}
            Conv (Conv&& c) : kernel_tensor {std::move(c.kernel_tensor)}, comptype {c.comptype} {}

            ~Conv ()
            {
                ~kernel_tensor();
            }


            ublas::tensor<NUM_TYPE> apply(const ublas::tensor<NUM_TYPE>& out_tensor) const override;
            ublas::tensor<NUM_TYPE> update(const ublas::tensor<NUM_TYPE>& gradient_tensor) const override;
            

    };
}

#endif