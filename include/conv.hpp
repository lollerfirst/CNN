#ifndef _CONV_HPP
#define _CONV_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <constraints.hpp>
#include <component.hpp>
#include <type_traits>

namespace cnn
{
    using namespace boost::numeric;

    /**
     * @brief Default filters for a 3x3x3 convolution
     * 
     */
    static double default_filters[] =
    {
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
            ublas::tensor<NUM_TYPE> kernel_tensor;
            short stride;
        
        public:
            
            template <typename Fn>
            requires Initializer<Fn, NUM_TYPE>
            constexpr Conv(std::size_t dim1_kernelmatrix, std::size_t dim2_kernelmatrix, std::size_t n_filters, short strd, Fn initializer_lambda)
            :
            comptype{CONV},
            kernel_tensor{{dim1_kernelmatrix, dim2_kernelmatrix, n_filters}},
            stride{strd}
            {
                // tensor custom initialization
                for (auto it = kernel_tensor.begin(); it < kernel_tensor.end(); ++it)
                {
                    *it = initializer_lambda();
                }
            }

            constexpr Conv()
            :
            comptype{CONV},
            kernel_tensor{{3, 3, 3}},
            stride{3UL}
            {
                short i = 0;
                for (auto it = kernel_tensor.begin(); it < kernel_tensor.end(); ++it)
                {
                    *it = default_filters[i++];
                }
            }

            Conv (const Conv& c) : kernel_tensor {c.kernel_tensor}, comptype {c.comptype} {}
            Conv (Conv&& c) : kernel_tensor {std::move(c.kernel_tensor)}, comptype {c.comptype} {}

            ~Conv ()
            {
                ~kernel_tensor();
            }


            ublas::tensor<NUM_TYPE>& apply(ublas::tensor<NUM_TYPE>& in_tensor) const noexcept;
            ublas::tensor<NUM_TYPE>& update(ublas::tensor<NUM_TYPE>& gradient_tensor) noexcept;
            

    };
}

#endif