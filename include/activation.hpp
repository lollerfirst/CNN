#ifndef _ACTIVATION_HPP
#define _ACTIVATION_HPP

#include <component.hpp>
#include <constraints.hpp>

namespace cnn
{

    using namespace boost::numeric;

    /**
     * @brief activation type for conditional compilation
     *
     * Links to different implemented methods (relu or softmax)
     */
    typedef enum
    {
        RELU,
        SOFTMAX

    } ActivType;

    /**
     * @brief Wrapper for the activation functions
     * Acts as a component of the cnn that can be applied and updated getting a gradient from it
     * 
     * @tparam NUM_TYPE
     * @tparam A_TYPE Type of the activation
     */
    template <Numeric NUM_TYPE, ActivType A_TYPE>
    class Activation : Component
    {
            
        public:
            constexpr Activation() : comptype{ACTIVATION} {}
            ~Activation(){}

            ublas::tensor<NUM_TYPE>& apply(const ublas::tensor<NUM_TYPE>& in_tensor) const noexcept;
            ublas::tensor<NUM_TYPE>& update(const ublas::tensor<NUM_TYPE>& gradient_tensor) noexcept;
            
            ublas::vector<NUM_TYPE>& apply(const ublas::vector<NUM_TYPE>& in_vector) const noexcept;
            ublas::vector<NUM_TYPE>& update(const ublas::vector<NUM_TYPE>& gradient_vector) noexcept;
    };
    
}

#endif