#ifndef _ACTIVATION_HPP
#define _ACTIVATION_HPP

#include <component.hpp>
#include <constraints.hpp>

namespace cnn
{

    using namespace boost::numeric;

    typedef enum
    {
        RELU,
        SOFTMAX

    } ActivType;

    template <ActivType A_TYPE, Numeric NUM_TYPE>
    class Activation : virtual Component<NUM_TYPE>
    {
            
        public:
            constexpr Activation(){}
            ~Activation(){}

            ublas::tensor<NUM_TYPE> apply(const ublas::tensor<NUM_TYPE>& in_tensor) const override;
            ublas::tensor<NUM_TYPE> update(const ublas::tensor<NUM_TYPE>& gradient_tensor) override;
            
            ublas::vector<NUM_TYPE> apply(const ublas::vector<NUM_TYPE>& in_vector) const override;
            ublas::vector<NUM_TYPE> update(const ublas::vector<NUM_TYPE>& gradient_vector) override;
    };
    
}

#endif