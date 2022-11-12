#ifndef _COMPONENT_HPP
#define _COMPONENT_HPP

#include <constraints.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace cnn
{
    typedef enum 
    {
        DEFAULT,
        DENSE,
        ACTIVATION,
        CONV,
        MAXPOOL,
        FLATTEN
    } comp_types;

    using namespace boost::numeric;

    template <Numeric NUM_TYPE>
    class Component
    {
        public:
            comp_types comptype;

            constexpr Component() : comptype{DEFAULT} {}

            virtual ublas::vector<NUM_TYPE> apply(const ublas::vector<NUM_TYPE>& in_vector) const = 0;
            virtual ublas::tensor<NUM_TYPE> apply(const ublas::tensor<NUM_TYPE>& in_tensor) const = 0;

            virtual ublas::vector<NUM_TYPE> update(const ublas::vector<NUM_TYPE>& gradient_vector) = 0;
            virtual ublas::tensor<NUM_TYPE> update(const ublas::tensor<NUM_TYPE>& gradient_tensor) = 0;

            virtual ublas::vector<NUM_TYPE> apply(const ublas::tensor<NUM_TYPE>& in_tensor) const = 0;
            virtual ublas::vector<NUM_TYPE> update(const ublas::vector<NUM_TYPE>& gradient_vector) = 0;

            virtual ~Component() = 0;
    };
}

#endif