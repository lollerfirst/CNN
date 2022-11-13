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

    class Component
    {
        public:
            comp_types comptype;

            constexpr Component() : comptype{DEFAULT} {}
    };
}

#endif