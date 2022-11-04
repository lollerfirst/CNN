#ifndef _COMPONENT_HPP
#define _COMPONENT_HPP

#include <constraints.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace cnn
{
    template <Numeric NUM_TYPE>
    class Component
    {
        private: 
            using namespace boost::numeric;

        public:
            virtual ublas::vector<NUM_TYPE> apply(const ublas::vector<NUM_TYPE>& in_vector) const = 0;
            virtual ublas::vector<NUM_TYPE> update(const ublas::vector<NUM_TYPE>& out_vector) const = 0;
    };
}

#endif