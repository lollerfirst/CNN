#ifndef _CNN_HPP
#define _CNN_HPP

#include <dense.hpp>
#include <conv.hpp>
#include <maxpool.hpp>
#include <array>

namespace cnn
{

    /**
     * @brief Convolutional Neural Network wrapper
     * 
     */
    template <typename C, typename ... Args>
    class Cnn
    {
        private:
            std::array<C, sizeof...(Args) + 1> pipeline;
        
        public:

            
    };
}

#endif