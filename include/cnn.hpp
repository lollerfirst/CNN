#ifndef _CNN_HPP
#define _CNN_HPP

#include <dense.hpp>
#include <conv.hpp>
#include <maxpool.hpp>
#include <array>
#include <component.hpp>

namespace cnn
{

    /**
     * @brief Convolutional Neural Network wrapper
     * 
     */
    template <Component ... Args>
    class CNN
    {
        private:
            std::array<Component, sizeof...(Args)> pipeline;
        
        public:

            constexpr CNN(const Component& ... list) : pipeline{list} {}
            
    };
}

#endif