#ifndef _CNN_HPP
#define _CNN_HPP


#include <array>
#include <component.hpp>
#include <constraints.hpp>

namespace cnn
{

    /**
     * @brief Convolutional Neural Network wrapper
     * 
     */
    template <Component*... Args>
    class CNN
    {
        private:
            std::array<Component*, sizeof...(Args)> pipeline;
            
        
        public:

            constexpr CNN(Args... list) : pipeline{list} {}

            ~CNN()
            {
                ~pipeline();
            }
            
            template <Numeric NUM_TYPE>
            int train(const ublas::tensor<NUM_TYPE>& train_set, std::size_t batch_size) noexcept;

            template <Numeric NUM_TYPE>
            int train(const ublas::matrix<NUM_TYPE>& train_set, std::size_t batch_size) noexcept;

            template <Numeric NUM_TYPE>
            int test(const ublas::tensor<NUM_TYPE>& test_set, std::size_t batch_size) noexcept;

            template <Numeric NUM_TYPE>
            int test(const ublas::matrix<NUM_TYPE>& test_set, std::size_t batch_size) noexcept;
    };
}

#endif