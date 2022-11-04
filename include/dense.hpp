#ifndef _DENSE_HPP
#define _DENSE_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <functional>
#include <type_traits>
#include <constraints.hpp>
#include <component.hpp>

namespace cnn
{


    /**
     * @brief Class for the fully connected layer
     * 
     * @tparam NUM_TYPE type for weights matrix and bias vector
     */
    template<Numeric NUM_TYPE>
    class Dense : virtual Component<NUM_TYPE>
    {
        private:

            using namespace boost::numeric;

            ublas::matrix<NUM_TYPE> weight_matrix;
            ublas::vector<NUM_TYPE> bias_vector;
            double dropout;
        
        public:
            
            /**
             * @brief Constructs a Dense layer given the dimensions and dropout rate
             * 
             * @param from dimension of the incoming vector
             * @param to dimension of the outgoing vector
             * @param dout dropout rate
             */
            constexpr Dense(std::size_t from, std::size_t to, double dout = 0.2f) :
            weight_matrix{from, to},
            bias_vector{to},
            dropout{dout} {}

            /**
             * @brief Constructs a Dense layer given dimensions, initializer lambdas and dropout rate
             *  
             * @tparam Fn deducted type of the callable passed in as an initializer
             *  
             * @param from dimension of the incoming vector
             * @param to dimension of the outgoing vector
             * @param dropout dropout rate
             */
            template <typename Fn>
            requires Initializer<Fn, NUM_TYPE>
            constexpr Dense (std::size_t from, std::size_t to, Fn w_initializer, Fn b_initializer, double dropout = 0.2f)
                : Dense{from, to, dropout}
                {
                    // #pragma omp parallel
                    for (auto i = weight_matrix.begin1(); i <= weight_matrix.end1(); ++i)
                    {
                        for (auto j = weight_matrix.begin2(); j <= weight_matrix.end2(); ++j)
                        {
                            weight_matrix(i, j) = w_initializer();
                        }

                        bias_vector[i] = b_initializer();
                    }
                }


            ~Dense() = default;

            Dense(const Dense& d) : weight_matrix{d.weight_matrix}, bias_vector{d.bias_vector} {}
            Dense(Dense&& d) : weight_matrix{std::move(d.weight_matrix)}, bias_vector{std::move(d.bias_vector)} {} 

            /**
             * @brief Applies weight matrix and bias vector
             * 
             * @param in_vector incoming From-dimensional vector
             * @return ublas::vector<NUM_TYPE>&& transformed output (network representation)
             */
            ublas::vector<NUM_TYPE> apply(const ublas::vector<NUM_TYPE>& in_vector) const override;
            

            /**
             * @brief Updates the weight matrix and bias vector
             * 
             * @param gradient_vector vector containing the error from which calculate the error for each weight
             * @return ublas::vector<NUM_TYPE> Returns computed gradients
             */
            ublas::vector<NUM_TYPE> update(const ublas::vector<NUM_TYPE>& gradient_vector) override;
    };
}


#endif 