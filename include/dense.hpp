#ifndef _DENSE_HPP
#define _DENSE_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <functional>
#include <type_traits>

namespace cnn
{
    
    template<typename INT_TYPE>
    class Dense
    {
        private:
            template <typename Fn>
            concept Initializer = std::is_callable_v<Fn> 
                                    && std::same_as_v<INT_TYPE, std::result_of_t<Fn(void)>>;

            using namespace boost::numeric;

            ublas::matrix<INT_TYPE> weight_matrix;
            ublas::vector<INT_TYPE> bias_vector;

            double dropout;
        
        public:
            
            constexpr Dense(std::size_t from, std::size_t to, double dout = 0.2f) :
            weight_matrix{from, to},
            bias_vector{to},
            dropout{dout} {}

            template <Initializer Fn>
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

            Dense(const Dense&) = delete;
            Dense(Dense&&) = delete;

            ublas::vector<INT_TYPE>&& apply(const ublas::vector<INT_TYPE>& in_vector) const;

    };
}


#endif 