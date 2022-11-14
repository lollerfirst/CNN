#include <activation.hpp>
#include <contraints.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>

#define APPLY_SIGNATURE(DAT_TYPE, NUM_TYPE, ACT_TYPE) \
auto& cnn::Activation<NUM_TYPE, ACT_TYPE>::apply(DAT_TYPE& in_tensor) noexcept const -> DAT_TYPE

using namespace boost::numeric;

template <typename DAT_TYPE, Numeric NUM_TYPE, ActivType ACT_TYPE>
requires DataShape<DAT_TYPE, NUM_TYPE>
auto& cnn::Activation<NUM_TYPE, ACT_TYPE>::apply(DAT_TYPE& in_tensor) noexcept const -> DAT_TYPE 
{

    if constexpr(ACT_TYPE == cnn::RELU)
    {
        for (auto iter = in_tensor.begin(); iter < in_tensor.end(); ++iter)
        {
            // f(x) = max(0, x);
            *iter = std::max(static_cast<NUM_TYPE>(0), *iter); 
        }
    }
    else if constexpr(ACT_TYPE == cnn::SOFTMAX)
    {
        auto max = std::max_element(in_tensor.begin(), in_tensor.end());

        auto sum = std::accumulate(in_tensor.begin(), in_tensor.end(), static_cast<NUM_TYPE>(0), [max](const auto& a, const auto& b) {
            return (a + std::exp(b - max));
        });

        auto constant = max + std::log(sum);
        std::for_each(in_tensor.begin(), in_tensor.last(), [constant](auto& a){
            a -= constant;
        });
    }

    //RVO COPY-ELISION
    return in_tensor; 
}

template APPLY_SIGNATURE(ublas::vector<double>, double, cnn::RELU);
template APPLY_SIGNATURE(ublas::vector<double>, double, cnn::SOFTMAX);
template APPLY_SIGNATURE(ublas::tensor<double>, double, cnn::RELU);
template APPLY_SIGNATURE(ublas::tensor<double>, double, cnn::SOFTMAX);

template APPLY_SIGNATURE(ublas::vector<float>, float, cnn::RELU);
template APPLY_SIGNATURE(ublas::vector<float>, float, cnn::SOFTMAX);
template APPLY_SIGNATURE(ublas::tensor<float>, float, cnn::RELU);
template APPLY_SIGNATURE(ublas::tensor<float>, float, cnn::SOFTMAX);

template APPLY_SIGNATURE(ublas::vector<long double>, long double, cnn::RELU);
template APPLY_SIGNATURE(ublas::vector<long double>, long double, cnn::SOFTMAX);
template APPLY_SIGNATURE(ublas::tensor<long double>, long double, cnn::RELU);
template APPLY_SIGNATURE(ublas::tensor<long double>, long double, cnn::SOFTMAX);