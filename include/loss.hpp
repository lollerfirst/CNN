#ifndef _LOSS_HPP
#define _LOSS_HPP

#include <constraints.hpp>
#include <array>

namespace cnn
{
    using namespace boost::numeric;

    typedef enum
    {
        MEAN_SQUARED,
        CROSS_ENTROPY
    } LossType;

    template <Numeric NUM_TYPE, LossType LOSS_TYPE, std::size_t BATCH_SIZE>
    class Loss
    {
        private:
            std::array<NUM_TYPE, BATCH_SIZE> backlog;
            std::size_t last;

        public:

            constexpr Loss() : backlog{}, last{0UL} {}

            ~Loss()
            {
                ~backlog();
            }

            auto calculate(const ublas::vector<NUM_TYPE>& prediction) -> NUM_TYPE;
            auto report() const -> NUM_TYPE;
    };

}

#endif