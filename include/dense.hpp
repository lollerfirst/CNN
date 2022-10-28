#include <Eigen/Dense>

namespace cnn
{
    template<int From_Vector, int To_Vector, typename Int_Type = long int>
    class Dense{
        private:
            using type = Int_Type;
            Eigen::Matrix<type, From_Vector, To_Vector> weight_matrix;
            Eigen::Matrix<type, From_Vector, To_Vector> bias_matrix;
        
        public:
            
            Dense() : weight_matrix{}, bias_matrix{};

            Dense();
            
            ~Dense() = default;

            Dense(const Dense&) = delete;
            Dense(Dense&&) = delete;

            void apply(const Eigen::Matrix<type, From_Vector, 1>& in_vector, Eigen::Matrix<type, To_Vector, 1>&) const;

    };

}