#include <iostream>
#include <gtest/gtest.h>
#include "../include/component.hpp"
#include "../include/dense.hpp"
#include "../include/activation.hpp"
#include "../include/conv.hpp"
#include "../include/flatten.hpp"
#include "../include/maxpool.hpp"
#include "../include/loss.hpp"
#include "../include/initializers.hpp"
#include <cstdlib>
#include <ctime>

using namespace boost::numeric;


class ActivationTest : testing::Test{
    protected:
        void SetUp() override
        {
            dtensor = ublas::tensor<double>{3,3,3};
            dvector = ublas::vector<double>{27};
            itensor = ublas::tensor<long>{3,3,3};
            ivector = ublas::vector<long>{27};

            std::for_each(dtensor.begin(), dtensor.end(), [](auto& a){
                a = default_initializer<double>();
            });

            std::for_each(dvector.begin(), dvector.end(), [](auto& a){
                a = default_initializer<double>();
            });

            std::for_each(itensor.begin(), itensor.end(), [](auto& a){
                a = default_initializer<long>();
            });

            std::for_each(ivector.begin(), ivector.end(), [](auto& a){
                a = default_initializer<long>();
            });
        }

        void TearDown() override
        {
            ~ActivationTest();
        }

        ublas::tensor<double> dtensor;
        ublas::vector<double> dvector;
        ublas::tensor<long> itensor;
        ublas::vector<long> ivector; 
} 

TEST_F(ActivationTest, ApplyTest){
    
    cnn::Activation<double, cnn::RELU> activation1;
    dtensor = activation1.apply(dtensor);
    
    cnn::Activation<double, cnn::SOFTMAX> activation2;
    dvector = activation2.apply(dtensor);

    cnn::Activation<long, cnn::RELU> activation3;
    itensor = activation3.apply(itensor);

    cnn::Activation<long, cnn::SOFTMAX> activation4;
    ivector = activation4.apply(ivector);
}

TEST_F(ActivationTest, UpdateTest){
    
}

TEST(ComponentTest, TestCompType)
{
    
}

TEST(CnnTest, TestConctructor){

}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(argc, argv);
    RUN_ALL_TESTS();
}