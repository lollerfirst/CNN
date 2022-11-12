#include <iostream>
#include <gtest/gtest.h>
#include "../include/component.hpp"
#include "../include/dense.hpp"
#include "../include/activation.hpp"
#include "../include/conv.hpp"
#include "../include/flatten.hpp"
#include "../include/maxpool.hpp"

TEST(ComponentTest, TestCompType)
{
    cnn::Component<double>* c;

    cnn::Component<double> c0{};
    ASSERT((c0->comptype == cnn::DEFAULT));

    cnn::Dense<double> d{10, 3};
    c = &d;
    ASSERT((c->comptype == cnn::DENSE));

    cnn::Activation<double, cnn::RELU> a{};
    c = &a;
    ASSERT((c->comptype == cnn::ACTIVATION));
    
    cnn::Conv<double> cnv{};
    c = &cnv;
    ASSERT((c->comptype == cnn::CONV));

    cnn::Flatten<double> flt{};
    c = &flt;
    ASSERT((c->comptype == cnn::FLATTEN));

    cnn::MaxPool<double> mxpl{};
    c = &mxpl;
    ASSERT((c->comptype == cnn::MAXPOOL));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(argc, argv);
    RUN_ALL_TESTS();
}