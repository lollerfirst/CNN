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
    
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(argc, argv);
    RUN_ALL_TESTS();
}