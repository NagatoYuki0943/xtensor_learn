#include <iostream>
#include "main.h"
#include "xt_test.hpp"


int main(int argc, char* argv[]) {
    std::cout << "argc: " << argc << std::endl;
    std::cout << "program name: " << argv[0] << std::endl;

    xt::print_options::set_precision(8);         // 8位小数
    xt::print_options::set_line_width(90);       // 每行最多显示90个字符
    // xt::print_options::set_threshold(1000);   // 数组元素超过1000个时显示省略号
    // xt::print_options::set_edge_items(3);     // 显示头尾各3个元素

    std::cout << "---------------------------------------- xtensor test ----------------------------------------" << std::endl;
    xt_test::basic();
    xt_test::size_shape();
    xt_test::min_max();
    xt_test::calc();
    xt_test::compare();
    xt_test::filter();
    xt_test::filter_speed_test1();
    xt_test::row_col();
    xt_test::get_item();
    xt_test::view();
    xt_test::index_view();
    xt_test::test_get_value_by_indices();
    xt_test::reshape_view();
    xt_test::transpose();
    xt_test::squeeze_expand_dims();
    xt_test::concatenate();
    xt_test::stack();
    xt_test::sort();
    xt_test::where();
    xt_test::quantile();
    xt_test::eval();
    xt_test::deep_copy();
    xt_test::meshgrid();
    xt_test::triu_tril();
    xt_test::matrix_dot();
    std::cout << "---------------------------------------- xtensor test ----------------------------------------\n\n" << std::endl;

    return 0;
}
