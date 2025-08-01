#pragma once
#ifndef XT_TEST_HPP
#define XT_TEST_HPP

#include <tuple>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <chrono>

namespace xt_test
{

    template <typename T>
    static xt::xarray<T> add(xt::xarray<T> &a, xt::xarray<T> &b)
    {
        return a + b;
    }

    void basic()
    {
        std::cout << "basic start" << std::endl;

        xt::xarray<double> arr0 = xt::xarray<double>::from_shape({0});
        std::cout << "arr0: " << arr0 << std::endl;
        arr0 = xt::xarray<double>::from_shape({0, 0});
        std::cout << "arr0: " << arr0 << std::endl;
        arr0 = xt::xarray<double>::from_shape({2, 3});
        std::cout << "arr0: " << arr0 << std::endl;
        // 使用 auto 声明类型后, 不能被修改, 要显示声明类型才能修改
        xt::xarray<double> arr1 = xt::empty<double>({2, 3});
        std::cout << "arr1: " << arr1 << std::endl;
        xt::xarray<double> arr2 = xt::zeros<double>({1, 1});
        arr2(0, 0) = 1.0;
        std::cout << "arr2: " << arr2 << std::endl;
        xt::xarray<double> arr3 = xt::ones<double>({2, 3});
        arr3(0, 0) = 2.0;
        std::cout << "arr3: " << arr3 << std::endl;
        xt::xarray<double> arr4 = xt::eye<double>(3);
        std::cout << "arr4: " << arr4 << std::endl;
        xt::xarray<double> arr5 = xt::arange<double>(10);
        std::cout << "arr5: " << arr5 << std::endl;
        xt::xarray<double> arr6 = xt::linspace<double>(0, 1, 11);
        std::cout << "arr6: " << arr6 << std::endl;
        xt::xarray<double> arr7 = xt::logspace<double>(0, 1, 11);
        std::cout << "arr7: " << arr7 << std::endl;
        // random 要用 xt::eval, 防止惰性计算，导致每次使用时都重新生成
        xt::xarray<double> arr8 = xt::eval(xt::random::randn<double>({2, 3}));
        std::cout << "arr8: " << arr8 << std::endl;
        xt::xarray<double> arr9 = xt::eval(xt::random::rand<double>({2, 3}));
        std::cout << "arr9: " << arr9 << std::endl;
        xt::xarray<int> arr10 = xt::eval(xt::random::randint<int>({2, 3}, 0, 10));
        std::cout << "arr10: " << arr10 << std::endl;
        xt::xarray<double> arr11 = xt::cast<double>(arr10);
        std::cout << "arr11: " << arr11 << std::endl;
        xt::xarray<double> arr12 = xt::arange(10);
        std::cout << "arr12: " << arr12 << std::endl;
        xt::xarray<double> arr13 = xt::arange(10, 20);
        std::cout << "arr13: " << arr13 << std::endl;
        xt::xarray<double> arr14 = xt::arange(10, 20, 2);
        std::cout << "arr14: " << arr14 << std::endl;
        // 使用 int 生成 bool
        xt::xarray<bool> arr15 = xt::zeros<int>({2, 3});
        std::cout << "arr15: " << arr15 << std::endl;
        xt::xarray<bool> arr16 = xt::ones<int>({2, 3});
        std::cout << "arr16: " << arr16 << std::endl;

        std::cout << "basic end\n"
                  << std::endl;
    }

    void size_shape()
    {
        std::cout << "size_shape start" << std::endl;

        xt::xarray<double> arr1 = xt::xarray<double>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
        std::cout << "arr1:\n"
                  << arr1 << std::endl;
        //{{ 1.,  2.,  3.},
        // { 4.,  5.,  6.}}

        std::cout << "arr1 size: " << arr1.size() << std::endl;
        // arr1 size: 6

        std::cout << "arr1 shape: (" << arr1.shape(0) << ", " << arr1.shape(1) << ")" << std::endl;
        // arr1 shape: (2, 3)

        std::cout << "arr1 dimension: " << arr1.dimension() << std::endl;
        // arr1 dimension: 2

        std::cout << "size_shape end\n"
                  << std::endl;
    }

    void min_max()
    {
        std::cout << "min_max start" << std::endl;

        auto x = xt::eval(xt::random::randn<double>({2, 3}));
        std::cout << "x: " << x << std::endl;
        // 不指定轴, 计算所有元素的最大值...
        std::cout << "x amax: " << xt::amax(x) << " amin: " << xt::amin(x)
                  << " mean: " << xt::mean(x) << " average: " << xt::average(x)
                  << " variance: " << xt::variance(x) << " stddev: " << xt::stddev(x)
                  << " sum: " << xt::sum(x) << " prod: " << xt::prod(x) << std::endl;

        std::cout << "maximum: " << xt::maximum(xt::xarray<double>({1, 2, 3}), xt::xarray<double>({2, 3, 4})) << std::endl;
        // maximum: { 2.,  3.,  4.}

        std::cout << "minimum: " << xt::minimum(xt::xarray<double>({1, 2, 3}), xt::xarray<double>({2, 3, 4})) << std::endl;
        // minimum: { 1.,  2.,  3.}

        std::cout << "minimum: " << xt::minimum(xt::xarray<double>({1, 2, 3}), xt::xarray<double>({2})) << std::endl;
        // minimum: { 1.,  2.,  2.}

        // 指定轴
        std::cout << "arr17 amax: " << xt::amax(x) << " amax(0): " << xt::amax(x, {0}) << " amax(1): " << xt::amax(x, {1}) << std::endl;

        std::cout << "min_max end\n"
                  << std::endl;
    }

    void calc()
    {
        std::cout << "calc start" << std::endl;

        xt::xarray<double> x{
            {1.0, 2.0, 3.0},
            {2.0, 5.0, 7.0},
            {2.0, 5.0, 7.0}};
        x = x + 1.0;

        xt::xarray<double> z = add(x, x);
        std::cout << "z: \n"
                  << z << std::endl;
        //{{  4.,   6.,   8.},
        // {  6.,  12.,  16.},
        // {  6.,  12.,  16.}}

        x = {1, 2, 3};
        xt::xarray<double> y = {4, 5, 6};
        z = add(x, y);
        std::cout << "z: \n"
                  << z << std::endl;
        // { 5.,  7.,  9.}

        x = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}};
        y = {-1, -2};
        std::cout << "y: \n"
                  << y << std::endl;
        // {-1., -2.}

        z = add(x, y);
        std::cout << "z: \n"
                  << z << std::endl;
        //{{ 0.,  0.},
        // { 2.,  2.},
        // { 4.,  4.}}

        y = xt::reshape_view(y, {-1, 2});
        std::cout << "y: \n"
                  << y << std::endl;
        // {{-1., -2.}}

        z = add(x, y);
        std::cout << "z: \n"
                  << z << std::endl;
        //{{ 0.,  0.},
        // { 2.,  2.},
        // { 4.,  4.}}

        //{{ 1.,  2.},
        // { 3.,  4.},
        // { 5.,  6.}}

        std::cout << "square:\n"
                  << xt::square(x) << std::endl;
        //{{  1.,   4.},
        // {  9.,  16.},
        // { 25.,  36.}}

        std::cout << "pow 2:\n"
                  << xt::pow(x, 2) << std::endl;
        //{{  1.,   4.},
        // {  9.,  16.},
        // { 25.,  36.}}

        std::cout << "pow 3:\n"
                  << xt::pow(x, 3) << std::endl;
        //{{   1.,    8.},
        // {  27.,   64.},
        // { 125.,  216.}}

        // pow 3 等价于 cube
        std::cout << "cube:\n"
                  << xt::cube(x) << std::endl;
        //{{   1.,    8.},
        // {  27.,   64.},
        // { 125.,  216.}}

        std::cout << "pow 0.5:\n"
                  << xt::pow(x, 0.5) << std::endl;
        //{{ 1.        ,  1.41421356},
        // { 1.73205081,  2.        },
        // { 2.23606798,  2.44948974}}

        std::cout << "sqrt:\n"
                  << xt::sqrt(x) << std::endl;
        //{{ 1.        ,  1.41421356},
        // { 1.73205081,  2.        },
        // { 2.23606798,  2.44948974}}

        std::cout << "exp:\n"
                  << xt::exp(x) << std::endl;
        //{{   2.71828183,    7.3890561 },
        // {  20.08553692,   54.59815003},
        // { 148.4131591 ,  403.42879349}}

        std::cout << "exp2:\n"
                  << xt::exp2(x) << std::endl;
        //{{  2.,   4.},
        // {  8.,  16.},
        // { 32.,  64.}}

        std::cout << "log:\n"
                  << xt::log(x) << std::endl;
        //{{ 0.        ,  0.69314718},
        // { 1.09861229,  1.38629436},
        // { 1.60943791,  1.79175947}}

        std::cout << "log2:\n"
                  << xt::log2(x) << std::endl;
        //{{ 0.        ,  1.        },
        // { 1.5849625 ,  2.        },
        // { 2.32192809,  2.5849625 }}

        std::cout << "log10:\n"
                  << xt::log10(x) << std::endl;
        //{{ 0.        ,  0.30103   },
        // { 0.47712125,  0.60205999},
        // { 0.69897   ,  0.77815125}}

        std::cout << "sin:\n"
                  << xt::sin(x) << std::endl;
        //{{ 0.84147098,  0.90929743},
        // { 0.14112001, -0.7568025 },
        // {-0.95892427, -0.2794155 }}

        std::cout << "cos:\n"
                  << xt::cos(x) << std::endl;
        //{{ 0.54030231, -0.41614684},
        // {-0.9899925 , -0.65364362},
        // { 0.28366219,  0.96017029}}

        std::cout << "tan:\n"
                  << xt::tan(x) << std::endl;
        //{{ 1.55740772, -2.18503986},
        // {-0.14254654,  1.15782128},
        // {-3.38051501, -0.29100619}}

        std::cout << "sign:\n"
                  << xt::sign(x) << std::endl;
        //{{ 1.,  1.},
        // { 1.,  1.},
        // { 1.,  1.}}

        x = {
            {1.4, 2.4},
            {3.5, 4.5},
            {5.5, 6.6}};

        std::cout << "floor:\n"
                  << xt::floor(x) << std::endl;
        //{{ 1.,  2.},
        // { 3.,  4.},
        // { 5.,  6.}}

        std::cout << "ceil:\n"
                  << xt::ceil(x) << std::endl;
        //{{ 2.,  3.},
        // { 4.,  5.},
        // { 6.,  7.}}

        std::cout << "round:\n"
                  << xt::round(x) << std::endl;
        //{{ 1.,  2.},
        // { 4.,  5.},
        // { 6.,  7.}}

        std::cout << "round:\n"
                  << xt::round(xt::xarray<double>{0.5, 1.5, 2.5, 3.5, 4.5, 5.5}) << std::endl;
        // { 1.,  2.,  3.,  4.,  5.,  6.}

        auto cumsum_res = xt::cumsum(x);
        auto cumprod_res = xt::cumprod(x);
        std::cout << "x cumsum:\n"
                  << cumsum_res << std::endl;
        // {  1.4,   3.8,   7.3,  11.8,  17.3,  23.9}

        std::cout << "x cumprod:\n"
                  << cumprod_res << std::endl;
        // {    1.4   ,     3.36  ,    11.76  ,    52.92  ,   291.06  ,  1920.996 }

        std::cout << "calc end\n"
                  << std::endl;
    }

    void compare()
    {
        std::cout << "compare start" << std::endl;
        xt::xarray<double> x1 = {1., 2., 3., 4., 5.};
        xt::xarray<double> x2 = {2., 3., 4., 5., 6.};
        std::cout << "x1: " << x1 << std::endl;
        std::cout << "x2: " << x2 << std::endl;
        // 直接打印要加上括号
        std::cout << "equal: " << (x1 == x2) << std::endl;
        // equal: 0

        auto cmp1 = x1 == x2;
        std::cout << "equal: " << cmp1 << std::endl;
        // equal: 0

        cmp1 = xt::operator==(x1, x2);
        std::cout << "equal: " << cmp1 << std::endl;
        // equal: 0

        // 每个值单独比较
        std::cout << "xt::equal: " << xt::equal(x1, x2) << std::endl;
        // xt::equal: {false, false, false, false, false}

        auto cmp2 = x1 != x2;
        std::cout << "not equal: " << cmp2 << std::endl;
        // not equal: 1

        cmp2 = xt::operator!=(x1, x2);
        std::cout << "not equal: " << cmp2 << std::endl;
        // not equal: 1

        // 每个值单独比较
        std::cout << "xt::not_equal: " << xt::not_equal(x1, x2) << std::endl;
        // xt::not_equal: { true,  true,  true,  true,  true}

        auto cmp3 = x1 < x2;
        std::cout << "less: " << cmp3 << std::endl;
        // less: { true,  true,  true,  true,  true }

        auto cmp3_ = xt::operator<(x1, x2);
        std::cout << "less: " << cmp3_ << std::endl;
        // less: { true,  true,  true,  true,  true }

        std::cout << "xt::less: " << xt::less(x1, x2) << std::endl;
        // xt::less: { true,  true,  true,  true,  true}

        auto cmp4 = x1 <= x2;
        std::cout << "less equal: " << cmp4 << std::endl;
        // less equal: { true,  true,  true,  true,  true }

        auto cmp4_ = xt::operator<=(x1, x2);
        std::cout << "less equal: " << cmp4_ << std::endl;
        // less equal: { true,  true,  true,  true,  true }

        std::cout << "xt::less_equal: " << xt::less_equal(x1, x2) << std::endl;
        // xt::less_equal: { true,  true,  true,  true,  true}

        auto cmp5 = x1 > x2;
        std::cout << "greater: " << cmp5 << std::endl;
        // greater: { false, false, false, false, false }

        auto cmp5_ = xt::operator>(x1, x2);
        std::cout << "greater: " << cmp5_ << std::endl;
        // greater: { false, false, false, false, false }

        std::cout << "xt::greater: " << xt::greater(x1, x2) << std::endl;
        // xt::greater: {false, false, false, false, false}

        // 下面时对比单个数字

        std::cout << "x1 >= 3: " << (x1 >= 3) << std::endl;
        // x1 >= 3: {false, false,  true,  true,  true}

        std::cout << "x1 < 3: " << (x1 < 3) << std::endl;
        // x1 < 3: { true,  true, false, false, false}

        std::cout << "xt::equal to 1: " << xt::equal(x1, 1) << std::endl;
        // xt::equal to 1: { true, false, false, false, false}

        std::cout << "xt::not_equal to 1: " << xt::not_equal(x1, 1) << std::endl;
        // xt::not_equal to 1 : {false, true, true, true, true}

        // && (逻辑与)
        x1 = {0, 1., 0., 1., 2.};
        x2 = {0, 0., 1., 1., 1.};
        auto cmp6 = x1 && x2;
        std::cout << "x1 && x2 = " << cmp6 << std::endl;
        // x1 && x2 = {false, false, false,  true,  true}
        auto cmp6_ = xt::operator&&(x1, x2);
        std::cout << "xt::operator&&(x1, x2) = " << cmp6_ << std::endl;
        // xt::operator&&(x1, x2) = {false, false, false,  true,  true}

        // || (逻辑或)
        auto cmp7 = x1 || x2;
        std::cout << "x1 || x2 = " << cmp7 << std::endl;
        // x1 || x2 = {false,  true,  true,  true,  true}
        auto cmp7_ = xt::operator||(x1, x2);
        std::cout << "xt::operator||(x1, x2) = " << cmp7_ << std::endl;
        // xt::operator||(x1, x2) = {false,  true,  true,  true,  true}

        x1 = {1, -2, 3};
        x2 = {0, -1, 2};
        xt::xarray<double> x3 = {0, 0, 0};

        std::cout << "x1 all = " << xt::all(x1) << std::endl;
        // x1 all = 1

        std::cout << "x1 any = " << xt::any(x1) << std::endl;
        // x1 any = 1

        std::cout << "x2 all = " << xt::all(x2) << std::endl;
        // x2 all = 0

        std::cout << "x2 any = " << xt::any(x2) << std::endl;
        // x2 any = 1

        std::cout << "x3 all = " << xt::all(x3) << std::endl;
        // x3 all = 0

        std::cout << "x3 any = " << xt::any(x3) << std::endl;
        // x3 any = 0

        std::cout << "compare end\n"
                  << std::endl;
    }

    void filter()
    {
        // filter 是用于过滤数组元素的函数，可以指定条件，返回满足条件的元素
        // 返回一个1D视图，其中选择了条件计算为真的元素。
        // 引用参数，会修改原始数据
        // 可以使用 eval 避免影响原始数据

        std::cout << "filter start" << std::endl;

        xt::xarray<double> x1 = {
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
        xt::xarray<bool> b = {
            false, true, false, true, false, true, false, true, false, true};
        std::cout << "1 dim filter:\n"
                  << xt::filter(x1, b) << std::endl;
        // { 2., 4., 6., 8., 10. }

        x1 = {
            {1., 2., 3., 4., 5.},
            {6., 7., 8., 9., 10.}};
        b = {
            {false, true, false, true, false},
            {true, false, true, false, true}};
        std::cout << "2 dim filter:\n"
                  << xt::filter(x1, b) << std::endl;
        // { 2., 4., 6., 8., 10. }

        // 过滤并修改
        xt::filter(x1, b) = -1;
        std::cout << "x1 after filter:\n"
                  << x1 << std::endl;
        // {{ 1., -1.,  3., -1.,  5.},
        //  {-1.,  7., -1.,  9., -1.}}

        // 使用 eval 避免影响原始数据
        xt::eval(xt::filter(x1, b)) = 100;
        std::cout << "x1 after filter:\n"
                  << x1 << std::endl;
        // {{ 1., -1.,  3., -1.,  5.},
        //  {-1.,  7., -1.,  9., -1.}}

        std::cout << "filter end\n"
                  << std::endl;
    }

    void filter_speed_test()
    {
        // 过滤速度测试
        std::cout << "filter speed test start" << std::endl;

        xt::xarray<double> arr = xt::arange(1000000);
        xt::xarray<bool> mask = xt::random::randint<int>({1000000}, 0, 2);

        auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto res1 = xt::filter(arr, mask);
        auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout << "xt::filter time: " << t2 - t1 << " ms" << std::endl;

        std::vector<double> res_vec;
        for (int i = 0; i < arr.shape()[0]; i++)
        {
            if (mask(i))
                res_vec.push_back(arr(i));
        }
        auto res2 = xt::adapt(res_vec);
        auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        std::cout << "for loop filter time: " << t3 - t2 << " ms" << std::endl;

        std::cout << "res1.size = " << res1.size() << std::endl;
        std::cout << "res2.szie = " << res2.size() << std::endl;
        std::cout << "res1 == res2: " << (res1 == res2) << std::endl;

        std::cout << "filter speed test end\n"
                  << std::endl;
    }

    void row_col()
    {
        std::cout << "row_col start" << std::endl;

        // row col 必须处理2维度数组
        xt::xarray<double> x0 = xt::reshape_view(xt::arange(6), {2, 3});
        std::cout << "x0: \n"
                  << x0 << std::endl;
        //{{ 0.,  1.,  2.},
        // { 3.,  4.,  5.}}

        // 取出第0行
        xt::xarray<double> row0 = xt::row(x0, 0);
        std::cout << "x0 row0: \n"
                  << row0 << std::endl;
        // { 0.,  1.,  2.}

        // 取出第0列
        xt::xarray<double> col0 = xt::col(x0, 0);
        std::cout << "x0 col0: \n"
                  << col0 << std::endl;
        // { 0.,  3.}

        std::cout << "row_col end\n"
                  << std::endl;
    }

    void get_item()
    {
        std::cout << "get_item start" << std::endl;
        xt::xarray<double> x = {1};
        std::cout << "x: " << x << std::endl;
        // x: { 1.}

        std::cout << "x[0]: " << x[0] << std::endl;
        // x[0]: 1

        std::cout << "x(0): " << x(0) << std::endl;
        // x(0): 1

        x = {{1, 2, 3},
             {4, 5, 6},
             {7, 8, 9}};
        std::cout << "x: " << x << std::endl;
        // x: {{ 1.,  2.,  3.},
        //     { 4.,  5.,  6.},
        //     { 7.,  8.,  9.}}

        std::cout << "x[0]: " << x[0] << std::endl;
        // x[0]: 1

        std::cout << "x(0): " << x(0) << std::endl;
        // x(0): 1

        std::cout << "x[0, 0]: " << x[0, 0] << std::endl;
        // x[0, 0]: 1

        std::cout << "x(0, 0): " << x(0, 0) << std::endl;
        // x(0, 0) : 1

        // 多维数据使用 () 而不是 []
        std::cout << "x[1, 1]: " << x[1, 1] << std::endl;
        // x[1, 1]: 2

        std::cout << "x(1, 1): " << x(1, 1) << std::endl;
        // x(1, 1) : 5

        // 多维数据使用 () 而不是 []
        std::cout << "x[2, 1]: " << x[2, 1] << std::endl;
        // x[2, 1] : 2

        std::cout << "x(2, 1): " << x(2, 1) << std::endl;
        // x(2, 1) : 8

        std::cout << "get_item end\n"
                  << std::endl;
    }

    void view()
    {
        // xt::view 是用于创建数组视图的函数，可以进行切片、索引等操作，而且不会创建新的内存
        // 需要独立副本时，使用 xt::eval() 或显式复制
        std::cout << "view start" << std::endl;

        // 1. 基本用法 - 切片
        xt::xarray<int> arr1 = {1, 2, 3, 4, 5, 6};
        std::cout << "Original array:\n"
                  << arr1 << std::endl;
        // {1, 2, 3, 4, 5, 6}

        // 不能使用 xt::xarray<int> 必须使用 xt::range
        // std::cout << "View of range(1, 4):\n" << xt::view(arr1, xt::xarray<int> { 1, 4 }) << std::endl;
        // {2, 3, 4}

        xt::xarray<int> view = xt::view(arr1, xt::range(1, 4));
        std::cout << "View of range(1, 4):\n"
                  << view << std::endl;
        // {2, 3, 4}

        // 2. 切片 - 步长
        view = xt::view(arr1, xt::range(0, 6, 2));
        std::cout << "View of range(0, 6, 2):\n"
                  << view << std::endl;
        // {1, 3, 5}

        // 3. 切片 - 负数索引
        view = xt::view(arr1, xt::range(-3, -1));
        std::cout << "View of range(-3, -1):\n"
                  << view << std::endl;
        // {4, 5}

        xt::xarray<int> arr2 = {{1, 2, 3},
                                {4, 5, 6},
                                {7, 8, 9}};
        std::cout << "Original array:\n"
                  << arr2 << std::endl;
        // {{1, 2, 3},
        //  {4, 5, 6},
        //  {7, 8, 9}}

        xt::xarray<int> view1 = xt::view(arr2, xt::all(), xt::range(1, 3));
        std::cout << "View of all(), range(1, 3):\n"
                  << view1 << std::endl;
        // {{2, 3},
        //  {5, 6},
        //  {8, 9}}

        view1 = xt::view(arr2, xt::all(), 1);
        std::cout << "View of all(), 1:\n"
                  << view1 << std::endl;
        // {2, 5, 8}

        view1 = xt::view(arr2, 1, xt::all());
        std::cout << "View of 1, all():\n"
                  << view1 << std::endl;
        // {4, 5, 6}

        auto view2 = xt::view(arr2, 1, 2);
        std::cout << "View of 1, 2:\n"
                  << view2 << std::endl;
        // 6

        // 修改视图会影响原始数据
        view2 = 10;
        std::cout << "Modified view1:\n"
                  << arr2 << std::endl;
        // {{1, 2, 3},
        //  {4, 5, 10},
        //  {7, 8, 9}}

        // 使用 eval 避免影响原始数据
        auto view3 = xt::eval(xt::view(arr2, 1, 2));
        view3 = 100;
        std::cout << "Modified view2:\n"
                  << arr2 << std::endl;
        // {{1, 2, 3},
        //  {4, 5, 10},
        //  {7, 8, 9}}

        std::cout << "view end\n"
                  << std::endl;
    }

    void index_view()
    {
        std::cout << "index_view start" << std::endl;

        xt::xarray<double> a = {
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
        // 这里可以使用 xt::xindex 代替 xt::xarray<int>, 但是只能是一维数组
        xt::xarray<int> b = {
            1, 3, 5, 7, 9};

        std::cout << xt::index_view(a, b) << std::endl;
        // { 2., 4., 6., 8., 10. }

        a = {{1, 2, 3},
             {4, 5, 6}};
        b = xt::index_view(a, {{0, 0}, {0, 1}, {1, 1}});
        std::cout << b << std::endl;
        // {1, 2, 5}

        b = {{0, 0}, {0, 1}, {1, 1}};
        // 注意返回的是一个一维数据, 是把 x1 和 b 都展平, 然后根据 b 的值取出对应位置的值
        std::cout << xt::index_view(a, b) << std::endl;
        // { 1.,  1.,  1.,  2.,  2.,  2.}

        // 这种写法可以避免取出有问题的数据
        std::vector<std::array<size_t, 2>> b1 = {{0, 0}, {0, 1}, {1, 1}};
        std::cout << xt::index_view(a, b1) << std::endl;
        // { 1.,  2.,  5.}

        std::cout << "index_view end\n"
                  << std::endl;
    }

    template <typename T>
    xt::xarray<T> get_xt_value_by_indices(const xt::xarray<T> &boxes, const xt::xarray<int> &indices)
    {
        int n = indices.shape(0);
        int w = boxes.shape(1);
        xt::xarray<T> result = xt::zeros<T>({n, w});
        for (int i = 0; i < n; ++i)
        {
            xt::view(result, i, xt::all()) = xt::view(boxes, indices(i), xt::all());
        }

        return result;
    }

    void test_get_value_by_indices()
    {
        std::cout << "test_get_value_by_indices start" << std::endl;

        xt::xarray<int> boxes = {
            {0, 0},
            {1, 1},
            {2, 2},
            {3, 3},
            {4, 4},
        };
        std::cout << "boxes:\n"
                  << boxes << std::endl;
        //{{0, 0},
        // {1, 1},
        // {2, 2},
        // {3, 3},
        // {4, 4}}

        xt::xarray<int> indices = {1, 3, 4, 2, 0};
        std::cout << "get boxes:\n"
                  << get_xt_value_by_indices(boxes, indices) << std::endl;
        //{{1, 1},
        // {3, 3},
        // {4, 4},
        // {2, 2},
        // {0, 0}}

        std::cout << "test_get_value_by_indices end\n"
                  << std::endl;
    }

    void reshape_view()
    {
        // reshape_view 是 xtensor 中用于重新调整数组形状的视图操作，它不会创建新的内存，而是提供一个新的视角来查看原始数据。
        // 需要独立副本时，使用 xt::eval() 或显式复制
        std::cout << "reshape_view start" << std::endl;

        // 1. 基本用法 - 1D到2D
        xt::xarray<int> arr1 = {1, 2, 3, 4, 5, 6};
        std::cout << "Original array:\n"
                  << arr1 << std::endl;
        // {1, 2, 3, 4, 5, 6}

        // 重塑为2x3数组
        auto reshaped1 = xt::reshape_view(arr1, {2, 3});
        std::cout << "Reshaped to 2x3:\n"
                  << reshaped1 << std::endl;
        /*
        {{1, 2, 3},
         {4, 5, 6}}
        */

        // 重塑为3x2数组
        auto reshaped2 = xt::reshape_view(arr1, {reshaped1.shape(1), reshaped1.shape(0)});
        std::cout << "Reshaped to 3x2:\n"
                  << reshaped2 << std::endl;
        /*
        {{1, 2},
         {3, 4},
         {5, 6}}
        */

        // 2. 2D到1D
        xt::xarray<int> arr2 = {{1, 2, 3},
                                {4, 5, 6}};
        auto flattened = xt::reshape_view(arr2, {6});
        std::cout << "Flattened array:\n"
                  << flattened << std::endl;
        // {1, 2, 3, 4, 5, 6}
        flattened = xt::reshape_view(arr2, {arr2.size()});
        std::cout << "Flattened array:\n"
                  << flattened << std::endl;
        // {1, 2, 3, 4, 5, 6}

        // 3. 添加维度
        auto expanded = xt::reshape_view(arr1, {1, static_cast<int>(arr1.size())});
        std::cout << "Expand dim:\n"
                  << expanded << std::endl;
        // {{1, 2, 3, 4, 5, 6}}

        // 4. 使用-1自动计算维度
        auto auto_reshaped = xt::reshape_view(arr1, {3, -1});
        std::cout << "Auto-reshaped to 3x2:\n"
                  << auto_reshaped << std::endl;
        /*
        {{1, 2},
         {3, 4},
         {5, 6}}
        */

        // 5. 修改视图会影响原始数据
        reshaped1(0, 1) = 10;
        std::cout << "Original array after modifying view:\n"
                  << arr1 << std::endl;
        // {1, 10, 3, 4, 5, 6}

        // 6. 使用 eval 避免影响原始数据
        auto reshaped3 = xt::eval(xt::reshape_view(arr1, {2, 3}));
        reshaped3(0, 1) = 100;
        std::cout << "Original array after modifying view:\n"
                  << arr1 << std::endl;
        // {1, 10, 3, 4, 5, 6}

        // 7. 计算总元素数
        std::cout << "Total elements: " << arr1.size() << std::endl;
        // Total elements: 6

        std::cout << "reshape_view end\n"
                  << std::endl;
    }

    void transpose()
    {
        // 主要可以使用 xt::swapaxes 和 xt::transpose 函数。
        // 如果不想修改原数组，可以使用 xt::eval 函数。
        std::cout << "transpose start" << std::endl;

        // 1. 使用 swapaxes 交换两个维度
        xt::xarray<int> e = xt::reshape_view(xt::arange(24), {2, 3, 4});
        auto e_t = xt::swapaxes(e, 0, 1);
        std::cout << "Original 3D array:\n"
                  << e << std::endl;
        // {{{ 0,  1,  2,  3},
        //   { 4,  5,  6,  7},
        //   { 8,  9, 10, 11}},
        //  {{12, 13, 14, 15},
        //   {16, 17, 18, 19},
        //   {20, 21, 22, 23}}
        // }
        std::cout << "Swapped axes:\n"
                  << e_t << std::endl;
        // {{{ 0,  1,  2,  3},
        //   {12, 13, 14, 15}},
        //  {{ 4,  5,  6,  7},
        //   {16, 17, 18, 19}},
        //  {{ 8,  9, 10, 11},
        //   {20, 21, 22, 23}}}

        // 2. 使用 xt::transpose 进行简单转置
        xt::xarray<int> a = {{1, 2, 3},
                             {4, 5, 6}}; // 2x3矩阵
        std::cout << "Original matrix:\n"
                  << a << std::endl;
        /*
        {{1, 2, 3},
         {4, 5, 6}}
        */

        // 转置矩阵
        auto a_t = xt::transpose(a);
        std::cout << "Transposed matrix:\n"
                  << a_t << std::endl;
        /*
        {{1, 4},
         {2, 5},
         {3, 6}}
        */

        // 3. 使用 transpose 指定维度顺序
        xt::xarray<int> b = xt::reshape_view(xt::arange(24), {2, 3, 4}); // 3D数组
        std::cout << "\nOriginal 3D array:\n"
                  << b << std::endl;
        // {{{ 0,  1,  2,  3},
        //   { 4,  5,  6,  7},
        //   { 8,  9, 10, 11}},
        //  {{12, 13, 14, 15},
        //   {16, 17, 18, 19},
        //   {20, 21, 22, 23}}}
        // 交换维度顺序(0,1,2) -> (2,0,1)
        auto b_t = xt::transpose(b, {2, 0, 1});
        std::cout << "Permuted dimensions:\n"
                  << b_t << std::endl;
        // {{{ 0,  4,  8},
        //   {12, 16, 20}},
        //  {{ 1,  5,  9},
        //   {13, 17, 21}},
        //  {{ 2,  6, 10},
        //   {14, 18, 22}},
        //  {{ 3,  7, 11},
        //   {15, 19, 23}}}

        // 4. 创建可修改的转置视图
        xt::xarray<int> c = {{1, 2, 3},
                             {4, 5, 6}};

        // 创建转置视图
        auto c_t = xt::transpose(c);
        // 修改转置后的矩阵
        c_t(0, 1) = 10; // 修改会影响原矩阵
        std::cout << "\nOriginal matrix after modifying transpose:\n"
                  << c << std::endl;
        // {{ 1, 2, 3 },
        //  {10, 5, 6 }}
        std::cout << "Transposed matrix after modification:\n"
                  << c_t << std::endl;
        // {{ 1, 10 },
        //  { 2,  5 },
        //  { 3,  6 }}

        // 5. 使用 eval 创建转置的副本
        xt::xarray<int> d = {{1, 2, 3},
                             {4, 5, 6}};

        // 创建转置的副本
        auto d_t = xt::eval(xt::transpose(d));
        // 修改副本不会影响原矩阵
        d_t(0, 1) = 10;
        std::cout << "\nOriginal matrix (unchanged):\n"
                  << d << std::endl;
        // {{ 1, 2, 3 },
        //  { 4, 5, 6 }}
        std::cout << "Transposed copy after modification:\n"
                  << d_t << std::endl;
        // {{ 1, 10 },
        //  { 2,  5 },
        //  { 3,  6 }}

        std::cout << "transpose end\n"
                  << std::endl;
    }

    void squeeze_expand_dims()
    {
        // xt::squeeze 和 xt::expand_dims 是两个重要的函数，用于压缩和扩展数组的维度。
        // 如果不想修改原数组，可以使用 xt::eval 函数。
        std::cout << "squeeze expand_dims start" << std::endl;

        // 1. 使用 squeeze 压缩维度
        xt::xarray<int> a = {{{1, 2, 3}},
                             {{4, 5, 6}}}; // 2x3矩阵

        // 压缩维度
        auto a_s = xt::squeeze(a);
        std::cout << "Original matrix:\n"
                  << a << std::endl;
        // {{{1, 2, 3}},
        //  {{4, 5, 6}}}
        std::cout << "Squeezed matrix:\n"
                  << a_s << std::endl;
        // {{1, 2, 3},
        //  {4, 5, 6}}

        // 2. 使用 expand_dims 扩展维度
        auto a_e = xt::expand_dims(a_s, 0);
        std::cout << "Original matrix:\n"
                  << a_s << std::endl;
        // {{1, 2, 3},
        //  {4, 5, 6}}
        std::cout << "Expanded matrix:\n"
                  << a_e << std::endl;
        // {{{1, 2, 3}},
        //  {{4, 5, 6}}}

        a_e = xt::expand_dims(a_s, 1);
        std::cout << "Original matrix:\n"
                  << a_s << std::endl;
        // {{1, 2, 3},
        //  {4, 5, 6}}
        std::cout << "Expanded matrix:\n"
                  << a_e << std::endl;
        // {{{1, 2, 3}},
        //  {{4, 5, 6}}}

        std::cout << "squeeze expand_dims end\n"
                  << std::endl;
    }

    void concatenate()
    {
        // xt::concatenate 是用于合并数组的函数，可以指定轴和方向。
        // 如果不想修改原数组，可以使用 xt::eval 函数。
        std::cout << "concatenate start" << std::endl;

        // 1. 合并数组
        xt::xarray<int> a = {{1, 2, 3},
                             {4, 5, 6}};
        xt::xarray<int> b = {{7, 8, 9},
                             {10, 11, 12}};
        xt::xarray<int> c = {{13, 14, 15},
                             {16, 17, 18}};

        // 合并数组
        auto ab = xt::concatenate(xt::xtuple(a, b), 0);
        std::cout << "Concatenated array:\n"
                  << ab << std::endl;
        // {{1, 2, 3},
        //  {4, 5, 6},
        //  {7, 8, 9},
        //  {10, 11, 12}}

        auto ac = xt::concatenate(xt::xtuple(a, c), 1);
        std::cout << "Concatenated array:\n"
                  << ac << std::endl;
        // {{1, 2, 3, 13, 14, 15},
        //  {4, 5, 6, 16, 17, 18}}

        auto abc1 = xt::concatenate(xt::xtuple(a, b, c), 0);
        std::cout << "Concatenated array:\n"
                  << abc1 << std::endl;
        // {{ 1,  2,  3},
        //  { 4,  5,  6},
        //  { 7,  8,  9},
        //  {10, 11, 12},
        //  {13, 14, 15},
        //  {16, 17, 18}}

        auto abc2 = xt::concatenate(xt::xtuple(a, b, c), 1);
        std::cout << "Concatenated array:\n"
                  << abc2 << std::endl;
        // {{ 1,  2,  3,  7,  8,  9, 13, 14, 15},
        //  { 4,  5,  6, 10, 11, 12, 16, 17, 18}}

        // 2. 合并不同类型数组
        xt::xarray<int> d = {{1, 2, 3},
                             {4, 5, 6}};
        xt::xarray<double> e = {{7.5, 8.5, 9.5},
                                {10.5, 11.5, 12.5}};

        // 合并不同类型数组
        auto de = xt::concatenate(xt::xtuple(d, e), 1);
        std::cout << "Concatenated array:\n"
                  << de << std::endl;
        // {{1, 2, 3, 7, 8, 9},
        //  {4, 5, 6, 10, 11, 12}}

        auto f = xt::xarray<double>{1, 2, 3, 4, 5};
        std::cout << "f:\n"
                  << f << std::endl;
        // { 1.,  2.,  3.,  4.,  5.}

        xt::xarray<double> f1 = xt::reshape_view(f, {static_cast<int>(f.shape(0)), -1});
        std::cout << "f1:\n"
                  << f1 << std::endl;
        //{{ 1.},
        // { 2.},
        // { 3.},
        // { 4.},
        // { 5.}}

        xt::xarray<double> f2 = xt::concatenate(xt::xtuple(f, f), 0);
        std::cout << "f2:\n"
                  << f2 << std::endl;
        // { 1.,  2.,  3.,  4.,  5.,  1.,  2.,  3.,  4.,  5.}

        // 一维数组合并到 dim=1 是无效的
        xt::xarray<double> f3 = xt::concatenate(xt::xtuple(f, f), 1);
        std::cout << "f3:\n"
                  << f3 << std::endl;
        // { 1.,  2.,  3.,  4.,  5.}

        // 要提前将原数组扩充维度
        xt::xarray<double> f4 = xt::concatenate(xt::xtuple(f1, f1), 1);
        std::cout << "f4:\n"
                  << f4 << std::endl;
        //{{ 1.,  1.},
        // { 2.,  2.},
        // { 3.,  3.},
        // { 4.,  4.},
        // { 5.,  5.}}

        std::cout << "concatenate end\n"
                  << std::endl;
    }

    void stack()
    {
        std::cout << "stack start" << std::endl;
        auto a = xt::xarray<double>{1, 2, 3, 4, 5};
        std::cout << "Original array:\n"
                  << a << std::endl;
        // { 1.,  2.,  3.,  4.,  5.}

        auto b = xt::stack(xt::xtuple(a, a), 0);
        std::cout << "Stacked array:\n"
                  << b << std::endl;
        // {{ 1.,  2.,  3.,  4.,  5.},
        //  { 1.,  2.,  3.,  4.,  5.}}

        auto c = xt::stack(xt::xtuple(a, a), 1);
        std::cout << "Stacked array:\n"
                  << c << std::endl;
        // {{ 1.,  1.},
        //  { 2.,  2.},
        //  { 3.,  3.},
        //  { 4.,  4.},
        //  { 5.,  5.}}

        std::cout << "stack end\n"
                  << std::endl;
    }

    void sort()
    {
        std::cout << "sort start" << std::endl;
        xt::xarray<double> arr = {1, 3, 1, 6, 4, 8, -1, 4, 5, 6};
        std::cout << "arr:\n"
                  << arr << std::endl;
        //{ 1., 3., 1., 6., 4., 8., -1., 4., 5., 6. }

        std::cout << "sorted axis=-1:\n"
                  << xt::sort(arr) << std::endl;
        //{-1., 1., 1., 3., 4., 4., 5., 6., 6., 8.}

        std::cout << "sorted axis=-1 index:\n"
                  << xt::argsort(arr) << std::endl;
        //{6, 0, 2, 1, 4, 7, 8, 3, 9, 5}

        arr = {{1, 3, 1, 6, 4},
               {8, -1, 4, 5, 6},
               {1, 10, 9, 7, 8},
               {10, 12, 1, 3, 7}};

        // 默认在最后一个轴上进行排序, 即按列排序
        xt::xarray<double> arr1 = xt::sort(arr);
        std::cout << "arr:\n"
                  << arr << std::endl;
        //{{  1.,   3.,   1.,   6.,   4.},
        // {  8.,  -1.,   4.,   5.,   6.},
        // {  1.,  10.,   9.,   7.,   8.},
        // { 10.,  12.,   1.,   3.,   7.}}

        std::cout << "sorted axis=-1:\n"
                  << arr1 << std::endl;
        //{{  1.,   1.,   3.,   4.,   6.},
        // { -1.,   4.,   5.,   6.,   8.},
        // {  1.,   7.,   8.,   9.,  10.},
        // {  1.,   3.,   7.,  10.,  12.}}

        xt::xarray<double> arr2 = xt::sort(arr, 0);
        std::cout << "sorted axis=0:\n"
                  << arr2 << std::endl;
        //{{  1.,  -1.,   1.,   3.,   4.},
        // {  1.,   3.,   1.,   5.,   6.},
        // {  8.,  10.,   4.,   6.,   7.},
        // { 10.,  12.,   9.,   7.,   8.}}

        xt::xarray<int> arr3 = xt::argsort(arr);
        std::cout << "sorted axis=-1 index:\n"
                  << arr3 << std::endl;
        //{{0, 2, 1, 4, 3},
        // {1, 2, 3, 4, 0},
        // {0, 3, 4, 2, 1},
        // {2, 3, 4, 0, 1}}

        xt::xarray<int> arr4 = xt::argsort(arr, 0);
        std::cout << "sorted axis=0 index:\n"
                  << arr4 << std::endl;
        //{{0, 1, 0, 3, 0},
        // {2, 0, 3, 1, 1},
        // {1, 2, 1, 0, 3},
        // {3, 3, 2, 2, 2}}

        std::cout << "sort end\n"
                  << std::endl;
    }

    void where()
    {
        std::cout << "where start" << std::endl;

        // 创建一个示例数组
        xt::xarray<double> arr = {{1, 2, 3},
                                  {4, 5, 6},
                                  {7, 8, 9}};

        auto arr_flatten = xt::flatten(arr);
        // 找出所有大于5的元素的索引
        // xt::where returns a tuple of coordinate arrays
        auto indices = xt::where(arr_flatten > 5);

        // indices[0] 包含行索引
        // indices[1] 包含列索引
        // std::cout << "indices: " << indices << std::endl;
        // std::cout << "Row indices: " << indices[0] << std::endl;
        // std::cout << "Column indices: " << indices[1] << std::endl;

        // 使用这些索引访问原始元素
        // 使用 index_view 时，要将所有数据看作一维数据
        auto values = xt::eval(xt::index_view(arr_flatten, indices[0]));
        std::cout << "Values: " << values << std::endl;
        // Values: { 6.,  7.,  8.,  9.}

        xt::xarray<bool> b = {false, true, true, false};
        xt::xarray<int> a1 = {1, 2, 3, 4};
        xt::xarray<int> a2 = {11, 12, 13, 14};

        xt::xarray<int> res = xt::where(b, a1, a2);
        std::cout << "Result:\n"
                  << res << std::endl;
        // {11,  2,  3, 14}

        b = b.reshape({2, 2});
        a1 = a1.reshape({2, 2});
        a2 = a2.reshape({2, 2});

        res = xt::where(b, a1, a2);
        std::cout << "Result:\n"
                  << res << std::endl;
        // {{11,  2},
        //  { 3, 14}}

        std::cout << "where end\n"
                  << std::endl;
    }

    void quantile()
    {
        std::cout << "quantile start" << std::endl;

        xt::xarray<double> arr = xt::arange<double>(0, 101);
        std::cout << "arr:\n"
                  << arr << std::endl;
        //{   0.,    1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,   10.,   11.,   12.,   13.,   14.,
        //   15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,   23.,   24.,   25.,   26.,   27.,   28.,   29.,
        //   30.,   31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,   40.,   41.,   42.,   43.,   44.,
        //   45.,   46.,   47.,   48.,   49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,
        //   60.,   61.,   62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,   73.,   74.,
        //   75.,   76.,   77.,   78.,   79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,
        //   90.,   91.,   92.,   93.,   94.,   95.,   96.,   97.,   98.,   99.,  100.}

        xt::xarray<double> q = xt::quantile(arr, { 0.5 }, 0);
        std::cout << "quantile 0.5: " << q << std::endl;
        // quantile 0.5: { 50.}

        q = xt::quantile(arr, { 0 }, 0);
        std::cout << "quantile 0.0: " << q << std::endl;
        // quantile 0.0 : { 0.}

        q = xt::quantile(arr, { 0.95 }, 0);
        std::cout << "quantile 0.95: " << q << std::endl;
        // quantile 0.95 : { 95.        }

        q = xt::quantile(arr, { 0.05, 0.95 }, 0);
        std::cout << "quantile 0.05, 0.95: " << q << std::endl;
        // quantile 0.05, 0.95 : {  5., 95.        }

        arr = xt::logspace<double>(0, 2, 10);
        std::cout << "arr:\n"
            << arr << std::endl;
        //{   1.        ,    1.66810054,    2.7825594 ,    4.64158883,    7.74263683,   12.91549665,
        //   21.5443469 ,   35.93813664,   59.94842503,  100.        }

        q = xt::quantile(arr, { 0.0, 0.5, 0.95, 1.0 }, 0);
        std::cout << "quantile 0.0, 0.5, 0.95, 1.0: " << q << std::endl;
        // quantile 0.0, 0.5, 0.95, 1.0: {   1.        ,   10.32906674,   81.97679126,  100.        }

        std::cout << "quantile end\n"
                  << std::endl;
    }

    void eval()
    {
        // xt::eval 是 xtensor 中一个重要的函数，用于强制计算表达式并返回结果。
        // 返回的结果不影响原始数据。
        std::cout << "eval start" << std::endl;

        // 1. 基本用法 - 简单表达式
        xt::xarray<double> a = {1, 2, 3, 4};
        xt::xarray<double> b = {5, 6, 7, 8};

        // 创建一个延迟计算的表达式
        auto expr = a + b * 2;
        // 强制计算表达式
        auto result = xt::eval(expr);
        std::cout << "Expression result:\n"
                  << result << std::endl;
        // {11, 14, 17, 20}

        // 2. 避免重复计算
        // 不使用eval，表达式会被计算多次
        auto complex_expr = xt::cos(a) + xt::sin(b);
        for (size_t i = 0; i < 3; ++i)
        {
            // 每次使用都会重新计算 cos(a) 和 sin(b)
            std::cout << complex_expr << std::endl;
        }

        // 使用eval，表达式只计算一次
        auto evaluated = xt::eval(xt::cos(a) + xt::sin(b));
        for (size_t i = 0; i < 3; ++i)
        {
            // 直接使用计算好的结果
            std::cout << evaluated << std::endl;
        }

        // 3. 临时视图的计算
        xt::xarray<double> matrix = xt::ones<double>({3, 3});
        // 对视图进行计算并存储结果
        auto view_result = xt::eval(xt::view(matrix, xt::range(0, 2), xt::range(0, 2)) * 2);
        std::cout << "View result:\n"
                  << view_result << std::endl;

        // 4. 链式操作中的使用
        auto chain_result = xt::eval(xt::cos(xt::sin(a + b)));
        std::cout << "Chain result:\n"
                  << chain_result << std::endl;

        std::cout << "eval end\n"
                  << std::endl;
    }

    void deep_copy()
    {
        std::cout << "deep copy start" << std::endl;

        // 1. 使用拷贝构造函数
        xt::xarray<int> a = {{1, 2, 3},
                             {4, 5, 6}};
        xt::xarray<int> a_copy(a); // 深拷贝
        // 修改副本
        a_copy(0, 0) = 10;
        std::cout << "Original array:\n"
                  << a << std::endl;
        std::cout << "Copy modified:\n"
                  << a_copy << std::endl;

        // 2. 使用operator=
        xt::xarray<int> b = {{1, 2, 3},
                             {4, 5, 6}};
        xt::xarray<int> b_copy = b; // 深拷贝
        b_copy(0, 0) = 10;
        std::cout << "\nOriginal array:\n"
                  << b << std::endl;
        std::cout << "Copy modified:\n"
                  << b_copy << std::endl;

        // 3. 使用xt::eval
        auto expr = b * 2;            // 创建表达式
        auto c_copy = xt::eval(expr); // 深拷贝表达式结果
        c_copy(0, 0) = 10;            // 修改副本
        std::cout << "\nOriginal array:\n"
                  << expr << std::endl;
        std::cout << "Copy modified::\n"
                  << c_copy << std::endl;

        // 4. 使用xt::zeros_like或ones_like后复制
        xt::xarray<int> d = {{1, 2, 3},
                             {4, 5, 6}};
        auto d_copy = xt::zeros_like(d); // 创建相同形状的数组
        d_copy = d;                      // 复制数据
        d_copy(0, 0) = 10;
        std::cout << "\nOriginal array:\n"
                  << d << std::endl;
        std::cout << "Copy modified:\n"
                  << d_copy << std::endl;

        // 5. 使用xt::xarray::from_shape
        xt::xarray<int> e = {{1, 2, 3},
                             {4, 5, 6}};
        auto e_copy = xt::xarray<int>::from_shape(e.shape());
        std::copy(e.begin(), e.end(), e_copy.begin());
        e_copy(0, 0) = 10;
        std::cout << "\nOriginal array:\n"
                  << e << std::endl;
        std::cout << "Copy modified:\n"
                  << e_copy << std::endl;

        std::cout << "deep copy end\n"
                  << std::endl;
    }

    void meshgrid()
    {
        std::cout << "meshgrid end\n"
                  << std::endl;

        xt::xarray<double> x = {0, 1, 2, 3};
        xt::xarray<double> y = {0, 1, 2};

        // ij indexing
        auto xy_grid = xt::meshgrid(y, x);
        auto x_grid = std::get<1>(xy_grid);
        auto y_grid = std::get<0>(xy_grid);

        std::cout << "x_grid:\n"
                  << x_grid << std::endl;
        //{{ 0.,  1.,  2.,  3.},
        // { 0.,  1.,  2.,  3.},
        // { 0.,  1.,  2.,  3.}}

        std::cout << "y_grid:\n"
                  << y_grid << std::endl;
        //{{ 0.,  0.,  0.,  0.},
        // { 1.,  1.,  1.,  1.},
        // { 2.,  2.,  2.,  2.}}

        std::cout << "meshgrid end\n"
                  << std::endl;
    }

    void triu_tril()
    {
        std::cout << "triu_tril start" << std::endl;
        xt::xarray<double> a = {{1, 2, 3},
                                {4, 5, 6},
                                {7, 8, 9}};
        std::cout << "Original array:\n"
                  << a << std::endl;
        //{{ 1.,  2.,  3.},
        // { 4.,  5.,  6.},
        // { 7.,  8.,  9.}}

        xt::xarray<double> upper = xt::triu(a);
        std::cout << "Upper triangle:\n"
                  << upper << std::endl;
        //{{ 1.,  2.,  3.},
        // { 0.,  5.,  6.},
        // { 0.,  0.,  9.}}

        upper = xt::triu(a, 1);
        std::cout << "Upper triangle k=1:\n"
                  << upper << std::endl;
        //{{ 0.,  2.,  3.},
        // { 0.,  0.,  6.},
        // { 0.,  0.,  0.}}

        upper = xt::triu(a, -1);
        std::cout << "Upper triangle k=-1:\n"
                  << upper << std::endl;
        //{{ 1.,  2.,  3.},
        // { 4.,  5.,  6.},
        // { 0.,  8.,  9.}}

        xt::xarray<double> lower = xt::tril(a);
        std::cout << "Lower triangle:\n"
                  << lower << std::endl;
        //{{ 1.,  0.,  0.},
        // { 4.,  5.,  0.},
        // { 7.,  8.,  9.}}

        lower = xt::tril(a, 1);
        std::cout << "Lower triangle k=1:\n"
                  << lower << std::endl;
        //{{ 1.,  2.,  0.},
        // { 4.,  5.,  6.},
        // { 7.,  8.,  9.}}

        lower = xt::tril(a, -1);
        std::cout << "Lower triangle k=-1:\n"
                  << lower << std::endl;
        //{{ 0.,  0.,  0.},
        // { 4.,  0.,  0.},
        // { 7.,  8.,  0.}}

        std::cout << "triu_tril end\n"
                  << std::endl;
    }

    void matrix_dot()
    {
        std::cout << "matrix_dot start" << std::endl;

        xt::xarray<double> x = xt::random::randn<double>({256, 1024});

        xt::xarray<double> y = xt::random::randn<double>({1024, 256});

        // 开始计时
        auto start = std::chrono::high_resolution_clock::now();

        xt::xarray<double> z = xt::zeros<double>({256, 256});
        for (int i = 0; i < 10; i++)
        {
            std::cout << i << std::endl;
            z = xt::linalg::dot(x, y);
            // xt::blas::dot(x, y, z); // 只有一个值的结果，原因不明
        }

        // 结束计时
        auto end = std::chrono::high_resolution_clock::now();

        // 计算持续时间并转换为毫秒
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 输出结果
        std::cout << "use " << duration << " ms" << std::endl;

        std::cout << "z: \n"
                  << z << std::endl;

        std::cout << "matrix_dot end\n"
                  << std::endl;
    }

}

#endif // XT_TEST_HPP
