#include <gtest/gtest.h>
#include <tense/tense.h>

using namespace Tense;

using TTypes1 = testing::Types<Tensor<int>>;
using TTypes2 = testing::Types<Tensor<std::complex<int>>>;

template <typename T>
struct Tensor1 : public ::testing::Test
{
};

template <typename T>
struct Tensor2 : public ::testing::Test
{
};

TYPED_TEST_SUITE(Tensor1, TTypes1);
TYPED_TEST_SUITE(Tensor2, TTypes2);

size_t TCount = 0;

template <typename Expr1, typename _Expr2>
void Check(const Expr1 &expr1, _Expr2 _expr2)
{
    auto expr2 = _expr2.eval();
    using Expr2 = decltype(expr2);

    static_assert(std::is_same<typename Expr1::Type, typename Expr2::Type>::value, "Different Types");

    Expr2 temp = expr1;
    if constexpr (std::is_integral<typename Expr1::Type>::value)
    {
        bool X1 = expr2.equal(expr1).item(), X2 = expr2.equal(temp).item();
        if (!X1 || !X2) print(expr1, expr2);
        EXPECT_TRUE(X1);
        EXPECT_TRUE(X2);
    }
    else
    {
        bool X1 = expr2.close(expr1, 1e-5).item(), X2 = expr2.close(temp, 1e-5).item();
        if (!X1 || !X2) print(expr1, expr2);
        EXPECT_TRUE(X1);
        EXPECT_TRUE(X2);
    }
}

TYPED_TEST(Tensor1, Unary)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;
    using BTen = Tensor<bool>;

    Ten M1({1, 2, 3, 1}, {1, 2, 3, 4, 5, 6});
    BTen M2({1, 2, 3, 1}, {1, 0, 1, 0, 1, 0});
    Check(M1.template unary<Type>([](auto val1) { return -val1; }), Ten({1, 2, 3, 1}, {-1, -2, -3, -4, -5, -6}));
    Check(M1.neg().abs().square(), Ten({1, 2, 3, 1}, {1, 4, 9, 16, 25, 36}));
    Check(M1.pow(2).fmax(10).add(-2), Ten({1, 2, 3, 1}, {-1, 2, 7, 8, 8, 8}));
    Check(M1.sub(3).zero(), BTen({1, 2, 3, 1}, {0, 0, 1, 0, 0, 0}));
    Check(M1.gt(3), BTen({1, 2, 3, 1}, {0, 0, 0, 1, 1, 1}));
    Check(M2._not()._xor(1), M2);
}

TYPED_TEST(Tensor1, Binary)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;
    using BTen = Tensor<bool>;
    using CTen = Tensor<std::complex<Type>>;

    Ten M1({1, 2, 3, 1}, {1, 2, 3, 4, 5, 6}), M2({1, 2, 3, 1}, {-1, -2, -3, -4, -5, -6});
    BTen M3({1, 2, 3, 1}, {1, 0, 1, 0, 1, 0});
    Check(M1.template binary<Type>(M2, [](auto val1, auto val2) { return val1 + val2; }),
          Ten({1, 2, 3, 1}, {0, 0, 0, 0, 0, 0}));
    Check(M1.mul(M2).abs().sqrt(), Ten({1, 2, 3, 1}, {1, 2, 3, 4, 5, 6}));
    Check(M1.rshift(1).gt(1), BTen({1, 2, 3, 1}, {0, 0, 0, 1, 1, 1}));
    Check(M1.mask(Ten({1, 2, 3, 1}, {0, 1, 0, 1, 0, 1})), Ten({1, 2, 3, 1}, {0, 2, 0, 4, 0, 6}));
    Check(M3._and(M3._not())._or(M3), M3);
    Check(Ten({1, 2, 3, 1}, 1).complex(Ten({1, 2, 3, 1}, 2)), CTen({1, 2, 3, 1}, std::complex<Type>(1, 2)));
}

TYPED_TEST(Tensor1, Reduce)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;
    using BTen = Tensor<bool>;
    using STen = Tensor<Size>;

    Ten M1({2, 2, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1});
    Ten M2({2, 2, 3}, {0, 3, 2, 2, 1, 3, 2, 1, 2, 0, 1, 2});
    Check(M1.reduce(
              0, [](auto val1, auto val2) { return val1 + val2; }, 2),
          Ten({2, 2}, {1, 1, 1, 3}));
    Check(M1.reduce(
              0, [](auto val1, auto val2) { return val1 + val2; }, 1),
          Ten({2}, {2, 4}));
    Check(M1.reduce(
              0, [](auto val1, auto val2) { return val1 + val2; }, 0),
          Ten({1}, {6}));

    Check(M1.sum(2), Ten({2, 2}, {1, 1, 1, 3}));
    Check(M1.sum(1), Ten({2}, {2, 4}));
    Check(M1.sum(0), Ten({1}, {6}));

    Check(M1.prod(2), Ten({2, 2}, {0, 0, 0, 1}));
    Check(M1.prod(1), Ten({2}, {0, 0}));
    Check(M1.prod(0), Ten({1}, {0}));

    Check(M2.max(2), Ten({2, 2}, {3, 3, 2, 2}));
    Check(M2.max(1), Ten({2}, {3, 2}));
    Check(M2.max(0), Ten({1}, {3}));

    Check(M2.min(2), Ten({2, 2}, {0, 1, 1, 0}));
    Check(M2.min(1), Ten({2}, {0, 0}));
    Check(M2.min(0), Ten({1}, {0}));

    Check(M2.count(2, 2), STen({2, 2}, {1, 1, 2, 1}));
    Check(M2.count(2, 1), STen({2}, {2, 3}));
    Check(M2.count(2, 0), STen({1}, {5}));

    Check(M2.contains(3, 2), BTen({2, 2}, {1, 1, 0, 0}));
    Check(M2.contains(3, 1), BTen({2}, {1, 0}));
    Check(M2.contains(3, 0), BTen({1}, {1}));

    Check(M1.equal(1, 2), BTen({2, 2}, {0, 0, 0, 1}));
    Check(M1.equal(1, 1), BTen({2}, {0, 0}));
    Check(M1.equal(1, 0), BTen({1}, {false}));
}

TYPED_TEST(Tensor1, Repeat)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;

    Ten M1({1, 2}, {1, 2}), M2({3}, {1, 2, 3});
    Check(M1.repeat({1}), Ten({1, 1, 2}, {1, 2}));
    Check(M1.repeat({2}), Ten({2, 1, 2}, {1, 2, 1, 2}));
    Check(M1.repeat({2, 1}), Ten({2, 1, 1, 2}, {1, 2, 1, 2}));
    Check(M1.repeat({2, 2}), Ten({2, 2, 1, 2}, {1, 2, 1, 2, 1, 2, 1, 2}));

    Check(M2.repeat({1}), Ten({1, 3}, {1, 2, 3}));
    Check(M2.repeat({3, 1}), Ten({3, 1, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3}));
}

TYPED_TEST(Tensor1, Select)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;

    Ten M1({3}, {1, 2, 3}), M2({2, 2}, {1, 2, 3, 4}), M3({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_EQ(M1({0}), 1);
    EXPECT_EQ(M1({1}), 2);
    EXPECT_EQ(M1({2}), 3);

    EXPECT_EQ(M2({0, 0}), 1);
    EXPECT_EQ(M2({0, 1}), 2);
    EXPECT_EQ(M2({1, 0}), 3);
    EXPECT_EQ(M2({1, 1}), 4);

    EXPECT_EQ(M3({0, 0, 0}), 1);
    EXPECT_EQ(M3({0, 0, 1}), 2);
    EXPECT_EQ(M3({0, 1, 0}), 3);
    EXPECT_EQ(M3({0, 1, 1}), 4);
    EXPECT_EQ(M3({1, 0, 0}), 5);
    EXPECT_EQ(M3({1, 0, 1}), 6);
    EXPECT_EQ(M3({1, 1, 0}), 7);
    EXPECT_EQ(M3({1, 1, 1}), 8);
}

TYPED_TEST(Tensor1, All)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;
    using BTen = Tensor<bool>;

    Ten M1({1, 4, 4}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1});
    Check(M1.all(2), BTen({1, 4}, {0, 0, 0, 1}));
    Check(M1.all(1), BTen({1}, {0}));
    Check(M1.all(0), BTen({1}, {0}));
}

TYPED_TEST(Tensor1, Any)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;
    using BTen = Tensor<bool>;

    Ten M1({1, 4, 4}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1});
    Check(M1.any(2), BTen({1, 4}, {0, 1, 1, 1}));
    Check(M1.any(1), BTen({1}, {1}));
    Check(M1.any(0), BTen({1}, {1}));
}

TYPED_TEST(Tensor1, Flip)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;

    Ten M1({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Check(M1.flip(2), Ten({2, 2, 3}, {3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10}));
    Check(M1.flip(1), Ten({2, 2, 3}, {6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7}));
    Check(M1.flip(0), Ten({2, 2, 3}, {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}));
}

TYPED_TEST(Tensor1, Index)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;
    using STen = Tensor<Size>;

    Ten M1({2, 3, 3}, {2, 0, 1, 0, 1, 1, 0, 2, 3,  //
                       1, 2, 2, 1, 0, 1, 2, 1, 0});
    Check(M1.maxi(2), STen({2, 3}, {0, 1, 2, 1, 0, 0}));
    Check(M1.maxi(1), STen({2}, {8, 1}));
    Check(M1.maxi(0), STen({1}, {8}));

    Check(M1.mini(2), STen({2, 3}, {1, 0, 0, 0, 1, 2}));
    Check(M1.mini(1), STen({2}, {1, 4}));
    Check(M1.mini(0), STen({1}, {1}));
}

TYPED_TEST(Tensor1, Cat)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;

    Ten M1({3}, {1, 2, 3}), M2({3}, {4, 5, 6});
    Check(Ten::cat({M1, M2}, 0), Ten({2, 3}, {1, 2, 3, 4, 5, 6}));
    Check(Ten::cat({M1, M2}, 1), Ten({6}, {1, 4, 2, 5, 3, 6}));

    Ten M3({2, 2}, {1, 2, 3, 4}), M4({2, 2}, {5, 6, 7, 8});
    Check(Ten::cat({M3, M4}, 0), Ten({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    Check(Ten::cat({M3, M4}, 1), Ten({4, 2}, {1, 2, 5, 6, 3, 4, 7, 8}));
    Check(Ten::cat({M3, M4}, 2), Ten({2, 4}, {1, 5, 2, 6, 3, 7, 4, 8}));
}

TYPED_TEST(Tensor2, Complex)
{
    using Ten = TypeParam;
    using Type = typename Ten::Type;
    using FTen = Tensor<typename Type::value_type>;

    Ten M1({1, 2, 3, 1}, Type(4, 0));
    Check(M1.abs(), FTen({1, 2, 3, 1}, 4));
    Check(M1.conj(), M1);
    Check(M1.proj(), Ten({1, 2, 3, 1}, std::proj(Type(4, 0))));
    Check(M1.polar(), Ten({1, 2, 3, 1}, std::polar(4, 0)));
    Check(M1.arg(), FTen({1, 2, 3, 1}, std::arg(Type(4, 0))));
    Check(M1.norm(), FTen({1, 2, 3, 1}, std::norm(Type(4, 0))));
    Check(M1.real(), FTen({1, 2, 3, 1}, 4));
    Check(M1.imag(), FTen({1, 2, 3, 1}, {0, 0, 0, 0, 0, 0}));
    Check(M1.template type<typename Type::value_type>(), M1.real().eval());
}
