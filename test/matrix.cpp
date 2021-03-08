#include <gtest/gtest.h>
#include <tense/tense.h>

using namespace Tense;

using MTypes1 = testing::Types<Matrix<Row, int>, Matrix<Col, int>>;
using MTypes2 = testing::Types<Matrix<Row, float>, Matrix<Col, float>>;
using MTypes3 = testing::Types<Matrix<Row, std::complex<int>>, Matrix<Col, std::complex<int>>>;
using MTypes4 = testing::Types<Matrix<Row, std::complex<float>>, Matrix<Col, std::complex<float>>>;

template <typename T>
struct Matrix1 : public ::testing::Test
{
};
template <typename T>
struct Matrix2 : public ::testing::Test
{
};
template <typename T>
struct Matrix3 : public ::testing::Test
{
};
template <typename T>
struct Matrix4 : public ::testing::Test
{
};

TYPED_TEST_SUITE(Matrix1, MTypes1);
TYPED_TEST_SUITE(Matrix2, MTypes2);
TYPED_TEST_SUITE(Matrix3, MTypes3);
TYPED_TEST_SUITE(Matrix4, MTypes4);

size_t MCount = 0;

template <typename Expr1, typename _Expr2>
void Check(const Expr1 &expr1, _Expr2 _expr2)
{
    if constexpr (std::is_integral<Expr1>::value && std::is_integral<_Expr2>::value)
        EXPECT_TRUE(std::abs(double(expr1) - double(_expr2)) < 1e-5);
    else
    {
        auto expr2 = _expr2.eval();
        using Expr2 = decltype(expr2);

        static_assert(std::is_same<typename Expr1::Type, typename Expr2::Type>::value, "Different Types");
        static_assert(std::is_same<typename Expr1::Major, typename Expr2::Major>::value, "Different Majors");

        Expr2 temp = expr1;
        if constexpr (std::is_integral<typename Expr1::Type>::value)
        {
            bool X1 = expr2.equal(expr1), X2 = expr2.equal(temp);
            if (!X1 || !X2) print(expr1, expr2);
            EXPECT_TRUE(X1);
            EXPECT_TRUE(X2);
        }
        else
        {
            using Type = typename Tense::Impl::FromComplex<typename Expr1::Type>::Type;
            bool X1 = expr2.close(expr1, Type(1e-5)), X2 = expr2.close(temp, Type(1e-5));
            if (!X1 || !X2) print(expr1, expr2);
            EXPECT_TRUE(X1);
            EXPECT_TRUE(X2);
        }
    }
}

TYPED_TEST(Matrix1, Matrix)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    {
        Mat M;
        EXPECT_FALSE(M.valid());
    }

    {
        Mat M(2, 3, 4);
        EXPECT_TRUE(M.valid());
        EXPECT_EQ(M.rows(), 2);
        EXPECT_EQ(M.cols(), 3);
        EXPECT_EQ(M.size(), 6);
        for (Size i = 0; i < 2; ++i)
            for (Size j = 0; j < 3; ++j) EXPECT_EQ(M(i, j), 4);
    }

    {
        Mat M({{1, 2, 3}, {4, 5, 6}});
        EXPECT_TRUE(M.valid());
        EXPECT_EQ(M.rows(), 2);
        EXPECT_EQ(M.cols(), 3);
        EXPECT_EQ(M.size(), 6);
        EXPECT_EQ(M(0, 0), 1);
        EXPECT_EQ(M(0, 1), 2);
        EXPECT_EQ(M(0, 2), 3);
        EXPECT_EQ(M(1, 0), 4);
        EXPECT_EQ(M(1, 1), 5);
        EXPECT_EQ(M(1, 2), 6);
    }

    {
        std::vector<int> V = {1, 2, 3, 4, 5, 6};
        if (std::is_same<Major, Col>::value) V = {1, 4, 2, 5, 3, 6};
        Mat M(2, 3, V.data());
        EXPECT_TRUE(M.valid());
        EXPECT_EQ(M.rows(), 2);
        EXPECT_EQ(M.cols(), 3);
        EXPECT_EQ(M.size(), 6);
    }

    {
        Mat M(2, 3);
        M = 4;
        EXPECT_TRUE(M.valid());
        EXPECT_EQ(M.rows(), 2);
        EXPECT_EQ(M.cols(), 3);
        EXPECT_EQ(M.size(), 6);
        for (Size i = 0; i < 2; ++i)
            for (Size j = 0; j < 3; ++j) EXPECT_EQ(M(i, j), 4);
    }

    {
        Mat M({{1, 2, 3}, {4, 5, 6}});
        M = M.reshape(3, 2).eval();
        EXPECT_TRUE(M.valid());
        EXPECT_EQ(M.rows(), 3);
        EXPECT_EQ(M.cols(), 2);
        EXPECT_EQ(M.size(), 6);
        EXPECT_EQ(M(0, 0), 1);
        EXPECT_EQ(M(0, 1), 2);
        EXPECT_EQ(M(1, 0), 3);
        EXPECT_EQ(M(1, 1), 4);
        EXPECT_EQ(M(2, 0), 5);
        EXPECT_EQ(M(2, 1), 6);
    }

    {
        Mat M({{1, 2, 3}, {4, 5, 6}});
        M.resize(2, 2);
        EXPECT_TRUE(M.valid());
        EXPECT_EQ(M.rows(), 2);
        EXPECT_EQ(M.cols(), 2);
        EXPECT_EQ(M.size(), 4);
        EXPECT_EQ(M(0, 0), 1);
        EXPECT_EQ(M(0, 1), 2);
        EXPECT_EQ(M(1, 0), 4);
        EXPECT_EQ(M(1, 1), 5);
        M.resize(4, 1);
        EXPECT_TRUE(M.valid());
        EXPECT_EQ(M.rows(), 4);
        EXPECT_EQ(M.cols(), 1);
        EXPECT_EQ(M.size(), 4);
        EXPECT_EQ(M(0, 0), 1);
        EXPECT_EQ(M(1, 0), 4);
        EXPECT_EQ(M(2, 0), 0);
        EXPECT_EQ(M(3, 0), 0);
    }
}

TYPED_TEST(Matrix1, Unary)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;
    using BMat = Matrix<Major, bool>;

    Mat M1 = {{1, 2, 3}, {4, 5, 6}};
    BMat M2 = {{1, 0, 1}, {0, 1, 0}};
    Check(M1.template unary<Type>([](auto val1) { return -val1; }), Mat({{-1, -2, -3}, {-4, -5, -6}}));
    Check(M1.neg().abs().square(), Mat({{1, 4, 9}, {16, 25, 36}}));
    Check(M1.pow(2).fmax(10).add(-2), Mat({{-1, 2, 7}, {8, 8, 8}}));
    Check(M1.sub(3).zero(), BMat({{0, 0, 1}, {0, 0, 0}}));
    Check(M1.gt(3), BMat({{0, 0, 0}, {1, 1, 1}}));
    Check(M2._not()._xor(1), M2);
}

TYPED_TEST(Matrix1, Binary)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;
    using BMat = Matrix<Major, bool>;
    using CMat = Matrix<Major, std::complex<Type>>;

    Mat M1 = Mat({{1, 2, 3}, {4, 5, 6}}), M2 = Mat({{-1, -2, -3}, {-4, -5, -6}});
    BMat M3 = BMat({{1, 0, 1}, {0, 1, 0}});
    Check(M1.template binary<Type>(M2, [](auto val1, auto val2) { return val1 + val2; }), Mat({{0, 0, 0}, {0, 0, 0}}));
    Check(M1.mul(M2).abs().sqrt(), Mat({{1, 2, 3}, {4, 5, 6}}));
    Check(M1.rshift(1).gt(1), BMat({{0, 0, 0}, {1, 1, 1}}));
    Check(M1.mask(Mat({{0, 1, 0}, {1, 0, 1}})), Mat({{0, 2, 0}, {4, 0, 6}}));
    Check(M3._and(M3._not())._or(M3), M3);
    Check(Mat(2, 3, 1).complex(Mat(2, 3, 2)), CMat(2, 3, std::complex<Type>(1, 2)));
}

TYPED_TEST(Matrix1, Square)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    Check(M1.upper(), Mat({{1, 2, 3}, {0, 5, 6}, {0, 0, 9}}));
    Check(M1.lower(), Mat({{1, 0, 0}, {4, 5, 0}, {7, 8, 9}}));
    Check(M1.zupper(), Mat({{0, 2, 3}, {0, 0, 6}, {0, 0, 0}}));
    Check(M1.zlower(), Mat({{0, 0, 0}, {4, 0, 0}, {7, 8, 0}}));
    Check(M1.oupper(), Mat({{1, 2, 3}, {0, 1, 6}, {0, 0, 1}}));
    Check(M1.olower(), Mat({{1, 0, 0}, {4, 1, 0}, {7, 8, 1}}));
    Check(M1.usymm(), Mat({{1, 2, 3}, {2, 5, 6}, {3, 6, 9}}));
    Check(M1.lsymm(), Mat({{1, 4, 7}, {4, 5, 8}, {7, 8, 9}}));
    Check(M1.diagonal(), Mat({{1, 0, 0}, {0, 5, 0}, {0, 0, 9}}));
    Check(M1.zdiagonal(), Mat({{0, 2, 3}, {4, 0, 6}, {7, 8, 0}}));
    Check(M1.odiagonal(), Mat({{1, 2, 3}, {4, 1, 6}, {7, 8, 1}}));
}

TYPED_TEST(Matrix1, Reduce)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;
    using BMat = Matrix<Major, bool>;
    using SMat = Matrix<Major, Size>;

    Mat M1 = Mat({{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {1, 1, 0, 1}});
    Mat M2 = Mat({{0, 3, 1, 3}, {3, 3, 2, 3}, {2, 0, 3, 2}, {1, 0, 3, 0}});
    Check(M1.breduce(2, 2, 0, [](auto val1, auto val2) { return val1 + val2; }), Mat({{0, 3}, {2, 3}}));
    Check(M1.rreduce(0, [](auto val1, auto val2) { return val1 + val2; }), Mat({1, 2, 2, 3}));
    Check(M1.creduce(0, [](auto val1, auto val2) { return val1 + val2; }), Mat({{1, 1, 2, 4}}));
    Check(M1.reduce(0, [](auto val1, auto val2) { return val1 + val2; }), 8);

    Check(M1.bsum(2, 2), Mat({{0, 3}, {2, 3}}));
    Check(M1.rsum(), Mat({1, 2, 2, 3}));
    Check(M1.csum(), Mat({{1, 1, 2, 4}}));
    Check(M1.sum(), 8);

    Check(M1.bprod(2, 2), Mat({{0, 0}, {0, 0}}));
    Check(M1.rprod(), Mat({0, 0, 0, 0}));
    Check(M1.cprod(), Mat({{0, 0, 0, 1}}));
    Check(M1.prod(), 0);

    Check(M2.bmax(2, 2), Mat({{3, 3}, {2, 3}}));
    Check(M2.rmax(), Mat({3, 3, 3, 3}));
    Check(M2.cmax(), Mat({{3, 3, 3, 3}}));
    Check(M2.max(), 3);

    Check(M2.bmin(2, 2), Mat({{0, 1}, {0, 0}}));
    Check(M2.rmin(), Mat({0, 2, 0, 0}));
    Check(M2.cmin(), Mat({{0, 0, 1, 0}}));
    Check(M2.min(), 0);

    Check(M2.bcount(2, 2, 3), SMat({{3, 2}, {0, 2}}));
    Check(M2.rcount(3), SMat({2, 3, 1, 1}));
    Check(M2.ccount(3), SMat({{1, 2, 2, 2}}));
    Check(M2.count(3), 7);

    Check(M2.bcontains(2, 2, 0), BMat({{1, 0}, {1, 1}}));
    Check(M2.rcontains(0), BMat({1, 0, 1, 1}));
    Check(M2.ccontains(0), BMat({{1, 1, 0, 1}}));
    Check(M2.contains(0), 1);

    Check(M1.bequal(2, 2, 0), BMat({{1, 0}, {0, 0}}));
    Check(M1.requal(0), BMat(4, 1, false));
    Check(M1.cequal(0), BMat(1, 4, false));
    Check(M1.equal(0), 0);
}

TYPED_TEST(Matrix1, Repeat)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat::init(1, 1, 1), M2 = Mat({{1, 2, 3}}), M3 = Mat({1, 2}), M4 = Mat({{1, 2}, {3, 4}});
    Check(M4.brepeat(2, 2), Mat({{1, 2, 1, 2}, {3, 4, 3, 4}, {1, 2, 1, 2}, {3, 4, 3, 4}}));
    Check(M2.rrepeat(2), Mat({{1, 2, 3}, {1, 2, 3}}));
    Check(M3.crepeat(3), Mat({{1, 1, 1}, {2, 2, 2}}));
    Check(M1.repeat(2, 3), Mat::init(2, 3, 1));
}

TYPED_TEST(Matrix1, Select1)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat({{1, 2, 3}, {4, 5, 6}}), M2 = Mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    Check(M1.block(0, 0, 2, 3), M1);
    Check(M1.block(0, 0, 2, 2), Mat({{1, 2}, {4, 5}}));
    Check(M1.block(0, 1, 2, 2), Mat({{2, 3}, {5, 6}}));
    Check(M1.block(1, 1, 1, 1), Mat::init(1, 1, 5));
    Check(M1.row(0), Mat({{1, 2, 3}}));
    Check(M1.row(1), Mat({{4, 5, 6}}));
    Check(M1.col(0), Mat({1, 4}));
    Check(M1.col(1), Mat({2, 5}));
    Check(M1.col(2), Mat({3, 6}));
    Check(M2.diag(), Mat({1, 5, 9}));
    Check(M1.elem(0, 0), Mat::init(1, 1, 1));
    Check(M1.elem(0, 1), Mat::init(1, 1, 2));
    Check(M1.elem(0, 2), Mat::init(1, 1, 3));
    Check(M1.elem(1, 0), Mat::init(1, 1, 4));
    Check(M1.elem(1, 1), Mat::init(1, 1, 5));
    Check(M1.elem(1, 2), Mat::init(1, 1, 6));
}

TYPED_TEST(Matrix1, Select2)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat({{1, 2, 3}, {4, 5, 6}});
    Check(M1.index({1, 0, 1}, Cut()), Mat({{4, 5, 6}, {1, 2, 3}, {4, 5, 6}}));
    Check(M1.index(Cut(), {2, 1, 2}), Mat({{3, 2, 3}, {6, 5, 6}}));
    Check(M1.index({1, 0, 1}, {2, 1, 2}), Mat({{6, 5, 6}, {3, 2, 3}, {6, 5, 6}}));
}

TYPED_TEST(Matrix1, Turn)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat({{1, 2, 3}, {4, 5, 6}});
    Check(M1.rturn(1), Mat({{4, 5, 6}, {1, 2, 3}}));
    Check(M1.cturn(2), Mat({{2, 3, 1}, {5, 6, 4}}));
    Check(M1.turn(1, 2), Mat({{5, 6, 4}, {2, 3, 1}}));
}

TYPED_TEST(Matrix1, Cat)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat({{1, 2}, {3, 4}}), M2 = Mat({{5, 6}, {7, 8}});
    Check(M1.rcat0(M2), Mat({{1, 2}, {3, 4}, {5, 6}, {7, 8}}));
    Check(M1.ccat0(M2), Mat({{1, 2, 5, 6}, {3, 4, 7, 8}}));
    Check(M1.rcat1(M2), Mat({{1, 2}, {5, 6}, {3, 4}, {7, 8}}));
    Check(M1.ccat1(M2), Mat({{1, 5, 2, 6}, {3, 7, 4, 8}}));
}

TYPED_TEST(Matrix1, Index)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;
    using SMat = Matrix<Major, Size>;

    Mat M1 = Mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    Check(M1.rmaxidx(), SMat({2, 2, 2}));
    Check(M1.cmaxidx(), SMat({{2, 2, 2}}));
    Check(M1.rminidx(), SMat({0, 0, 0}));
    Check(M1.cminidx(), SMat({{0, 0, 0}}));
}

TYPED_TEST(Matrix1, All)
{
    using Type = typename TypeParam::Type;
    using Major = typename TypeParam::Major;
    using Mat = Matrix<Major, bool>;

    Mat M1 = Mat({{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 1, 1}, {1, 1, 1, 1}});
    Check(M1.ball(2, 2), Mat({{0, 0}, {0, 1}}));
    Check(M1.rall(), Mat({0, 0, 0, 1}));
    Check(M1.call(), Mat({{0, 0, 0, 1}}));
    Check(M1.all(), 0);
}

TYPED_TEST(Matrix1, Any)
{
    using Type = typename TypeParam::Type;
    using Major = typename TypeParam::Major;
    using Mat = Matrix<Major, bool>;

    Mat M1 = Mat({{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 1, 1}});
    Check(M1.bany(2, 2), Mat({{0, 1}, {1, 1}}));
    Check(M1.rany(), Mat({0, 1, 1, 1}));
    Check(M1.cany(), Mat({{0, 1, 1, 1}}));
    Check(M1.any(), 1);
}

TYPED_TEST(Matrix1, Flip)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
    Check(M1.bflip(1, 2), Mat({{2, 1, 4, 3}, {6, 5, 8, 7}, {10, 9, 12, 11}, {14, 13, 16, 15}}));
    Check(M1.bflip(2, 2), Mat({{6, 5, 8, 7}, {2, 1, 4, 3}, {14, 13, 16, 15}, {10, 9, 12, 11}}));
    Check(M1.rflip(), Mat({{4, 3, 2, 1}, {8, 7, 6, 5}, {12, 11, 10, 9}, {16, 15, 14, 13}}));
    Check(M1.cflip(), Mat({{13, 14, 15, 16}, {9, 10, 11, 12}, {5, 6, 7, 8}, {1, 2, 3, 4}}));
    Check(M1.flip(), Mat({{16, 15, 14, 13}, {12, 11, 10, 9}, {8, 7, 6, 5}, {4, 3, 2, 1}}));
}

TYPED_TEST(Matrix1, Init)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Check(Mat::eye(3), Mat({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}));
    Check(Mat::eye(3, 4), Mat({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}));
    Check(Mat::eye(4, 3), Mat({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}}));
    Check(Mat::seq(2, 3, 1, 7), Mat({{1, 2, 3}, {4, 5, 6}}));
    // TODO static
}

TYPED_TEST(Matrix3, Complex)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;
    using FMat = Matrix<Major, typename Type::value_type>;

    Mat M1(2, 3, Type(4, 0));
    Check(M1.abs(), FMat(2, 3, 4));
    Check(M1.conj(), M1);
    Check(M1.proj(), Mat(2, 3, std::proj(Type(4, 0))));
    Check(M1.polar(), Mat(2, 3, std::polar(4, 0)));
    Check(M1.arg(), FMat(2, 3, std::arg(Type(4, 0))));
    Check(M1.norm(), FMat(2, 3, std::norm(Type(4, 0))));
    Check(M1.real(), FMat(2, 3, 4));
    Check(M1.imag(), FMat({{0, 0, 0}, {0, 0, 0}}));
    Check(M1.template type<typename Type::value_type>(), M1.real().eval());
}

TYPED_TEST(Matrix1, Other)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;
    using BMat = Matrix<Major, bool>;
    using FMat = Matrix<Major, float>;
    using CMat = Matrix<Major, std::complex<Type>>;

    Mat M1 = Mat({{1, 2, 3}, {4, 5, 6}}), M2 = Mat({{1, 2, 3}, {4, 5, 7}});
    FMat M3 = FMat::uniform(2, 3), M4 = FMat::uniform(3, 4);
    Mat M5 = Mat({{1, 2, 3}}), M6 = Mat({{3, 2, 1}}), M7 = Mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    Check(M1.template pow<3>(), Mat({{1, 8, 27}, {64, 125, 216}}));
    Check(M1.equal(M1), 1);
    Check(M1.equal(M2), 0);
    Check(M1.clip(2, 5), Mat({{2, 2, 3}, {4, 5, 5}}));
    Check(M1.expr().reshape(3, 0), Mat({{1, 2}, {3, 4}, {5, 6}}));
    Check(M3._mm(M4), M3.mm(M4));
    Check(M4.trans()._mm(M3.trans()), M4.trans().mm(M3.trans()));
    Check(M5.dot(M6), 10);
    Check(M5.trans().asdiag(), Mat({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}));
    Check(M7.trace(), 15);
    Check(M1.trans(), Mat({{1, 4}, {2, 5}, {3, 6}}));
    Check(M4.expr(), M4);
    Check(M1.template type<float>(), FMat({{1, 2, 3}, {4, 5, 6}}));
    Check(M1.template type<std::complex<Type>>(), CMat({{1, 2, 3}, {4, 5, 6}}));

    M1.where([](auto val1) { return val1 % 2 == 0; }) = 1;
    Check(M1, Mat({{1, 1, 3}, {1, 5, 1}}));
    M1 = Mat({{1, 2, 3}, {4, 5, 6}});
    Check(M1.where([](auto val1) { return val1 % 2 == 0; }), Mat({{0, 2, 0}, {4, 0, 6}}));
}

TYPED_TEST(Matrix1, Immediate)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat({{1, 2, 3}, {4, 5, 6}}), M2 = Mat({{-1, -2, -3}, {-4, -5, -6}});
    M1.rshuffle();
    M1.cshuffle();
    M1.shuffle();
    M1 = Mat({{4, 2, 6}, {3, 1, 5}});
    Check(M1.rsort(), Mat({{2, 4, 6}, {1, 3, 5}}));
    Check(M1.csort(), Mat({{3, 1, 5}, {4, 2, 6}}));
    Check(M1.sort(), Mat({{1, 2, 3}, {4, 5, 6}}));
}

TYPED_TEST(Matrix2, Blas)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;

    Mat M1 = Mat::uniform(2, 3), M2 = Mat::uniform(3, 4), M3 = Mat::uniform(3, 3), M4 = Mat::uniform(3, 3);

    Check(M1._mm(M2), M1.mm(M2));
    Check(M2.trans()._mm(M1.trans()), M2.trans().mm(M1.trans()));
    Check(M1._mm(M3.trans()), M1.mm(M3.trans()));
    Check(M3.trans()._mm(M2), M3.trans().mm(M2));

    Check(M3.usymm()._mm(M2), M3.usymm().mm(M2));
    Check(M3.lsymm()._mm(M2), M3.lsymm().mm(M2));
    Check(M1._mm(M3.usymm()), M1.mm(M3.usymm()));
    Check(M1._mm(M3.lsymm()), M1.mm(M3.lsymm()));

    Check(M3.upper()._mm(M2), M3.upper().mm(M2));
    Check(M3.lower()._mm(M2), M3.lower().mm(M2));
    Check(M1._mm(M3.upper()), M1.mm(M3.upper()));
    Check(M1._mm(M3.lower()), M1.mm(M3.lower()));

    Check(M3.oupper()._mm(M2), M3.oupper().mm(M2));
    Check(M3.olower()._mm(M2), M3.olower().mm(M2));
    Check(M1._mm(M3.oupper()), M1.mm(M3.oupper()));
    Check(M1._mm(M3.olower()), M1.mm(M3.olower()));

    Check(M3.usymm().trans()._mm(M2), M3.usymm().trans().mm(M2));
    Check(M3.lsymm().trans()._mm(M2), M3.lsymm().trans().mm(M2));
    Check(M1._mm(M3.usymm().trans()), M1.mm(M3.usymm().trans()));
    Check(M1._mm(M3.lsymm().trans()), M1.mm(M3.lsymm().trans()));

    Check(M3.trans().usymm()._mm(M2), M3.trans().usymm().mm(M2));
    Check(M3.trans().lsymm()._mm(M2), M3.trans().lsymm().mm(M2));
    Check(M1._mm(M3.trans().usymm()), M1.mm(M3.trans().usymm()));
    Check(M1._mm(M3.trans().lsymm()), M1.mm(M3.trans().lsymm()));

    Check(M3.upper().trans()._mm(M2), M3.upper().trans().mm(M2));
    Check(M3.lower().trans()._mm(M2), M3.lower().trans().mm(M2));
    Check(M1._mm(M3.upper().trans()), M1.mm(M3.upper().trans()));
    Check(M1._mm(M3.lower().trans()), M1.mm(M3.lower().trans()));

    Check(M3.trans().upper()._mm(M2), M3.trans().upper().mm(M2));
    Check(M3.trans().lower()._mm(M2), M3.trans().lower().mm(M2));
    Check(M1._mm(M3.trans().upper()), M1.mm(M3.trans().upper()));
    Check(M1._mm(M3.trans().lower()), M1.mm(M3.trans().lower()));
}

TYPED_TEST(Matrix4, BlasComp)
{
    using Mat = TypeParam;
    using Type = typename Mat::Type;
    using Major = typename Mat::Major;
    using FMat = Matrix<Major, typename Type::value_type>;

    Mat M1 = FMat::uniform(2, 3).template type<Type>(), M2 = FMat::uniform(3, 4).template type<Type>(),
        M3 = FMat::uniform(3, 3).template type<Type>(), M4 = FMat::uniform(3, 3).template type<Type>();

    Check(M3.conj().upper().trans()._mm(M2), M3.conj().upper().trans().mm(M2));
    Check(M3.conj().lower().trans()._mm(M2), M3.conj().lower().trans().mm(M2));
    Check(M1._mm(M3.conj().upper().trans()), M1.mm(M3.conj().upper().trans()));
    Check(M1._mm(M3.conj().lower().trans()), M1.mm(M3.conj().lower().trans()));

    Check(M3.conj().trans().upper()._mm(M2), M3.conj().trans().upper().mm(M2));
    Check(M3.conj().trans().lower()._mm(M2), M3.conj().trans().lower().mm(M2));
    Check(M1._mm(M3.conj().trans().upper()), M1.mm(M3.conj().trans().upper()));
    Check(M1._mm(M3.conj().trans().lower()), M1.mm(M3.conj().trans().lower()));

    Check(M3.upper().trans().conj()._mm(M2), M3.upper().trans().conj().mm(M2));
    Check(M3.lower().trans().conj()._mm(M2), M3.lower().trans().conj().mm(M2));
    Check(M1._mm(M3.upper().trans().conj()), M1.mm(M3.upper().trans().conj()));
    Check(M1._mm(M3.lower().trans().conj()), M1.mm(M3.lower().trans().conj()));

    Check(M3.trans().upper().conj()._mm(M2), M3.trans().upper().conj().mm(M2));
    Check(M3.trans().lower().conj()._mm(M2), M3.trans().lower().conj().mm(M2));
    Check(M1._mm(M3.trans().upper().conj()), M1.mm(M3.trans().upper().conj()));
    Check(M1._mm(M3.trans().lower().conj()), M1.mm(M3.trans().lower().conj()));
}
