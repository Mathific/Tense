#include <gtest/gtest.h>
#include <tense/tense.h>

#include <cmath>
#include <complex>
#include <sstream>
#include <type_traits>
#include <vector>

using namespace Tense;

#define TENSE_ALIGNMENT 64

// ============================================================================
// 1. FIXTURES AND TYPE DEFINITIONS
// ============================================================================

template <typename T>
class VectorTestBase : public ::testing::Test
{
protected:
    T get_val() { return T(42); }

    Vector<T> make_sequence(Size size)
    {
        Vector<T> v(size);
        for (Size i = 0; i < size; ++i) v[i] = T(i + 1);
        return v;
    }

    Vector<T> make_vec(T v1, T v2, T v3) { return Vector<T>{v1, v2, v3}; }
};

// Define the type lists
using AllTypes = ::testing::Types<int32_t, float, double, std::complex<float>>;
using RealTypes = ::testing::Types<int32_t, float, double>;  // Excludes complex
using IntTypes = ::testing::Types<int32_t, int64_t>;
using FloatTypes = ::testing::Types<float, double>;
using ComplexTypes = ::testing::Types<std::complex<float>, std::complex<double>>;

// Define the targeted test suites
template <typename T>
class VectorTest : public VectorTestBase<T>
{
};
template <typename T>
class VectorRealTest : public VectorTestBase<T>
{
};
template <typename T>
class VectorIntTest : public VectorTestBase<T>
{
};
template <typename T>
class VectorFloatTest : public VectorTestBase<T>
{
};
template <typename T>
class VectorComplexTest : public VectorTestBase<T>
{
};

// Register the suites with Google Test
TYPED_TEST_SUITE(VectorTest, AllTypes);
TYPED_TEST_SUITE(VectorRealTest, RealTypes);
TYPED_TEST_SUITE(VectorIntTest, IntTypes);
TYPED_TEST_SUITE(VectorFloatTest, FloatTypes);
TYPED_TEST_SUITE(VectorComplexTest, ComplexTypes);

// ============================================================================
// 2. LIFECYCLE & MEMORY
// ============================================================================

TYPED_TEST(VectorTest, DefaultConstructorIsInvalid)
{
    Vector<TypeParam> v;
    EXPECT_FALSE(v.valid());
    EXPECT_EQ(v.memory(), sizeof(std::shared_ptr<void>));  // Base memory for invalid
}

TYPED_TEST(VectorTest, SizeConstructorInitializesToZero)
{
    Vector<TypeParam> v(5);
    EXPECT_TRUE(v.valid());
    EXPECT_EQ(v.size(), 5);
    for (Size i = 0; i < 5; ++i) EXPECT_EQ(v[i], TypeParam(0));
}

TYPED_TEST(VectorTest, ValueConstructorInitializesToValue)
{
    TypeParam val = this->get_val();
    Vector<TypeParam> v(3, val);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v[0], val);
    EXPECT_EQ(v[2], val);
}

TYPED_TEST(VectorTest, InitializerListConstructor)
{
    Vector<TypeParam> v = {TypeParam(1), TypeParam(2), TypeParam(3)};
    EXPECT_TRUE(v.valid());
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v[0], TypeParam(1));
    EXPECT_EQ(v[2], TypeParam(3));
}

TYPED_TEST(VectorTest, CopyConstructor)
{
    Vector<TypeParam> v1 = {TypeParam(1), TypeParam(2)};
    Vector<TypeParam> v2(v1);
    EXPECT_TRUE(v2.valid());
    EXPECT_EQ(v2.size(), 2);
    EXPECT_EQ(v2[1], TypeParam(2));

    // Check they share data (default copy is shallow due to shared_ptr)
    v1[0] = TypeParam(99);
    EXPECT_EQ(v2[0], TypeParam(99));
}

TYPED_TEST(VectorTest, ExplicitCopyIsDeep)
{
    Vector<TypeParam> v1 = {TypeParam(1), TypeParam(2)};
    Vector<TypeParam> v2 = v1.copy();
    EXPECT_TRUE(v2.valid());

    // Modifying v1 should not affect v2
    v1[0] = TypeParam(99);
    EXPECT_EQ(v2[0], TypeParam(1));
}

TYPED_TEST(VectorTest, CopyAssignmentOperator)
{
    Vector<TypeParam> v1 = {TypeParam(1), TypeParam(2)};
    Vector<TypeParam> v2;
    v2 = v1;
    EXPECT_TRUE(v2.valid());
    EXPECT_EQ(v2.size(), 2);
    EXPECT_EQ(v2[0], TypeParam(1));
}

TYPED_TEST(VectorTest, AssignmentFromScalar)
{
    Vector<TypeParam> v(3);
    v = TypeParam(5);
    EXPECT_EQ(v[0], TypeParam(5));
    EXPECT_EQ(v[2], TypeParam(5));
}

TYPED_TEST(VectorTest, ModeCopyIsIndependent)
{
    TypeParam raw_data[] = {TypeParam(1), TypeParam(2), TypeParam(3)};
    Vector<TypeParam> v(3, raw_data, Mode::Copy);
    raw_data[0] = TypeParam(99);
    EXPECT_EQ(v[0], TypeParam(1));
}

TYPED_TEST(VectorTest, ModeHoldIsDependent)
{
    TypeParam raw_data[] = {TypeParam(1), TypeParam(2), TypeParam(3)};
    Vector<TypeParam> v(3, raw_data, Mode::Hold);
    raw_data[0] = TypeParam(99);
    EXPECT_EQ(v[0], TypeParam(99));
}

TYPED_TEST(VectorTest, ReleaseTransfersOwnership)
{
    Vector<TypeParam> v = {TypeParam(1), TypeParam(2)};
    TypeParam* ptr = v.release();

    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(ptr[0], TypeParam(1));
    EXPECT_FALSE(v.valid() && v.size() > 0 && v.data() != nullptr);

    ::operator delete[](ptr, std::align_val_t(TENSE_ALIGNMENT));
}

TYPED_TEST(VectorTest, ResizeExpandsAndPreservesData)
{
    Vector<TypeParam> v = {TypeParam(1), TypeParam(2)};
    v.resize(4);
    EXPECT_EQ(v.size(), 4);
    EXPECT_EQ(v[0], TypeParam(1));
    EXPECT_EQ(v[2], TypeParam(0));
}

TYPED_TEST(VectorTest, ResetInvalidatesVector)
{
    Vector<TypeParam> v(5);
    EXPECT_TRUE(v.valid());
    v.reset();
    EXPECT_FALSE(v.valid());
}

// ============================================================================
// 3. ELEMENT ACCESS & SLICING
// ============================================================================

TYPED_TEST(VectorTest, ReadWriteOperators)
{
    auto v = this->make_sequence(3);
    EXPECT_EQ(v[0], TypeParam(1));
    EXPECT_EQ(v(1), TypeParam(2));  // Test operator()

    v[0] = TypeParam(99);
    v(1) = TypeParam(100);
    EXPECT_EQ(v[0], TypeParam(99));
    EXPECT_EQ(v[1], TypeParam(100));
}

TYPED_TEST(VectorTest, IteratorsWorkWithStandardLibrary)
{
    auto v = this->make_sequence(3);
    std::vector<TypeParam> std_v(v.begin(), v.end());
    EXPECT_EQ(std_v.size(), 3);
    EXPECT_EQ(std_v[2], TypeParam(3));
}

TYPED_TEST(VectorTest, BlockReadWrite)
{
    auto v = this->make_sequence(5);
    auto b = v.block(1, 3);

    EXPECT_EQ(b.size(), 3);
    EXPECT_EQ(b[0], TypeParam(2));
    b[0] = TypeParam(99);
    EXPECT_EQ(v[1], TypeParam(99));
}

TYPED_TEST(VectorTest, LeftAndRightViews)
{
    auto v = this->make_sequence(4);
    auto l = v.left(2);
    auto r = v.right(2);
    l[0] = TypeParam(10);
    r[1] = TypeParam(40);
    EXPECT_EQ(v[0], TypeParam(10));
    EXPECT_EQ(v[3], TypeParam(40));
}

TYPED_TEST(VectorTest, IndexByCutPositiveStep)
{
    auto v = this->make_sequence(5);
    auto sliced = v.index(Cut(0, 2, 5));
    EXPECT_EQ(sliced.size(), 3);
    EXPECT_EQ(sliced[1], TypeParam(3));
}

TYPED_TEST(VectorTest, IndexByCutNegativeStep)
{
    auto v = this->make_sequence(5);
    auto sliced = v.index(Cut(4, -1, -1));
    EXPECT_EQ(sliced.size(), 5);
    EXPECT_EQ(sliced[0], TypeParam(5));
}

TYPED_TEST(VectorTest, IndexByVector)
{
    auto v = this->make_sequence(5);
    std::vector<Size> indices = {4, 1, 0};
    auto gathered = v.index(indices);
    EXPECT_EQ(gathered.size(), 3);
    EXPECT_EQ(gathered[0], TypeParam(5));
    EXPECT_EQ(gathered[2], TypeParam(1));
}

TYPED_TEST(VectorIntTest, RIndirect)
{
    auto v = this->make_sequence(4);  // [1, 2, 3, 4]
    Vector<TypeParam> indices = {3, 2, 1, 0};
    auto indirect_view = v.rindirect(indices);
    EXPECT_EQ(indirect_view.size(), 4);
    EXPECT_EQ(indirect_view[0], TypeParam(4));
    EXPECT_EQ(indirect_view[3], TypeParam(1));
}

TYPED_TEST(VectorTest, Cat0AppendsVectors)
{
    Vector<TypeParam> v1 = {TypeParam(1), TypeParam(2)};
    Vector<TypeParam> v2 = {TypeParam(3)};
    auto concat = v1.cat0(v2);
    EXPECT_EQ(concat.size(), 3);
    EXPECT_EQ(concat[0], TypeParam(1));
    EXPECT_EQ(concat[2], TypeParam(3));
}

TYPED_TEST(VectorTest, Cat1InterleavesVectors)
{
    Vector<TypeParam> v1 = {TypeParam(1), TypeParam(3)};
    Vector<TypeParam> v2 = {TypeParam(2), TypeParam(4)};
    auto interleaved = v1.cat1(v2);
    EXPECT_EQ(interleaved.size(), 4);
    EXPECT_EQ(interleaved[0], TypeParam(1));
    EXPECT_EQ(interleaved[1], TypeParam(2));
    EXPECT_EQ(interleaved[2], TypeParam(3));
    EXPECT_EQ(interleaved[3], TypeParam(4));
}

TYPED_TEST(VectorTest, SingleElementView)
{
    auto v = this->make_sequence(3);
    auto el = v.elem(1);
    EXPECT_EQ(el.size(), 1);
    EXPECT_EQ(el[0], TypeParam(2));
    el[0] = TypeParam(99);
    EXPECT_EQ(v[1], TypeParam(99));
}

// ============================================================================
// 4. ARITHMETIC & EXPRESSION TEMPLATES
// ============================================================================

TYPED_TEST(VectorTest, Addition)
{
    auto v1 = this->make_vec(TypeParam(2), TypeParam(2), TypeParam(2));
    auto v2 = this->make_vec(TypeParam(3), TypeParam(3), TypeParam(3));

    Vector<TypeParam> sum = v1 + v2;
    EXPECT_EQ(sum[0], TypeParam(5));

    Vector<TypeParam> sum_scalar = v1 + TypeParam(10);
    EXPECT_EQ(sum_scalar[0], TypeParam(12));

    Vector<TypeParam> rev_sum = TypeParam(10) + v1;
    EXPECT_EQ(rev_sum[0], TypeParam(12));
}

TYPED_TEST(VectorTest, SubtractionAndMultiplication)
{
    auto v1 = this->make_vec(TypeParam(5), TypeParam(5), TypeParam(5));
    auto v2 = this->make_vec(TypeParam(2), TypeParam(2), TypeParam(2));

    Vector<TypeParam> sub = v1 - v2;
    EXPECT_EQ(sub[0], TypeParam(3));

    Vector<TypeParam> mul = v1 * v2;
    EXPECT_EQ(mul[0], TypeParam(10));
}

TYPED_TEST(VectorTest, CompoundAssignment)
{
    auto v1 = this->make_vec(TypeParam(5), TypeParam(5), TypeParam(5));
    auto v2 = this->make_vec(TypeParam(2), TypeParam(2), TypeParam(2));

    v1 += v2;
    EXPECT_EQ(v1[0], TypeParam(7));
    v1 *= TypeParam(2);
    EXPECT_EQ(v1[0], TypeParam(14));
}

TYPED_TEST(VectorTest, LazyEvaluationTypeCheck)
{
    auto v1 = this->make_vec(TypeParam(1), TypeParam(1), TypeParam(1));
    auto v2 = this->make_vec(TypeParam(2), TypeParam(2), TypeParam(2));

    auto expr = v1 + v2;

    bool is_vector = std::is_same<decltype(expr), Vector<TypeParam>>::value;
    EXPECT_FALSE(is_vector);

    bool is_expr_base = std::is_base_of<VectorImpl::Expr, decltype(expr)>::value;
    EXPECT_TRUE(is_expr_base);

    Vector<TypeParam> evaluated = expr.eval();
    EXPECT_EQ(evaluated[0], TypeParam(3));
}

TYPED_TEST(VectorTest, ConstructFromExpression)
{
    Vector<TypeParam> v1 = {TypeParam(1), TypeParam(2)};
    Vector<TypeParam> v2 = {TypeParam(3), TypeParam(4)};

    Vector<TypeParam> v3(v1 + v2);
    EXPECT_TRUE(v3.valid());
    EXPECT_EQ(v3.size(), 2);
    EXPECT_EQ(v3[0], TypeParam(4));
}

TYPED_TEST(VectorTest, AssignExpressionToView)
{
    auto v1 = this->make_sequence(4);
    Vector<TypeParam> v2 = {TypeParam(10), TypeParam(20)};

    v1.index(Cut(0, 2, 4)) = (v2 * TypeParam(2));
    EXPECT_EQ(v1[0], TypeParam(20));
    EXPECT_EQ(v1[1], TypeParam(2));
    EXPECT_EQ(v1[2], TypeParam(40));
}

// ============================================================================
// 5. MATH OPERATIONS & CONSTANTS
// ============================================================================

TYPED_TEST(VectorIntTest, ModuloAndBitwise)
{
    Vector<TypeParam> v1(3, TypeParam(10));
    Vector<TypeParam> v2(3, TypeParam(3));

    Vector<TypeParam> mod = v1 % v2;
    EXPECT_EQ(mod[0], TypeParam(1));

    Vector<TypeParam> bit_and = v1 & v2;
    EXPECT_EQ(bit_and[0], TypeParam(2));

    Vector<TypeParam> bit_or = v1 | v2;
    EXPECT_EQ(bit_or[0], TypeParam(11));

    Vector<TypeParam> bit_xor = v1 ^ v2;
    EXPECT_EQ(bit_xor[0], TypeParam(9));

    Vector<TypeParam> lshift = v1 << TypeParam(1);
    EXPECT_EQ(lshift[0], TypeParam(20));

    Vector<TypeParam> rshift = v1 >> TypeParam(1);
    EXPECT_EQ(rshift[0], TypeParam(5));
}

TYPED_TEST(VectorFloatTest, UnaryTrigonometryAndPowers)
{
    Vector<TypeParam> v(3, TypeParam(0.0));

    Vector<TypeParam> cos_v = v.cos();
    EXPECT_NEAR(cos_v[0], TypeParam(1.0), 1e-5);

    Vector<TypeParam> sin_v = v.sin();
    EXPECT_NEAR(sin_v[0], TypeParam(0.0), 1e-5);

    Vector<TypeParam> exp_v = v.exp();
    EXPECT_NEAR(exp_v[0], TypeParam(1.0), 1e-5);

    Vector<TypeParam> exp2_v = v.exp2();
    EXPECT_NEAR(exp2_v[0], TypeParam(1.0), 1e-5);
}

TYPED_TEST(VectorFloatTest, AdvancedMathFuncs)
{
    Vector<TypeParam> v = {TypeParam(1.0), TypeParam(2.0), TypeParam(3.0)};

    auto ln_v = v.ln().eval();
    EXPECT_NEAR(ln_v[0], TypeParam(0.0), 1e-5);

    auto cbrt_v = v.cbrt().eval();
    EXPECT_NEAR(cbrt_v[0], TypeParam(1.0), 1e-5);

    auto square_v = v.square().eval();
    EXPECT_NEAR(square_v[1], TypeParam(4.0), 1e-5);

    auto cube_v = v.cube().eval();
    EXPECT_NEAR(cube_v[1], TypeParam(8.0), 1e-5);
}

TYPED_TEST(VectorRealTest, BinaryMathAndClip)
{
    Vector<TypeParam> v1(3, TypeParam(2.0));
    Vector<TypeParam> v2(3, TypeParam(3.0));

    Vector<TypeParam> p = v1.pow(v2);
    EXPECT_NEAR(p[0], TypeParam(8.0), 1e-5);

    Vector<TypeParam> v3 = {TypeParam(-1), TypeParam(2), TypeParam(10)};
    Vector<TypeParam> clipped = v3.clip(TypeParam(0), TypeParam(5));
    EXPECT_NEAR(clipped[0], TypeParam(0), 1e-5);
    EXPECT_NEAR(clipped[1], TypeParam(2), 1e-5);
    EXPECT_NEAR(clipped[2], TypeParam(5), 1e-5);
}

TYPED_TEST(VectorFloatTest, ReluAndSigmoid)
{
    auto v = this->make_vec(TypeParam(-2.0), TypeParam(0.0), TypeParam(2.0));

    auto relu_v = v.relu().eval();
    EXPECT_NEAR(relu_v[0], TypeParam(0.0), 1e-5);
    EXPECT_NEAR(relu_v[2], TypeParam(2.0), 1e-5);

    auto sig_v = v.sigmoid().eval();
    EXPECT_NEAR(sig_v[1], TypeParam(0.5), 1e-5);
}

TYPED_TEST(VectorRealTest, AbsAndNegation)
{
    Vector<TypeParam> v = {TypeParam(-5), TypeParam(3)};
    auto a = v.abs().eval();
    EXPECT_EQ(a[0], TypeParam(5));

    auto n = (-v).eval();
    EXPECT_EQ(n[0], TypeParam(5));
    EXPECT_EQ(n[1], TypeParam(-3));
}

TYPED_TEST(VectorComplexTest, RealAndImaginaryExtraction)
{
    using BaseType = typename TypeParam::value_type;
    Vector<TypeParam> v(3, TypeParam(BaseType(3.0), BaseType(4.0)));

    Vector<BaseType> r = v.real();
    Vector<BaseType> i = v.imag();

    EXPECT_FLOAT_EQ(r[0], BaseType(3.0));
    EXPECT_FLOAT_EQ(i[0], BaseType(4.0));
}

TYPED_TEST(VectorComplexTest, ComplexNormAndPolar)
{
    using BaseType = typename TypeParam::value_type;
    Vector<TypeParam> v(3, TypeParam(BaseType(3.0), BaseType(4.0)));

    Vector<BaseType> n = v.norm();
    EXPECT_FLOAT_EQ(n[0], BaseType(25.0));

    auto polar_v = v.polar().eval();
    auto expected = std::polar(BaseType(5.0), std::atan2(BaseType(4.0), BaseType(3.0)));
    EXPECT_FLOAT_EQ(polar_v[0].real(), expected.real());
}
// ============================================================================
// 6. REDUCTIONS, STATISTICS & NORMS
// ============================================================================

TYPED_TEST(VectorTest, SumProductMinMax)
{
    auto v = this->make_sequence(4);  // 1, 2, 3, 4
    EXPECT_EQ(v.sum(), TypeParam(10));
    EXPECT_EQ(v.prod(), TypeParam(24));

    // count elements equal to 2 (requires matching type or scalar conversion)
    EXPECT_EQ(v.count(TypeParam(2)), Size(1));
    EXPECT_EQ(v.count(TypeParam(99)), Size(0));
}

TYPED_TEST(VectorRealTest, MinMaxIndices)
{
    Vector<TypeParam> v = {TypeParam(4), TypeParam(9), TypeParam(1), TypeParam(9)};
    EXPECT_EQ(v.max(), TypeParam(9));
    EXPECT_EQ(v.min(), TypeParam(1));

    // Should return the index of the first maximum/minimum
    auto max_idx = v.maxidx();
    EXPECT_EQ(max_idx, Size(1));

    auto min_idx = v.minidx();
    EXPECT_EQ(min_idx, Size(2));
}

TYPED_TEST(VectorFloatTest, MeanVarianceAndStdDev)
{
    Vector<TypeParam> v = {TypeParam(2), TypeParam(4), TypeParam(4), TypeParam(4),
                           TypeParam(5), TypeParam(5), TypeParam(7), TypeParam(9)};

    EXPECT_NEAR(v.mean(), TypeParam(5.0), 1e-5);
    EXPECT_NEAR(v.var(0), TypeParam(4.0), 1e-5);      // Population variance
    EXPECT_NEAR(v.var(1), TypeParam(4.57142), 1e-4);  // Sample variance (dof=1)
    EXPECT_NEAR(v.std(0), TypeParam(2.0), 1e-5);
    EXPECT_NEAR(v.std(1), TypeParam(2.13808), 1e-4);
}

TYPED_TEST(VectorFloatTest, BlockStatistics)
{
    // 6 elements, block size 3. Blocks: [2, 4, 6] and [8, 10, 12]
    Vector<TypeParam> v = {TypeParam(2), TypeParam(4), TypeParam(6), TypeParam(8), TypeParam(10), TypeParam(12)};

    auto b_mean = v.bmean(3).eval();
    EXPECT_EQ(b_mean.size(), 2);
    EXPECT_NEAR(b_mean[0], TypeParam(4.0), 1e-5);
    EXPECT_NEAR(b_mean[1], TypeParam(10.0), 1e-5);

    auto b_var = v.bvar(3, 0).eval();
    EXPECT_EQ(b_var.size(), 2);
    EXPECT_NEAR(b_var[0], TypeParam(2.66667), 1e-4);
    EXPECT_NEAR(b_var[1], TypeParam(2.66667), 1e-4);
}

TYPED_TEST(VectorFloatTest, Covariance)
{
    Vector<TypeParam> v1 = {TypeParam(1), TypeParam(2), TypeParam(3)};
    Vector<TypeParam> v2 = {TypeParam(1), TypeParam(2), TypeParam(3)};

    EXPECT_NEAR(v1.cov(v2, 0), TypeParam(0.6666667), 1e-5);  // biased sample covariance
    EXPECT_NEAR(v1.cov(v2, 1), TypeParam(1.0), 1e-5);        // unbiased sample covariance
}

TYPED_TEST(VectorFloatTest, NormsAndDistances)
{
    Vector<TypeParam> v = {TypeParam(-3), TypeParam(4)};
    EXPECT_NEAR(v.template norm<1>(), TypeParam(7.0), 1e-5);           // Manhattan
    EXPECT_NEAR(v.template norm<2>(), TypeParam(5.0), 1e-5);           // Euclidean
    EXPECT_NEAR(v.template norm<Tense::Inf>(), TypeParam(4.0), 1e-5);  // Infinity

    Vector<TypeParam> v1 = {TypeParam(0), TypeParam(0)};
    Vector<TypeParam> v2 = {TypeParam(3), TypeParam(-4)};
    EXPECT_NEAR(v1.template distance<2>(v2), TypeParam(5.0), 1e-5);
    EXPECT_NEAR(v1.template distance<1>(v2), TypeParam(7.0), 1e-5);
}

TYPED_TEST(VectorFloatTest, BlockNormsAndDistances)
{
    Vector<TypeParam> v = {TypeParam(3), TypeParam(4), TypeParam(6), TypeParam(8)};
    auto b_norm2 = v.template bnorm<2>(2).eval();  // L2 norm of [3,4] and [6,8]
    EXPECT_EQ(b_norm2.size(), 2);
    EXPECT_NEAR(b_norm2[0], TypeParam(5.0), 1e-5);
    EXPECT_NEAR(b_norm2[1], TypeParam(10.0), 1e-5);
}

TYPED_TEST(VectorTest, DotProduct)
{
    Vector<TypeParam> v1 = {TypeParam(1), TypeParam(2), TypeParam(3)};
    Vector<TypeParam> v2 = {TypeParam(2), TypeParam(3), TypeParam(4)};

    auto result = v1.dot(v2);
    EXPECT_EQ(result, TypeParam(20));  // 2 + 6 + 12
}

TYPED_TEST(VectorFloatTest, NormalizeAndStandize)
{
    Vector<TypeParam> v = {TypeParam(0), TypeParam(5), TypeParam(10)};

    auto norm_v = v.normalize().eval();
    EXPECT_NEAR(norm_v[0], TypeParam(0.0), 1e-5);
    EXPECT_NEAR(norm_v[1], TypeParam(0.5), 1e-5);
    EXPECT_NEAR(norm_v[2], TypeParam(1.0), 1e-5);

    auto std_v = v.standize(0).eval();
    // mean = 5, std = sqrt(16.666) ~= 4.08248
    EXPECT_NEAR(std_v[0], TypeParam(-1.2247), 1e-4);
    EXPECT_NEAR(std_v[1], TypeParam(0.0), 1e-4);
    EXPECT_NEAR(std_v[2], TypeParam(1.2247), 1e-4);
}

TYPED_TEST(VectorFloatTest, BlockNormalizeAndStandize)
{
    Vector<TypeParam> v = {TypeParam(0), TypeParam(10), TypeParam(5), TypeParam(15)};

    // Normalize blocks of size 2. [0, 10] -> [0, 1], [5, 15] -> [0, 1]
    auto b_norm = v.bnormalize(2).eval();
    EXPECT_NEAR(b_norm[0], TypeParam(0.0), 1e-5);
    EXPECT_NEAR(b_norm[1], TypeParam(1.0), 1e-5);
    EXPECT_NEAR(b_norm[2], TypeParam(0.0), 1e-5);
    EXPECT_NEAR(b_norm[3], TypeParam(1.0), 1e-5);
}

// ============================================================================
// 7. COMPARISONS & LOGIC
// ============================================================================

TYPED_TEST(VectorRealTest, RelationalOperators)
{
    auto v1 = this->make_vec(TypeParam(1), TypeParam(5), TypeParam(3));
    auto v2 = this->make_vec(TypeParam(1), TypeParam(99), TypeParam(3));

    Vector<bool> eq = (v1 == v2).eval();
    EXPECT_TRUE(eq[0]);
    EXPECT_FALSE(eq[1]);
    EXPECT_TRUE(eq[2]);

    Vector<bool> lt = (v1 < v2).eval();
    EXPECT_FALSE(lt[0]);
    EXPECT_TRUE(lt[1]);

    Vector<bool> ne = (v1 != v2).eval();
    EXPECT_FALSE(ne[0]);
    EXPECT_TRUE(ne[1]);
}

TYPED_TEST(VectorRealTest, ScalarRelationalOperators)
{
    auto v = this->make_vec(TypeParam(1), TypeParam(5), TypeParam(3));

    Vector<bool> gt = (v > TypeParam(2)).eval();
    EXPECT_FALSE(gt[0]);
    EXPECT_TRUE(gt[1]);
    EXPECT_TRUE(gt[2]);

    Vector<bool> lte = (v <= TypeParam(3)).eval();
    EXPECT_TRUE(lte[0]);
    EXPECT_FALSE(lte[1]);
    EXPECT_TRUE(lte[2]);
}

TYPED_TEST(VectorFloatTest, CloseTolerance)
{
    Vector<TypeParam> v1 = {TypeParam(1.0), TypeParam(2.0)};
    Vector<TypeParam> v2 = {TypeParam(1.000001), TypeParam(2.000009)};

    EXPECT_FALSE(v1.equal(v2));
    EXPECT_TRUE(v1.close(v2, TypeParam(1e-4)));
    EXPECT_TRUE(v1.close(TypeParam(1.000001), TypeParam(1.5)));  // Scalar close
}

TYPED_TEST(VectorRealTest, BooleanReductions)
{
    Vector<TypeParam> v1 = {TypeParam(0), TypeParam(1), TypeParam(0)};
    Vector<TypeParam> v2 = {TypeParam(1), TypeParam(2), TypeParam(3)};

    EXPECT_TRUE(v1.any());
    EXPECT_FALSE(v1.all());
    EXPECT_TRUE(v2.all());

    EXPECT_TRUE(v1.contains(TypeParam(1)));
    EXPECT_FALSE(v1.contains(TypeParam(99)));
}

TYPED_TEST(VectorRealTest, BlockBooleanReductions)
{
    Vector<TypeParam> v = {TypeParam(0), TypeParam(1), TypeParam(1), TypeParam(1)};

    auto b_any = v.bany(2).eval();  // [any(0,1), any(1,1)] -> [true, true]
    EXPECT_TRUE(b_any[0]);
    EXPECT_TRUE(b_any[1]);

    auto b_all = v.ball(2).eval();  // [all(0,1), all(1,1)] -> [false, true]
    EXPECT_FALSE(b_all[0]);
    EXPECT_TRUE(b_all[1]);

    auto b_contains = v.bcontains(2, TypeParam(0)).eval();
    EXPECT_TRUE(b_contains[0]);
    EXPECT_FALSE(b_contains[1]);
}

// ============================================================================
// 8. GENERATORS & DISTRIBUTIONS
// ============================================================================

TYPED_TEST(VectorRealTest, StaticGenerators)
{
    auto z = Vector<TypeParam>::zeros(5);
    EXPECT_EQ(z.sum(), TypeParam(0));

    auto o = Vector<TypeParam>::ones(3);
    EXPECT_EQ(o.sum(), TypeParam(3));

    auto s = Vector<TypeParam>::seq(5, TypeParam(0), TypeParam(10));
    EXPECT_EQ(s[4], TypeParam(8));  // step is (10-0)/5 = 2. indices 0: 0, 1: 2, 2: 4, 3: 6, 4: 8
}

TYPED_TEST(VectorFloatTest, MathConstants)
{
    auto pi_v = Vector<TypeParam>::pi(2);
    EXPECT_NEAR(pi_v[0], TypeParam(M_PI), 1e-5);

    auto e_v = Vector<TypeParam>::e(2);
    EXPECT_NEAR(e_v[0], TypeParam(M_E), 1e-5);

    auto sqrt2_v = Vector<TypeParam>::sqrt2(2);
    EXPECT_NEAR(sqrt2_v[0], TypeParam(M_SQRT2), 1e-5);
}

TYPED_TEST(VectorFloatTest, LimitsConstants)
{
    auto inf_v = Vector<TypeParam>::inf(1);
    EXPECT_TRUE(std::isinf(inf_v[0]));

    auto nan_v = Vector<TypeParam>::nan(1);
    EXPECT_TRUE(std::isnan(nan_v[0]));

    auto max_v = Vector<TypeParam>::numax(1);
    EXPECT_EQ(max_v[0], std::numeric_limits<TypeParam>::max());
}

TYPED_TEST(VectorRealTest, UniformDistribution)
{
    auto u = Vector<TypeParam>::uniform(100, TypeParam(0), TypeParam(10));
    EXPECT_GE(std::abs(u.min()), 0);
    EXPECT_LE(std::abs(u.max()), 10);
}

TYPED_TEST(VectorFloatTest, StatisticalDistributions)
{
    auto norm = Vector<TypeParam>::normal(10000, 5.0, 2.0);
    EXPECT_NEAR(static_cast<double>(norm.mean()), 5.0, 0.1);
    EXPECT_NEAR(static_cast<double>(norm.std(1)), 2.0, 0.1);

    auto exp_dist = Vector<TypeParam>::exponential(100, 1.0);
    EXPECT_GE(exp_dist.min(), TypeParam(0));

    auto poisson = Vector<TypeParam>::poisson(100, 4.0);
    EXPECT_GE(poisson.min(), TypeParam(0));
}

// ============================================================================
// 9. ALGORITHMS & MUTATORS
// ============================================================================

TYPED_TEST(VectorRealTest, SortAndShuffle)
{
    Vector<TypeParam> v = {TypeParam(4), TypeParam(1), TypeParam(3), TypeParam(2)};
    auto asc = v.sort();
    EXPECT_EQ(asc[0], TypeParam(1));
    EXPECT_EQ(asc[3], TypeParam(4));

    // Custom comparator sort (Descending)
    auto desc = v.sort([](auto a, auto b) { return a > b; });
    EXPECT_EQ(desc[0], TypeParam(4));
    EXPECT_EQ(desc[3], TypeParam(1));

    auto expected_sum = v.sum();
    auto shuffled = v.shuffle();
    EXPECT_EQ(shuffled.sum(), expected_sum);
}

TYPED_TEST(VectorTest, ConditionalsWhere)
{
    auto v = this->make_sequence(5);
    auto filtered = v.where([](auto val) { return std::abs(val) > 3; }, TypeParam(0)).eval();
    EXPECT_EQ(filtered[0], TypeParam(0));
    EXPECT_EQ(filtered[4], TypeParam(5));

    auto idx_filtered = v.iwhere([](auto i) { return i % 2 == 0; }, TypeParam(-1)).eval();
    EXPECT_EQ(idx_filtered[0], TypeParam(1));
    EXPECT_EQ(idx_filtered[1], TypeParam(-1));
}

TYPED_TEST(VectorTest, WhereWithExpression)
{
    auto v1 = this->make_sequence(3);      // 1, 2, 3
    auto v2 = Vector<TypeParam>::ones(3);  // 1, 1, 1

    auto fe_where = v1.where([](auto val) { return std::abs(val) > 1; }, v2).eval();
    EXPECT_EQ(fe_where[0], TypeParam(1));  // takes from v2
    EXPECT_EQ(fe_where[1], TypeParam(2));  // takes from v1
}

TYPED_TEST(VectorTest, LayoutTransformations)
{
    auto v = this->make_sequence(4);

    auto flipped = v.flip().eval();
    EXPECT_EQ(flipped[0], TypeParam(4));
    EXPECT_EQ(flipped[3], TypeParam(1));

    auto turned = v.turn(1).eval();  // Right shift by 1
    EXPECT_EQ(turned[0], TypeParam(4));
    EXPECT_EQ(turned[1], TypeParam(1));

    auto bflipped = v.bflip(2).eval();  // Flip within blocks of size 2 -> [2, 1, 4, 3]
    EXPECT_EQ(bflipped[0], TypeParam(2));
    EXPECT_EQ(bflipped[1], TypeParam(1));
    EXPECT_EQ(bflipped[2], TypeParam(4));
    EXPECT_EQ(bflipped[3], TypeParam(3));
}

TYPED_TEST(VectorTest, Repeat)
{
    auto v = this->make_sequence(2);  // 1, 2
    auto rep = v.repeat(3).eval();
    EXPECT_EQ(rep.size(), 6);
    EXPECT_EQ(rep[0], TypeParam(1));
    EXPECT_EQ(rep[2], TypeParam(1));
    EXPECT_EQ(rep[5], TypeParam(2));
}

TYPED_TEST(VectorFloatTest, TypeConversion)
{
    Vector<TypeParam> v = {TypeParam(1.5), TypeParam(2.5)};
    auto int_v = v.template type<int32_t>().eval();
    EXPECT_EQ(int_v[0], 1);
    EXPECT_EQ(int_v[1], 2);
}

// ============================================================================
// 10. ERROR HANDLING (ASSERTIONS)
// ============================================================================

TYPED_TEST(VectorTest, ZeroSizeConstructorThrows)
{
    EXPECT_THROW({ Vector<TypeParam> v(0); }, std::runtime_error);
}

TYPED_TEST(VectorTest, OperatorSizeMismatchThrows)
{
    Vector<TypeParam> v1(3);
    Vector<TypeParam> v2(4);
    EXPECT_THROW(
        {
            auto expr = v1 + v2;
            expr.eval();
        },
        std::runtime_error);
}

TYPED_TEST(VectorTest, BlockOutOfBoundsThrows)
{
    Vector<TypeParam> v(5);
    EXPECT_THROW({ auto b = v.block(3, 3); }, std::runtime_error);
    EXPECT_THROW({ auto b = v.block(0, 0); }, std::runtime_error);  // Block size 0
}

TYPED_TEST(VectorTest, ElemOutOfBoundsThrows)
{
    Vector<TypeParam> v(5);
    EXPECT_THROW({ auto e = v.elem(5); }, std::runtime_error);
}

TYPED_TEST(VectorTest, IndexOutOfBoundsThrows)
{
    Vector<TypeParam> v(5);
    EXPECT_THROW({ auto i = v.index(Cut(0, 1, 6)); }, std::runtime_error);
}

TYPED_TEST(VectorTest, DivisibilityThrows)
{
    Vector<TypeParam> v(5);
    EXPECT_THROW({ auto result = v.bany(2); }, std::runtime_error);
    EXPECT_THROW({ auto result = v.ball(3); }, std::runtime_error);
    EXPECT_THROW({ auto result = v.bflip(4); }, std::runtime_error);
}

TYPED_TEST(VectorTest, DotProductSizeMismatchThrows)
{
    Vector<TypeParam> v1(3);
    Vector<TypeParam> v2(4);
    EXPECT_THROW({ v1.dot(v2); }, std::runtime_error);
}

TYPED_TEST(VectorTest, TurnExceedsSizeThrows)
{
    Vector<TypeParam> v(3);
    EXPECT_THROW({ auto t = v.turn(4); }, std::runtime_error);
}

// ============================================================================
// 11. INTEROPERABILITY & RAW VIEWS
// ============================================================================

TYPED_TEST(VectorTest, OstreamFormatting)
{
    Vector<TypeParam> v = {TypeParam(1), TypeParam(2)};
    std::stringstream ss;
    ss << v;
    EXPECT_TRUE(ss.str().find("Vector<") != std::string::npos);
    EXPECT_TRUE(ss.str().find("[") != std::string::npos);
}

TYPED_TEST(VectorTest, OstreamInvalidFormatting)
{
    Vector<TypeParam> v;
    std::stringstream ss;
    ss << v;
    EXPECT_TRUE(ss.str().find("Vector<>[]") != std::string::npos);
}

TYPED_TEST(VectorTest, WrapIntoTensor)
{
    Vector<TypeParam> v = {TypeParam(1), TypeParam(2), TypeParam(3)};
    Tensor<TypeParam> t = v.wrap(Mode::Hold);
    // Since we don't have tensor implementation detail here, we just check shape size
    EXPECT_EQ(t.shape().size(), 1);
}

TYPED_TEST(VectorTest, WrapIntoTensorOwn)
{
    Vector<TypeParam> v = {TypeParam(1), TypeParam(2)};
    Tensor<TypeParam> t = v.wrap(Mode::Own);
    EXPECT_FALSE(v.valid());  // Ownership should be released to tensor
}

TYPED_TEST(VectorTest, StaticAndStridedViews)
{
    auto s1 = Vector<TypeParam>::template stat<3>(TypeParam(7));
    EXPECT_EQ(s1[0], TypeParam(7));

    std::vector<TypeParam> list = {TypeParam(1), TypeParam(2), TypeParam(3)};
    auto s2 = Vector<TypeParam>::template stat<3>(list);
    EXPECT_EQ(s2[0], TypeParam(1));

    // Initializer list static view
    auto s3 = Vector<TypeParam>::template stat<2>({TypeParam(5), TypeParam(6)});
    EXPECT_EQ(s3[0], TypeParam(5));
    EXPECT_EQ(s3[1], TypeParam(6));

    TypeParam raw_data[] = {TypeParam(10), TypeParam(20), TypeParam(30), TypeParam(40)};
    VectorImpl::Strided<TypeParam> strided_view(2, raw_data, 2);  // size 2, stride 2
    EXPECT_EQ(strided_view[0], TypeParam(10));                    // index 0 * 2 = 0
    EXPECT_EQ(strided_view[1], TypeParam(30));                    // index 1 * 2 = 2
}
