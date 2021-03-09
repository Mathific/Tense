# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Matrix](#matrix)
- [Tensor](#tensor)
- [Initialization](#initialization)
- [Unary Operations](#unary-operations)
- [Binary Operations](#binary-operations)
- [Reduction Operations](#reduction-operations)
- [Class Operators](#class-operators)
- [Square Operations](#square-operations)
- [Random Generation](#random-generation)
- [Matrix Operations](#matrix-operations)
- [Others](#others)

# Introduction
``` c++
#include <tense/tense.h>
```

The library's interface exists in ```Tense``` namespace. The library provides these data types:
+ ```Size```: Size type of the library (unsigned integer).
+ ```Matrix<Major, Type>```: Row Major or Column Major Matrix.
+ ```Tensor<Type>```: N Dimensional Tensor.

The API is grouped in categories. Some of the functions are expression templates (lazy evaluation) and the rest are normal (immediate evaluation). Expression template functions can be chained together so there will be no need for temporary storage and execution speed will be improved.

# Matrix
```Matrix<Major, Type>``` is a template class representing 2D row major or col major arrays. 

here are type definitions of the class:
+ ```Matrix<Major, Type>::Type```: data type of matrix
+ ```Matrix<Major, Type>::Major```: majority of matrix. can be ```Tense::Row``` or ```Tense::Col```

Here are the class constructors:
+ ```Matrix<Major, Type>::Matrix(Size rows, Size cols, Type default_value=0)```: create a matrix with all values are zero
+ ```Matrix<Major, Type>::Matrix(Size rows, Size cols, std::initializer_list<Type> list)```: create a column matrix with initializer list
+ ```Matrix<Major, Type>::Matrix(Size rows, Size cols, std::initializer_list<std::initializer_list<Type>> list)```: create a matrix with 2D initializer list
+ ```Matrix<Major, Type>::Matrix(Size rows, Size cols, Type *data, Tense::Mode mode)```: create a matrix with data data pointer. ```Tense::Mode``` can be ```Hold, Copy, Own```
+ ```Matrix<Major, Type>::Matrix(expression)```: create a matrix from a matrix expression

Here are the class functions:
+ ```memory()```: memory occupied by this class
+ ```copy()```: deep copy of the matrix
+ ```resize(Size rows, Size cols)```: resize and cols of the matrix
+ ```release()```: release the memory pointer the matrix is holding
+ ```reset()```: reset (clear) the matrix
+ ```valid()```: is the matrix valid or empty (uninitialized)
+ ```rows()```: number of rows of the matrix
+ ```cols()```: number of cols of the matrix
+ ```size()```: number of elements of the matrix
+ ```data()```: data pointer of the matrix
+ ```begin()```: c++ style begin iterator
+ ```end()```: c++ style end iterator
+ ```operator[](Size index)```: same as ```data()[index]```, useful when the matrix is really one dimensional (row or col vector)
+ ```operator()(Size i, Size j)```: 2D access to matrix elements
+ ```operator=(Type scalar)```: fill the matrix with scalar
+ ```operator=(expression)```: fill the matrix expression values

Matrix Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;

float raw[6] = {1, 2, 3, 4, 5, 6};
Matrix A(/*rows=*/2, /*cols=*/3);
Matrix B(/*rows=*/2, /*cols=*/3, /*value=*/2.4);
Matrix C(/*rows=*/2, /*cols=*/3, /*list=*/{2, 1, 3, 5, 4, 6});
Matrix D(/*rows=*/2, /*cols=*/3, /*data_pointer=*/raw, /*mode=*/Mode::Hold);

print(A.rows(), A.cols(), A.size());
print("A:", A, "B:", B, "C:", C, "D:", D);
for (auto item : D) std::cout << item << ", ";
std::cout << std::endl;

for (Size i = 0; i < D.rows(); ++i)
{
    for (Size j = 0; j < D.cols(); ++j) std::cout << D(i, j) << ", ";
    std::cout << std::endl;
}
```

The Output:
``` c++
2 3 6 
A: Matrix<f32,2,3> [
    0, 0, 0
    0, 0, 0
] B: Matrix<f32,2,3> [
    2.4, 2.4, 2.4
    2.4, 2.4, 2.4
] C: Matrix<f32,2,3> [
    2, 1, 3
    5, 4, 6
] D: Matrix<f32,2,3> [
    1, 2, 3
    4, 5, 6
] 
1, 2, 3, 4, 5, 6, 
1, 2, 3, 
4, 5, 6,
```

# Tensor
```Tensor<Type>``` is a template class representing N dimensional arrays.

here are type definitions of the class:
+ ```Tensor<Type>::Type```: data type of tensor

Here are the class constructors:
+ ```Tensor<Type>::Tensor(Shape shape, Size default_value=0)```
+ ```Tensor<Type>::Tensor(Shape shape, std::vector<Type> list)```
+ ```Tensor<Type>::Tensor(Shape shape, Type *data, Tense::Mode mode)```: ```Tense::Mode``` can be ```Hold, Copy, Own```

Here are the class functions:
+ ```memory()```: memory taken by this class
+ ```copy()```: deep copy of the tensor
+ ```release()```: release the memory pointer the tensor is holding
+ ```reset()```: reset (clear) the tensor
+ ```valid()```: is the tensor valid or empty (uninitialized)
+ ```shape()```: shape of the tensor
+ ```size()```: number of elements of the tensor
+ ```size(Size index)```: shape item at index
+ ```data()```: data pointer of the tensor
+ ```begin()```: c++ style begin iterator
+ ```end()```: c++ style end iterator
+ ```operator[](Size index)```: same as ```data()[index]```
+ ```operator()(Size indexes...)```: ND access to tensor elements
+ ```operator=(Type scalar)```: fill the tensor with scalar
+ ```operator=(expression)```: fill the tensor expression values

Tensor Example:
``` c++
using Tensor = Tense::Tensor<float>;

float raw[6] = {1, 2, 3, 4, 5, 6};
Tensor A(/*shape=*/{1, 2, 3});
Tensor B(/*shape=*/{1, 2, 3}, /*value=*/2.4);
Tensor C(/*shape=*/{1, 2, 3}, /*list=*/{2, 1, 3, 5, 4, 6});
Tensor D(/*shape=*/{1, 2, 3}, /*data_pointer=*/raw, /*mode=*/Mode::Hold);

print(A.shape(), A.size(), A.size(0), A.size(1), A.size(2));
Tense::print("A:", A, "B:", B, "C:", C, "D:", D);
for (auto item : D) std::cout << item << ", ";
std::cout << std::endl;

for (Size i = 0; i < D.size(0); ++i)
{
    for (Size j = 0; j < D.size(1); ++j)
    {
        for (Size k = 0; k < D.size(2); ++k) std::cout << D(i, j, k) << ", ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
```

The Output:
``` c++
Shape<1,2,3> 6 1 2 3 
A: Tensor<f32,1,2,3> [
    0, 0, 0, 0, 0, 0
] B: Tensor<f32,1,2,3> [
    2.4, 2.4, 2.4, 2.4, 2.4, 2.4
] C: Tensor<f32,1,2,3> [
    2, 1, 3, 5, 4, 6
] D: Tensor<f32,1,2,3> [
    1, 2, 3, 4, 5, 6
] 
1, 2, 3, 4, 5, 6, 
1, 2, 3, 
4, 5, 6, 

```

# Initialization
These static functions create an expression that initializes matrices and tensors. Here is a list of them:

+ ```init```: creates an expression with a single value for all elements.
+ ```eye```: creates identity expression matrix with size of rows and cols or single size for both rows and cols
+ ```seq```: creates expression with sequenced numbers
+ ```strided```: creates expression matrix with strided data pointer
+ ```stat```: creates expression matrix with static data allocated on stack

These functions create and expression with a single special value for all elements:
+ ```numin```: minimum value of type
+ ```numax```: minimum value of type
+ ```lowest```: lower value of type
+ ```epsilon```: epsilon value of type
+ ```inf```: infinity value of type
+ ```nan```: nan value of type
+ ```zeros```: zero
+ ```ones```: one
+ ```e```: natural number
+ ```pi```: pi number
+ ```sqrt2```: square root of 2

Here is a matrix example of how to use initialization functions:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;

Matrix A(/*list=*/{{2, 1, 3}, {5, 4, 6}});
Matrix B = Matrix::init(/*rows=*/2, /*cols=*/3, /*value=*/2.4);
Matrix C = Matrix::nan(/*rows=*/2, /*cols=*/3);
Matrix D = Matrix::eye(/*rows=*/2, /*cols=*/3);
Matrix E = Matrix::seq(/*rows=*/2, /*cols=*/3, /*start=*/1, /*end=*/7);

float raw[6] = {1, 2, 3, 4, 5, 6};
auto F = Matrix::strided(/*rows=*/1, /*cols=*/3, raw, /*row-stride=*/1, /*col-stride=*/2);
auto G = Matrix::stat</*rows=*/2, /*cols=*/3>(/*value=*/2.4);
auto H = Matrix::stat</*rows=*/2, /*cols=*/3>(/*list=*/{{2, 1, 3}, {5, 4, 6}});

Tense::print("A:", A, "B:", B, "C:", C, "D:", D, "E:", E, "F:", F, "G:", G, "H:", H);
```

The Output:
``` c++
A: Matrix<f32,2,3> [
    2, 1, 3
    5, 4, 6
] B: Matrix<f32,2,3> [
    2.4, 2.4, 2.4
    2.4, 2.4, 2.4
] C: Matrix<f32,2,3> [
    nan, nan, nan
    nan, nan, nan
] D: Matrix<f32,2,3> [
    1, 0, 0
    0, 1, 0
] E: Matrix<f32,2,3> [
    1, 2, 3
    4, 5, 6
] F: Matrix<f32,1,3> [
    1, 3, 5
] G: Matrix<f32,2,3> [
    2.4, 2.4, 2.4
    2.4, 2.4, 2.4
] H: Matrix<f32,2,3> [
    2, 1, 3
    5, 4, 6
] 
```

Here is a tensor example of how to use initialization functions:
``` c++
using Tensor = Tense::Tensor<float>;
Tensor A = Tensor::init(/*shape=*/{1, 2, 3}, /*value=*/2.4);
Tensor B = Tensor::init(/*shape=*/{1, 2, 3}, /*list=*/{2, 1, 3, 5, 4, 6});
Tensor C = Tensor::inf(/*shape=*/{1, 2, 3});
Tensor D = Tensor::seq(/*shape=*/{1, 2, 3}, /*start=*/1, /*end=*/7);
Tense::print("A:", A, "B:", B, "C:", C, "D:", D);
```

The Output:
``` c++
A: Tensor<f32,1,2,3> [
    2.4, 2.4, 2.4, 2.4, 2.4, 2.4
] B: Tensor<f32,1,2,3> [
    2, 1, 3, 5, 4, 6
] C: Tensor<f32,1,2,3> [
    inf, inf, inf, inf, inf, inf
] D: Tensor<f32,1,2,3> [
    1, 2, 3, 4, 5, 6
] 
```

# Unary Operations
Unary functions operate on a matrix or a tensor elementwise. The general function is named ```unary``` in matrix and tensor class. This function takes in a function that does the elementwise operation.

Matrix Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
auto F = [](auto d) { return d + 1; };
Matrix A({{1, 2}, {3, 4}});
Matrix B = A.unary</*output-expression-data-type=*/float>(/*function=*/F);  // same as A.add(1)
Tense::print("A:", A, "B:", B);
```

The Output:
``` c++
A: Matrix<f32,2,2> [
    1, 2
    3, 4
] B: Matrix<f32,2,2> [
    2, 3
    4, 5
] 
```

Tensor Example:
``` c++
using Tensor = Tense::Tensor<float>;
auto F = [](auto d) { return d + 1; };
Tensor A = Tensor::init({1, 2, 2}, {1, 2, 3, 4});
Tensor B = A.unary</*output-expression-data-type=*/float>(/*function=*/F); // same as A.add(1)
Tense::print("A:", A, "B:", B);
```

The Output:
``` c++
A: Tensor<f32,1,2,2> [
    1, 2, 3, 4
] B: Tensor<f32,1,2,2> [
    2, 3, 4, 5
] 
```

There are functions that make things easy by implementing the lambda themselves so you don't have to. Here op1 (operand 1) are elements of matrix or tensor being operated on and op2 and op3 are input scalar arguments. Here is a list of them: 

C++ math and complex library functions: ```abs, cos, sin, tan, acos, asin, atan, cosh, sinh, tanh, acosh, asinh, exp, log, log2, log10, exp2, expm1, ilogb, log1p, sqrt, cbrt, erf, erfc, tgamma, lgamma, ceil, floor, trunc, round, lround, llround, rint, lrint, llrint, nearbyint, atan2, fdim, ldexp, scalbn, scalbln, pow, hypot, remainder, copysign, nextafter, nexttoward, fmin, fmax, proj, arg, norm, real, imag, isnan, isinf, conj, polar```.

Others:
+ ```type<T>```: convert op1 type to ```T```
+ ```neg```: negate op1
+ ```pos```: positate op1
+ ```clip```: clip the op1 between op2 and op3
+ ```square```: raise op1 to the power of 2
+ ```cube```: raise op1 to the power of 3
+ ```pow<N>```: same as ```pow``` mentioned above with template input (for speed)
+ ```frac```: fractional part of op1
+ ```ln```: natural logarithm of op1
+ ```rev```: 1 / op1
+ ```rsqrt```: inverse of square root of op1
+ ```relu```: ```std::max(0, op1)```
+ ```sigmoid```: sigmoid of op1
+ ```deg2rad```: degree to radian conversion of op1
+ ```rad2deg```: radian to degree conversion of op1
+ ```sign```: ```std::signbit(op1)```
+ ```zero```: is op1 == 0?
+ ```nonzero```: is op1 != 0?
+ ```add```: add op1 to op2
+ ```sub```: subtract op1 from op2
+ ```mul```: multiply op1 with op2
+ ```div```: divide op1 by op2
+ ```mod```: mod operation(%)
+ ```_and```: op1 and op2 'and' operation
+ ```_or```: op1 and op2 'or' operation
+ ```_xor```: op1 and op2 'xor' operation
+ ```lshift```: left shift op1 with op2
+ ```rshift```: right shift op1 with op2
+ ```revsub```: subtract op2 from op1
+ ```revdiv```: divide op2 by op1
+ ```revmod```: mod operation with input and data exchanged
+ ```revlshift```: left shift op2 with op1
+ ```revrshift```: right shift op2 with op1
+ ```heaviside```: heaviside function of op1 and op2
+ ```gt```: is op1 greater than op2?
+ ```ge```: is op1 greater than or equal to op2?
+ ```lt```: is op1 lower than op2?
+ ```le```: is op1 lower than or equal to op2?
+ ```eq```: is op1 equal to op2?
+ ```ne```: is op1 not equal to op2?

# Binary Operations
Binary functions operate on two matrices or tensors elementwise. The general function is named ```binary``` in matrix and tensor class. This function takes in a second operand and a function that does the elementwise operation.

Matrix Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
auto F = [](auto d1, auto d2) { return d1 * d2; };
Matrix A = Matrix::init(2, 2, 2), B = Matrix::init(2, 2, 3);
Matrix C = A.binary</*output-expression-data-type=*/float>(/*second-operand=*/B, /*function=*/F); // same as A.mul(B)
Tense::print("A:", A, "B:", B, "C:", C);
```

The Output:
``` c++
A: Matrix<f32,2,2> [
    2, 2
    2, 2
] B: Matrix<f32,2,2> [
    3, 3
    3, 3
] C: Matrix<f32,2,2> [
    6, 6
    6, 6
] 
```

Tensor Example:
``` c++
using Tensor = Tense::Tensor<float>;
auto F = [](auto d1, auto d2) { return d1 * d2; };
Tensor A = Tensor::init({1, 2, 2}, 2), B = Tensor::init({1, 2, 2}, 3);
Tensor C = A.binary</*output-expression-data-type=*/float>(/*second-operand=*/B, /*function=*/F); // same as A.mul(B)
Tense::print("A:", A, "B:", B, "C:", C);
```

The Output:
``` c++
A: Tensor<f32,1,2,2> [
    2, 2, 2, 2
] B: Tensor<f32,1,2,2> [
    3, 3, 3, 3
] C: Tensor<f32,1,2,2> [
    6, 6, 6, 6
] 
```

There are functions that make things easy by implementing the lambda themselves. Here op1 (operand 1) are elements of matrix or tensor being operated on and op2 is the second matrix or tensor. Here is a list of them: 

+ ```gt```: is op1 greater than op2?
+ ```ge```: is op1 greater than or equal to op2?
+ ```lt```: is op1 lower than op2?
+ ```le```: is op1 lower than or equal to op2?
+ ```eq```: is op1 equal to op2?
+ ```ne```: is op1 not equal to op2?
+ ```complex```: create complex output with op1 as real part and op2 as imaginary part
+ ```lshift```: left shift op1 with op2
+ ```rshift```: right shift op1 with op2
+ ```add```: add op1 to op2
+ ```sub```: subtract op1 from op2
+ ```mul```: multiply op1 with op2
+ ```div```: divide op1 by op2
+ ```mod```: mod operation(%)
+ ```_and```: op1 and op2 'and' operation
+ ```_or```: op1 and op2 'or' operation
+ ```_xor```: op1 and op2 'xor' operation
+ ```atan2```: ```std::atan2(op1, op2)```
+ ```pow```: ```std::pow(op1, op2)```
+ ```remainder```: ```std::remainder(op1, op2)```
+ ```fmin```: ```std::fmin(op1, op2)```
+ ```fmax```: ```std::fmax(op1, op2)```
+ ```mask```: returns ```op2 ? op1 : 0```
+ ```heaviside```: heaviside function of op1 and op2

# Reduction Operations
Reduction operations reduce matrices or tensors to smaller ones (lower dimension). There are 4 types reduction operations on matrices: blockwise, rowwise, columnwise, elementwise. You can pass the dimension to reduction operations on tensors. The functions are called ```breduce, rreduce, creduce, reduce``` on matrices and ```reduce``` on tensors. These functions take in an initial value and a function that does the operation.

Matrix Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
auto F = [](auto d1, auto d2) { return d1 + d2; };
Matrix A = Matrix::seq(4, 4, 1, 17);
Matrix B = A.breduce</*output-expression-data-type=*/float>(/*block-rows=*/2, /*block-cols=*/2, /*initial-value=*/0, /*function=*/F); // same as A.bsum(2, 2)
Matrix C = A.rreduce</*output-expression-data-type=*/float>(/*initial-value=*/0, /*function=*/F); // same as A.rsum()
Matrix D = A.creduce</*output-expression-data-type=*/float>(/*initial-value=*/0, /*function=*/F); // same as A.csum()
Matrix E = A.reduce</*output-expression-data-type=*/float>(/*initial-value=*/0, /*function=*/F); // same as A.sum()
Tense::print("A:", A, "B:", B, "C:", C, "D:", D, "E:", E);
```

The Output:
``` c++
A: Matrix<f32,4,4> [
     1,  2,  3,  4
     5,  6,  7,  8
     9, 10, 11, 12
    13, 14, 15, 16
] B: Matrix<f32,2,2> [
    14, 22
    46, 54
] C: Matrix<f32,4,1> [
    10
    26
    42
    58
] D: Matrix<f32,1,4> [
    28, 32, 36, 40
] E: Matrix<f32,1,1> [
    136
] 
```

Tensor Example:
``` c++
using Tensor = Tense::Tensor<float>;
auto F = [](auto d1, auto d2) { return d1 + d2; };
Tensor A = Tensor::init({2, 2, 2}, 2);
Tensor B = A.reduce</*output-expression-data-type=*/float>(/*initial-value=*/0, /*function=*/F, /*dimension=*/2); // same as A.sum(2)
Tensor C = A.reduce</*output-expression-data-type=*/float>(/*initial-value=*/0, /*function=*/F, /*dimension=*/1); // same as A.sum(1)
Tensor D = A.reduce</*output-expression-data-type=*/float>(/*initial-value=*/0, /*function=*/F, /*dimension=*/0); // same as A.sum()
Tense::print("A:", A, "B:", B, "C:", C, "D:", D);
```

The Output:
``` c++
A: Tensor<f32,2,2,2> [
    2, 2, 2, 2, 2, 2, 2, 2
] B: Tensor<f32,2,2> [
    4, 4, 4, 4
] C: Tensor<f32,2> [
    8, 8
] D: Tensor<f32,1> [
    16
] 
```

There are functions that make things easy by implementing the lambda themselves. Here is a list of them (blockwise have ```b``` prefix, rowwise have ```r``` prefix and colwise have ```c``` prefix where it makes sense): 

+ ```sum```: sum of elements
+ ```prod```: product of elements
+ ```max```: maximum of elements
+ ```min```: minimum of elements
+ ```count```: counts the number of elements equal to op2 
+ ```maxidx```: index of maximum element
+ ```minidx```: index of minimum element
+ ```all```: do all the elements evaluate to true?
+ ```any```: do any of the elements evaluate to true?
+ ```mean```: mean of elements
+ ```var```: variance of elements. takes in degrees of freedom optionally
+ ```cov```: covariance of elements and op2. takes in degrees of freedom optionally
+ ```std```: standard deviation of elements. takes in degrees of freedom optionally
+ ```norm<N>```: N'th norm of elements
+ ```contains```: do any of the elements equal op2?
+ ```equal```: do all of the elements equal op2?
+ ```close```: are all the elements close to op2?
+ ```distance<N>```: N'th distance of elements from op2
+ ```normalize```: min-max normalization of elements
+ ```standize```: standardize distribution of elements. takes in degrees of freedom optionally
+ ```trace```: trace of a matrix

# Class Operators
Mathematical operators are listed below for matrix and tensor types. They mean the same (elementwise) as applying them on C++ integers and floating-points except ```operator*``` on matrices which means matrix multiplication.

+ ```operator-```: uses function ```neg```
+ ```operator+```: uses function ```pos```
+ ```operator!```: uses function ```_not```
+ ```operator~```: uses function ```_not```
+ ```operator<```: uses function ```lt```
+ ```operator<=```: uses function ```le```
+ ```operator>```: uses function ```gt```
+ ```operator>=```: uses function ```ge```
+ ```operator==```: uses function ```eq```
+ ```operator!=```: uses function ```ne```
+ ```operator+```: uses function ```add```
+ ```operator-```: uses function ```sub```
+ Tensor ```operator*```: uses function ```mul```
+ Matrix ```operator*```: uses function ```mm```
+ ```operator/```: uses function ```div```
+ ```operator%```: uses function ```mod```
+ ```operator&```: uses function ```_and```
+ ```operator|```: uses function ```_or```
+ ```operator^```: uses function ```_xor```
+ ```operator<<```: uses function ```lshift```
+ ```operator>>```: uses function ```rshift```

Matrix Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::init(2, 2, 1);
Matrix B = Matrix::init(2, 2, 2);
Matrix C = (A + A) * B;
Tense::print("A:", A, "B:", B, "C:", C);
```

The Output:
``` c++
A: Matrix<f32,2,2> [
    1, 1
    1, 1
] B: Matrix<f32,2,2> [
    2, 2
    2, 2
] C: Matrix<f32,2,2> [
    8, 8
    8, 8
] 
```

Tensor Example:
``` c++
using Tensor = Tense::Tensor<float>;
Tensor A = Tensor::init({1, 2, 2}, 1);
Tensor B = Tensor::init({1, 2, 2}, 2);
Tensor C = (A + A) * B;
Tense::print("A:", A, "B:", B, "C:", C);
```

The Output:
``` c++
A: Tensor<f32,1,2,2> [
    1, 1, 1, 1
] B: Tensor<f32,1,2,2> [
    2, 2, 2, 2
] C: Tensor<f32,1,2,2> [
    4, 4, 4, 4
] 
```

# Square Operations
Square functions mask a part of a square matrix (for example selecting upper part of a matrix). Here is a list of them: 

+ ```upper```: upper and diagonal part of matrix with zeros elsewhere
+ ```lower```: lower and diagonal part of matrix with zeros elsewhere
+ ```oupper```: upper part of matrix with ones in diagonal and zeros elsewhere
+ ```olower```: lower part of matrix with ones in diagonal and zeros elsewhere
+ ```zupper```: upper part of matrix with zeros in diagonal and elsewhere
+ ```zlower```: lower part of matrix with zeros in diagonal and elsewhere
+ ```usymm```: lower part replaced with upper part
+ ```lsymm```: upper part replaced with lower part
+ ```uherm```: upper hermitian matrix
+ ```lherm```: lower hermitian matrix
+ ```diagonal```: diagonal part of matrix with zeros elsewhere
+ ```zdiagonal```: zeros in diagonal and original data elsewhere
+ ```odiagonal```: ones in diagonal and original data elsewhere

Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::seq(3, 3, 1, 10);
Matrix B = A.oupper();
Matrix C = A.diagonal();
Tense::print("A:", A, "B:", B, "C:", C);
```

The Output:
``` c++
A: Matrix<f32,3,3> [
    1, 2, 3
    4, 5, 6
    7, 8, 9
] B: Matrix<f32,3,3> [
    1, 2, 3
    0, 1, 6
    0, 0, 1
] C: Matrix<f32,3,3> [
    1, 0, 0
    0, 5, 0
    0, 0, 9
] 
```

# Random Generation
These static functions create an expression that draws its elements from random distributions. The general function is named ```dist``` in matrix and tensor class and takes in size of rows and cols and a ```std``` style random distribution. Here is a list of them:

+ ```uniform```: random numbers drawn from ```std::uniform_int_distribution``` or ```std::uniform_real_distribution```
+ ```bernoulli```: random numbers drawn from ```std::bernoulli_distribution```
+ ```binomial```: random numbers drawn from ```std::binomial_distribution```
+ ```geometric```: random numbers drawn from ```std::geometric_distribution```
+ ```pascal```: random numbers drawn from ```std::negative_binomial_distribution```
+ ```poisson```: random numbers drawn from ```std::poisson_distribution```
+ ```exponential```:random numbers drawn from ```std::exponential_distribution```
+ ```gamma```: random numbers drawn from ```std::gamma_distribution```
+ ```weibull```: random numbers drawn from ```std::weibull_distribution```
+ ```extremevalue```: random numbers drawn from ```std::extreme_value_distribution```
+ ```normal```: random numbers drawn from ```std::normal_distribution```
+ ```lognormal```: random numbers drawn from ```std::lognormal_distribution```
+ ```chisquared```: random numbers drawn from ```std::chi_squared_distribution```
+ ```cauchy```: random numbers drawn from ```std::cauchy_distribution```
+ ```fisherf```: random numbers drawn from ```std::fisher_f_distribution```
+ ```studentt```: random numbers drawn from ```std::student_t_distribution```

Matrix Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(/*rows=*/2, /*cols=*/2, /*a=*/0, /*b=*/1);
Matrix B = Matrix::normal(/*rows=*/2, /*cols=*/2, /*mean=*/0, /*std=*/1);
Tense::print("A:", A, "B:", B);
```

The Output:
``` c++
A: Matrix<f32,2,2> [
    0.224802, 0.431263
    0.603951, 0.298265
] B: Matrix<f32,2,2> [
    -0.364752, -0.326479
      1.35156,  -3.25021
] 
```

Tensor Example:
``` c++
using Tensor = Tense::Tensor<float>;
Tensor A = Tensor::uniform(/*shape=*/{1, 2, 2}, /*a=*/0, /*b=*/1);
Tensor B = Tensor::normal(/*shape=*/{1, 2, 2}, /*mean=*/0, /*std=*/1);
Tense::print("A:", A, "B:", B);
```

The Output:
``` c++
A: Tensor<f32,1,2,2> [
    0.500647, 0.184513, 0.593728, 0.47901
] B: Tensor<f32,1,2,2> [
    -0.687819, -1.88107, 0.0232958, 0.970518
] 
```

# Matrix Operations
Here are some matrix specific functions like decompositions and solutions:

+ ```mm```: matrix multiplication
+ ```inverse```: inverse of matrix
+ ```det```: determinant of matrix
+ ```plu```: lower-upper decomposition of matrix
+ ```cholesky```: Cholesky decomposition of matrix
+ ```qr```: QR decomposition of matrix
+ ```schur```: Schur decomposition of matrix
+ ```solve```: solve linear equations
+ ```ls```: least squares solution
+ ```eigen```: Eigen decomposition of matrix
+ ```svd```: singular value decomposition of matrix
+ ```rank```: rank of matrix

```mm``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(2, 5);
Matrix B = Matrix::uniform(5, 3);
Tense::print("A:", A);
Tense::print("B:", B);
Tense::print("A * B:", A.mm(B));
Tense::print("B' * A':", B.trans().mm(A.trans()));
```

The Output:
``` c++
A: Matrix<f32,2,5> [
    0.706187, 0.196863, 0.530463, 0.162713, 0.489317
    0.304355, 0.864272,  0.81781,  0.79162, 0.960777
] 
B: Matrix<f32,5,3> [
     0.544098,  0.264566, 0.0210872
     0.800658,  0.810884,  0.667704
     0.573298,  0.196324,  0.147106
    0.0846627, 0.0698271,  0.824859
     0.686063,  0.113635,  0.378424
] 
A * B: Matrix<f32,2,3> [
    1.19545, 0.517574, 0.543757
    2.05261,  1.10636,  1.72036
] 
B' * A': Matrix<f32,3,2> [
     1.19545, 2.05261
    0.517574, 1.10636
    0.543757, 1.72036
]
```

```inverse``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(3, 3);
Tense::print("A:", A);
Tense::print("A^-1:", A.inverse());
Tense::print("A * A^-1", A * A.inverse());
```

The Output:
``` c++
A: Matrix<f32,3,3> [
    0.706187, 0.196863, 0.530463
    0.162713, 0.489317, 0.304355
    0.864272,  0.81781,  0.79162
] 
A^-1: Matrix<f32,3,3> [
    -4.68537, -9.40728, 6.75648
    -4.54287, -3.40339, 4.35267
     9.80855,  13.7866,  -10.61
] 
A * A^-1 Matrix<f32,3,3> [
              1,  2.12745e-07, -6.56317e-07
    3.11432e-07,            1, -4.05251e-07
    3.13998e-07, -9.83302e-08,            1
]
```

```det``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(3, 3);
Tense::print("A:", A);
Tense::print("Determinant(A):", A.det());
```

The Output:
``` c++
A: Matrix<f32,3,3> [
    0.706187, 0.196863, 0.530463
    0.162713, 0.489317, 0.304355
    0.864272,  0.81781,  0.79162
] 
Determinant(A): 0.0295492
```

```plu``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(3, 3);
auto [lu, pivot] = A.plu();
Tense::print("A:", A);
Tense::print("Upper Part:", lu.upper());
Tense::print("Lower Part:", lu.olower());
Tense::print("Pivot:", pivot);
```

The Output:
``` c++
A: Matrix<f32,3,3> [
    0.706187, 0.196863, 0.530463
    0.162713, 0.489317, 0.304355
    0.864272,  0.81781,  0.79162
] 
Upper Part: Matrix<f32,3,3> [
    0.864272,  0.81781,   0.79162
           0, -0.47136, -0.116361
           0,        0,  0.072534
] 
Lower Part: Matrix<f32,3,3> [
           1,         0, 0
    0.817088,         1, 0
    0.188266, -0.711453, 1
] 
Pivot: Matrix<i32,3,1> [
    3
    3
    3
] 
```

```cholesky``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(3, 3).usymm();
auto chol = A.upper().cholesky();
Tense::print("A:", A);
Tense::print("Upper(A)` * Upper(A):", chol);
```

The Output:
```
A: Matrix<f32,3,3> [
    0.706187, 0.196863, 0.530463
    0.162713, 0.489317, 0.304355
    0.864272,  0.81781,  0.79162
] 
Upper(A)' * Upper(A): Matrix<f32,3,3> [
    0.840349, 0.234263, 0.631241
    0.162713, 0.659119, 0.237405
    0.864272,  0.81781, 0.580339
] 
```

```schur``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(3, 3);
auto [schur, vectors] = A.schur(/*compute-vectors=*/true);
Tense::print("A:", A);
Tense::print("Schur:", schur);
Tense::print("Schur Vectors:", vectors);
```

The Output:
``` c++
A: Matrix<f32,3,3> [
    0.706187, 0.196863, 0.530463
    0.162713, 0.489317, 0.304355
    0.864272,  0.81781,  0.79162
] 
Schur: Matrix<cf32,3,1> [
       (1.63901,0)
      (0.393885,0)
    (-0.0457714,0)
] 
Schur Vectors: Matrix<f32,3,3> [
     -0.51858, -0.697448, -0.494612
    -0.286644,  0.686809, -0.667929
     -0.80555,  0.204597,  0.556085
] 
```

```solve``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(4, 4);
Matrix B = Matrix::uniform(4, 2);
auto X = A.solve(B);
Tense::print("A:", A);
Tense::print("B:", B);
Tense::print("X:", X);
Tense::print("A * X:", A * X);
```

The Output:
``` c++
A: Matrix<f32,4,4> [
    0.706187, 0.196863, 0.530463,  0.162713
    0.489317, 0.304355, 0.864272,   0.81781
     0.79162, 0.960777, 0.459046, 0.0168298
    0.533293, 0.093081, 0.891685,  0.133331
] 
B: Matrix<f32,4,2> [
     0.544098, 0.264566
    0.0210872, 0.800658
     0.810884, 0.667704
     0.573298, 0.196324
] 
X: Matrix<f32,4,2> [
     0.635684, -0.00714725
     0.159471,     0.66336
     0.365801,   0.0535447
    -0.800493,    0.679842
] 
A * X: Matrix<f32,4,2> [
     0.544098, 0.264566
    0.0210871, 0.800658
     0.810884, 0.667704
     0.573298, 0.196324
]
```

```ls``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(4, 3);
Matrix B = Matrix::uniform(4, 2);
auto X = A.ls(B);
Tense::print("A:", A);
Tense::print("B:", B);
Tense::print("X:", X);
Tense::print("A * X:", A * X);
```

The Output:
``` c++
A: Matrix<f32,4,3> [
    0.706187, 0.196863,  0.530463
    0.162713, 0.489317,  0.304355
    0.864272,  0.81781,   0.79162
    0.960777, 0.459046, 0.0168298
] 
B: Matrix<f32,4,2> [
     0.544098, 0.264566
    0.0210872, 0.800658
     0.810884, 0.667704
     0.573298, 0.196324
] 
X: Matrix<f32,3,2> [
     0.670706, -0.356741
    -0.142246,   1.11699
     0.321328,  0.301995
] 
A * X: Matrix<f32,4,2> [
    0.616093, 0.128165
    0.137327, 0.580429
    0.717711, 0.844229
    0.584509, 0.175084
] 
```

```eigen``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(3, 3);
auto [eigen, left, right] = A.eigen(/*compute-left-eigenvectors=*/true, /*compute-right-eigenvectors=*/true);
Tense::print("Eigen:", eigen);
Tense::print("Left Eigenvectors:", left);
Tense::print("Right Eigenvectors:", right);
```

The Output:
``` c++
Eigen: Matrix<cf32,3,1> [
       (1.63901,0)
      (0.393885,0)
    (-0.0457714,0)
] 
Left Eigenvectors: Matrix<f32,3,3> [
    -0.627651, -0.510157, -0.494612
    -0.519748,  0.859788, -0.667929
    -0.579583, 0.0224743,  0.556085
] 
Right Eigenvectors: Matrix<f32,3,3> [
     -0.51858, -0.71769, -0.491457
    -0.286644, 0.674757, -0.312852
     -0.80555, 0.172116,  0.812769
] 
```

```svd``` Example:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(3, 3);
auto [singular, left, right] = A.svd(/*compute-U=*/true, /*compute-V'=*/true);
Tense::print("S:", singular);
Tense::print("U:", left);
Tense::print("V:", right.trans());
```

The Output:
``` c++
S: Matrix<f32,3,1> [
      1.74533
       0.4147
    0.0408258
] 
U: Matrix<f32,3,3> [
    -0.488316,  0.731798, -0.475415
    -0.305478, -0.653658, -0.692397
    -0.817454,  -0.19288,   0.54274
] 
V: Matrix<f32,3,3> [
    -0.630855,  0.587717,  0.506569
    -0.523756, -0.804249,  0.280824
    -0.572453, 0.0881593, -0.815185
]
```

```rank``` Example:
``` c++
A: Matrix<f32,5,3> [
    0.706187, 0.196863,  0.530463
    0.162713, 0.489317,  0.304355
    0.864272,  0.81781,   0.79162
    0.960777, 0.459046, 0.0168298
    0.533293, 0.093081,  0.891685
] 
Rank(A): 3 
```

The Output:
``` c++
using Matrix = Tense::Matrix<Tense::Row, float>;
Matrix A = Matrix::uniform(5, 3);
Tense::print("A:", A);
Tense::print("Rank(A):", A.rank(/*singular-value-theshold=*/1e-5));
```

# Others
+ ```brepeat(Size block-rows, Size block-cols)```: Repeat a matrix blockwise
+ ```rrepeat(Size rows)```: Repeat a col matrix rowwise
+ ```crepear(Size cols)```: Repeat a row matrix columnwise
+ ```repeat(Size rows, Size cols)```: Repeat a single element matrix (could be done with brepeat but this is way faster) 
+ ```repeat(Shape shape)```: 
+ ```block(i-start, j-start, i-size, j-size)```: select a block of matrix
+ ```row(Size row-index)```: select a row of matrix
+ ```col(Size col-index)```: select a col of matrix
+ ```elem(Size row-index, Size col-index)```: select an element of matrix
+ ```diag```: select main diagonal of matrix
+ ```index```: 
+ ```cat```: 
+ ```turn```: shift rows or columns or both of matrix
+ ```flip```: reverse rows or cols or both of matrix
+ ```asdiag```: use a column vector as diagnoal for matrix
+ ```where```: 
+ ```iwhere```: 
+ ```reshape(Size new-rows, Size new-cols)```: reshape matrix
+ ```reshape(Shape new-shape)```: reshape tensor
+ ```expr```: identity expression, useful for turning matrix or tensor to expression
+ ```eval```: evaluate expression to a matrix or a tensor
+ ```item```: first item of matrix or tensor, useful when matrix of tensor has one element
+ ```tensor```: 
+ ```matrix```: 
+ ```trans```: transpose of matrix
+ ```adjoint```:adjoint of matrix
+ ```dot(Expression other)```: matrix dot operation, matrices must have the same shape
+ ```indirect```: 
+ ```sort```: 
+ ```sortidx```: 
+ ```shuffle```: 
