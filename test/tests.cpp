#include <gtest/gtest.h>
#include <tense/tense.h>

using namespace Tense;

int main(int argc, char *argv[])
{
    using Matrix = Tense::Matrix<Tense::Row, float>;
    auto F = [](auto d) { return d + 1; };
    Matrix A({{1, 2}, {3, 4}});
    Matrix B = A.unary</*output-expression-data-type=*/float>(/*function=*/F);  // same as A.add(1)
    Tense::print("A:", A, "B:", B);
}
