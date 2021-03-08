#include <tense/tense.h>

int main()
{
    using Matrix = Tense::Matrix<Tense::Row, double>;
    // N ~ Normal(0, 1e-8)
    // Y = 0.1 + 0.5*X - 1.2*X^2 + 2*X^3 - .8*X^4 + N

    Matrix F = Matrix::uniform(100, 1);
    Matrix N = Matrix::normal(100, 1, 0, 1e-4);
    Matrix X = Matrix::ones(100, 5);
    Matrix T = {.1, .5, -1.2, 2, -.8};

    X.col(1) = F;
    X.col(2) = F.pow<2>();
    X.col(3) = F.pow<3>();
    X.col(4) = F.pow<4>();

    Matrix Y = X * T + N;
    Matrix TH = X.ls(Y);  // or TH = (X.trans() * X).inverse() * (X.trans() * Y);
    double E = Y.distance<2>(X * TH);

    Tense::print("Weights:", T.trans());
    Tense::print("Estimated Weights:", TH.trans());
    Tense::print("Residual Error:", E);
    return 0;
}

void functional()
{
    using Matrix = Tense::Matrix<Tense::Row, double>;
    using namespace Tense::Functional;
    // N ~ Normal(0, 1e-8)
    // Y = 0.1 + 0.5*X - 1.2*X^2 + 2*X^3 - .8*X^4 + N

    Matrix F = Matrix::uniform(100, 1);
    Matrix N = Matrix::normal(100, 1, 0, 1e-4);
    Matrix X = Matrix::ones(100, 5);
    Matrix T = {.1, .5, -1.2, 2, -.8};

    col(X, 1) = F;
    col(X, 2) = pow<2>(F);
    col(X, 3) = pow<3>(F);
    col(X, 4) = pow<4>(F);

    Matrix Y = X * T + N;
    Matrix TH = ls(X, Y);
    double E = distance<2>(Y, X * TH);

    Tense::print("Weights:", trans(T));
    Tense::print("Estimated Weights:", trans(TH));
    Tense::print("Residual Error:", E);
}
