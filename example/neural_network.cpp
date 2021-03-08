#include <tense/tense.h>

#include <bitset>

int main()
{
    using Tense::print, Tense::Size;
    using namespace Tense::Functional;
    using Matrix = Tense::Matrix<Tense::Row, float>;
    using BMatrix = Tense::Matrix<Tense::Row, bool>;

    BMatrix _X(32, 5);
    for (Size i = 0; i < 32; ++i)
    {
        std::bitset<5> bits(i);
        _X(i, 0) = bits[0];
        _X(i, 1) = bits[1];
        _X(i, 2) = bits[2];
        _X(i, 3) = bits[3];
        _X(i, 4) = bits[4];
    }
    BMatrix _T = col(_X, 0) & col(_X, 1) & col(_X, 2) | col(_X, 3) & col(_X, 4);

    float lrate = 1e-3;
    Matrix W1 = Matrix::normal(5, 2, 0, .1);
    Matrix W2 = Matrix::normal(2, 2, 0, .1);
    Matrix X = _X, T = ccat0(_T == 0, _T == 1);

    auto predict = [&](auto X) {
        auto S = X * W1;
        auto Z = tanh(S);
        auto P = Z * W2;
        auto Y = tanh(P);
        return eval(Y);
    };

    auto train = [&](auto X, auto T) {
        Matrix S = X * W1;
        Matrix Z = tanh(S);
        Matrix P = Z * W2;
        Matrix Y = tanh(P);

        Matrix D2 = mul(Y - T, 1 - square(tanh(P)));
        W2 -= trans(Z) * D2 * lrate;
        Matrix D1 = mul(D2 * W2, 1 - square(tanh(S)));
        W1 -= trans(X) * D1 * lrate;

        return Y;
    };

    for (Size i = 0; i < 10000; ++i)
    {
        Matrix Y = train(X, T);
        print("Iter:", i, ", Error:", distance<2>(Y, T) / 2);
    }

    Matrix Y = predict(X);
    auto E = rmaxidx(Y) == rmaxidx(T);
    print("Accuracy:", mean(type<float>(E)));
}
