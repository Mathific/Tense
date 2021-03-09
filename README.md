# Tense
```Tense``` is a C++17 matrix and tensor library. It is a fast expression template library that uses auto-vectorization. It also uses ```CBLAS``` and ```LAPACKE``` for linear algebra operations. 

# Installation
You can install ```Tense``` system-wide by downloading the [latest release](https://github.com/Mathific/Tense/releases/latest). Make sure you have ```CMake```, a ```C++``` compiler (the newer version, the faster), ```CBLAS``` and ```LAPACKE``` available on your system. Then run these in repository directory:

``` shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DTENSE_NATIVE=ON -DTENSE_INSTALL=ON -DTENSE_TESTS=OFF -DTENSE_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/usr/local/ ..
sudo cmake --install .
```

The ```CMake``` options are:
+ ```TENSE_INSTALL```: Create install for ```Tense```.
+ ```TENSE_TESTS```: Build the tests for ```Tense```.
+ ```TENSE_EXAMPLES```: Build the examples for ```Tense```.
+ ```TENSE_NATIVE```: Targets that link against ```Tense``` are specifically tuned for host CPU. This makes auto-vectorization possible.

Note that this installs [```BLASW``` library](https://github.com/Mathific/Blasw) too. Then you can use it in ```CMake```:

``` shell
find_package(Tense REQUIRED)
add_executable(myexec main.cpp)
target_link_libraries(myexec Tense::Tense)
```

Or you can use ```Tense``` as a ```CMake``` subdirectory by cloning the repository and putting it in your source directory and use it in ```CMake```:

```
add_subdirectory(Tense/)
add_executable(myexec main.cpp)
target_link_libraries(myexec Tense::Tense)
```

# Usage
Documentation of ```Tense``` API is [here](USAGE.md).

# Contributing
You can report bugs, ask questions and request features on [issues page](../../issues).

# License
This library is licensed under BSD 3-Clause permissive license. You can read it [here](LICENSE).
