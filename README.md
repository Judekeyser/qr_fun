# Forewords

MIT License

Copyright (c) 2022 Justin Dekeyser

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# QR Decomposition and FUN

The idea is to explore QR decomposition and have a little
**fun**
around the idea. We also would like to use it to compute
eigenvalues and eigenvectors of square matrices.

## Setting up the project

Importing the project in IntelliJ should occur quite smoothly.
Java 17 (OpenJDK I'd say) is mandatory (maybe not *mandatory*, but
recommended anyway).

Make sure you have downloaded the following third-party dependencies:
```
hamcrest-all-1.3
junit-4.13.2
```
and link them in IntelliJ
(File > Project Structure > (left pane) Modules > (tab) Dependencies).

## Tests

After linking the test dependencies, you should be able to run
the different unit tests.

# Evolution

We would like to improve efficiency of the algorithms using two different strategies:
1. Start by computing an Hessenberg form for the input matrix, and take benefit of the shape to lower the iterative products to `O(n^2)` complexity
2. Introduce a simply shift strategy that should improve convergence rate.

For reference, see for example https://www.cs.cornell.edu/~bindel/class/cs6210-f16/lec/2016-10-21.pdf
