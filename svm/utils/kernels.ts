import Matrix from "ml-matrix";

export function linearKernel(x1: Matrix, x2: Matrix): number {
    return x1.dot(x2);
}