import { Matrix } from "ml-matrix";
import { linearKernel } from "./utils/kernels";

export class SVM {
    private C: number = 1;
    private threshold: number = 0.00001;
    private alphas: number[] = [];
    private Eis: number[] = [];
    private bias: number = 0;
    private kernelFuncs = [linearKernel];
    private supportVecTags: Matrix = Matrix.zeros(1, 1);
    private supportVecMat: Matrix = Matrix.zeros(1, 1);
    constructor(penaltyCoefficient?: number, tolerance?: number) {
        if(penaltyCoefficient) {
            this.C = penaltyCoefficient;
        } 
        if(tolerance) {
            this.threshold = tolerance;
        }
    }

    /**
     * 剪辑函数
     * @param L 下界
     * @param H 上界
     * @param alpha 未剪辑的alpha
     */
    private _clip(L: number, H: number, alpha: number): number {
        if(L > H)
            throw new Error('Clip error, L cannot be larger than H!');
        if(H < alpha)
            return H;
        else if(L <= alpha && alpha <= H)
            return alpha;
        else
            return L;
    }

    private _gXi(data: Matrix, kernelFunc: number, xs: Matrix, ys: Matrix): number {
        let sum = 0;
        for(let i = 0; i < this.alphas.length; i++) {
            if(this.alphas[i] > 0)
                sum += this.alphas[i] * ys.get(i, 0) * this.kernelFuncs[kernelFunc](xs.getColumnVector(i), data);
        }
        return sum + this.bias;
    }

    private _calEi(data: Matrix, tag: number, kernelFunc: number, xs: Matrix, ys: Matrix): number {
        return this._gXi(data, kernelFunc, xs, ys) - tag;
    }

    /**
     * 启发式原则在内循环选择alpha2
     * @param index alpha1的索引
     * @param E1 E1
     */
    private _chooseAlpha2(i: number, E1: number): [number, number] {
        // 选择一个使得｜E1 - E2｜最大的alpha2
        let maxIndex = -1;
        let maxGap = 0;
        this.Eis.forEach((val, index) => {
            const gap = Math.abs(E1 - val);
            if(gap > maxGap && index != i) {
                maxIndex = index;
                maxGap = gap;
            }
        });
        return [maxIndex, maxGap];
    }

    private _calEta(kernelFunc: number, x1: Matrix, x2: Matrix) {
        const k11 = this.kernelFuncs[kernelFunc](x1, x1);
        const k22 = this.kernelFuncs[kernelFunc](x2, x2);
        const k12 = this.kernelFuncs[kernelFunc](x1, x2);
        return k11 + k22 - (2 * k12);
    }

    private _calBoundaries(alpha1: number, alpha2: number, labelEqual: boolean): [number, number] {
        let L = 0;
        let H = 0;
        if(labelEqual) {
            L = Math.max(0, this.alphas[alpha2] + this.alphas[alpha1] - this.C);
            H = Math.min(this.C, this.alphas[alpha2] + this.alphas[alpha1]);
        }
        else {
            L = Math.max(0, this.alphas[alpha2] - this.alphas[alpha1]);
            H = Math.min(this.C, this.C + this.alphas[alpha2] - this.alphas[alpha1]);
        }
        
        return [L, H];
    }

    private innerLoop(alpha1: number, trainXs: Matrix, trainYs: Matrix, kernelFunc: number): number {
        const alpha1Old = this.alphas[alpha1];
        const E1 = this.Eis[alpha1];
        const tag1 = trainYs.get(alpha1, 0);
        // choose alpha2
        let [alpha2, gap] = this._chooseAlpha2(alpha1, E1);
        const E2 = this.Eis[alpha2];
        const alpha2Old = this.alphas[alpha2];
        const tag2 = trainYs.get(alpha2, 0);
        // calculate eta
        const eta = this._calEta(kernelFunc, trainXs.getColumnVector(alpha1), trainXs.getColumnVector(alpha2));
        // calculate alpha2NewUnc
        const alpha2NewUnc = alpha2Old + tag2 * gap / eta;
        // calculate L and H
        const [L, H] = this._calBoundaries(alpha1, alpha2, tag1 == tag2);
        // clip alpha2NewUnc
        const alpha2New = this._clip(L, H, alpha2NewUnc);
        // decide update or not
        const improvement = Math.abs(this.alphas[alpha2] - alpha2New);
        if(improvement < 0.00001) {
            return 0;
        }
        else {
            // update alpha pair
            const alpha1New = alpha1Old + tag1 * tag2 * (alpha2Old - alpha2New);
            this.alphas[alpha1] = alpha1New;
            this.alphas[alpha2] = alpha2New;
            const x1 = trainXs.getColumnVector(alpha1);
            const x2 = trainXs.getColumnVector(alpha2);
            const k11 = this.kernelFuncs[kernelFunc](x1, x1);
            const k22 = this.kernelFuncs[kernelFunc](x2, x2);
            const k12 = this.kernelFuncs[kernelFunc](x1, x2);
            const alpha1Grad = alpha1New - alpha1Old;
            const alpha2Grad = alpha2New - alpha2Old;
            const b1New = -E1 - tag1 * k11 * alpha1Grad - tag2 * k12 * alpha2Grad + this.bias;
            const b2New = -E2 - tag1 * k12 * alpha1Grad - tag2 * k22 * alpha2Grad + this.bias;
            // update bias
            if(0 < alpha1New && alpha1New < this.C)
                this.bias = b1New;
            else if(0 < alpha2New && alpha2New < this.C)
                this.bias = b2New;
            else
                this.bias = (b1New + b2New) / 2;
            
            // update E1 and E2
            const E1New = this._calEi(x1, tag1, kernelFunc, trainXs, trainYs);
            const E2New = this._calEi(x2, tag2, kernelFunc, trainXs, trainYs);
            this.Eis[alpha1] = E1New;
            this.Eis[alpha2] = E2New;
            return 1;
        }
        
    }

    private _outerLoop(loopEntireSet: boolean, trainXs: Matrix, trainYs: Matrix, kernelFunc: number): [number, boolean] {
        let pairsUpdated = 0;
        // for every boundary points
        if(!loopEntireSet) {
            for(let i = 0; i < this.alphas.length; i++) {
                let E1 = this.Eis[i];
                const violation = Math.abs(E1 * trainYs.get(i, 0));
                if(0 < this.alphas[i] && this.alphas[i] < this.C && violation > this.threshold) {
                    // if it's a boundary point and it violated KKT conditions.
                    pairsUpdated += this.innerLoop(i,trainXs, trainYs, kernelFunc);
                }
            }
            return [pairsUpdated, pairsUpdated > 0 ? false : true];
        }
        // for every points
        else {
            for(let i = 0; i < this.alphas.length; i++) {
                let E1 = this.Eis[i];
                const violation = E1 * trainYs.get(i, 0);
                if((violation < -this.threshold && this.alphas[i] < this.C) || (violation > this.threshold && this.alphas[i] > 0)) {
                    // if this point violates KKT conditions
                    pairsUpdated += this.innerLoop(i, trainXs, trainYs, kernelFunc);
                }
            }
            return [pairsUpdated, false];
        }

        
    }

    public fit(trainXs: Matrix, trainYs: Matrix, kernelFunc: number, iterations: number) {
        let xs = trainXs.transpose();
        let ys = trainYs.transpose();
        let iters = iterations;
        let prevStatus = false;
        this.alphas = new Array(xs.columns).fill(0);
        for(let i = 0; i < ys.rows; i++) {
            this.Eis.push(-ys.get(i, 0));
        }
        while(iters--) {
            let [count, loopEntireSet] = this._outerLoop(prevStatus, xs, ys, kernelFunc);
            console.log(`loop ${iterations - iters} finished!`);
            if(count == 0 && prevStatus)
                break;
            prevStatus = loopEntireSet;
        }
        let supVecXs: number[][] = [];
        let supVecYs: number[][] = [];
        let supAlphas: number[] = [];
        // update support vectors
        this.alphas.forEach((val, index) => {
            if(val > 0) {
                supVecXs.push(trainXs.getRow(index));
                supVecYs.push([trainYs.get(0, index)]);
                supAlphas.push(val);
            }
        });
        this.supportVecTags = new Matrix(supVecYs);
        this.supportVecMat = new Matrix(supVecXs).transpose();
        this.alphas = supAlphas;
    }

    public predict(data: Matrix, tags: Matrix, kernelFunc: number) {
        if(!this.supportVecMat || !this.supportVecTags)
            throw new Error("You must train this model first!");
        let result = [];
        let xs = data.transpose();
        let ys = tags.transpose();
        let acc = 0;
        for(let i = 0; i < xs.columns; i++) {
            const prediction = this._gXi(xs.getColumnVector(i), kernelFunc, this.supportVecMat, this.supportVecTags) > 0 ? 1 : -1;
            if(ys.get(i, 0) == prediction)
                acc++;
            result.push(prediction);
        }
        console.log(acc / xs.columns);
        return result;
    }
}