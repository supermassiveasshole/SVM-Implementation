import * as iris from "./data/iris_processed.json";
import Matrix from "ml-matrix";
import { SVM } from "./svm/svm";
import * as cleveland from "./data/cleveland.json";

const map = new Map();
map.set('Iris-versicolor', 1);
map.set('Iris-setosa', -1);
let set: number[][] = [];
let labels: number[] = [];
(iris as string[]).forEach(val => {
    const splits = val.split(',');
    set.push([Number(splits[0]), Number(splits[1]), Number(splits[2]), Number(splits[3])]);
    labels.push(map.get(splits[4]));
});
const trainXs = new Matrix(set.slice(0, set.length - 20));
const trainYs = new Matrix([labels.slice(0, labels.length - 20)]);
const validateXs = new Matrix(set.slice(set.length - 20, set.length));
const validateYs = new Matrix([labels.slice(labels.length - 20, labels.length)]);

let svm = new SVM();
svm.fit(trainXs, trainYs, 0 ,2000);
const result = svm.predict(trainXs, trainYs, 0);
console.log(result);