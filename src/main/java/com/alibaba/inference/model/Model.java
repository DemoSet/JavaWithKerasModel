package com.alibaba.inference.model;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Lucien
 * @version 1.0.0
 */
public class Model {

    private MultiLayerNetwork model;

    public Model(String modelPath) throws Exception {
        System.err.printf("Start to load model from %s\n", modelPath);
        long start = System.currentTimeMillis();
        model = KerasModelImport.importKerasSequentialModelAndWeights(modelPath);
        long end = System.currentTimeMillis();
        System.err.printf("Load model finished, cost %d ms\n", end - start);
    }

    public int predict(int[] tokens) {
        INDArray input = Nd4j.zeros(1, tokens.length);

        for (int i = 0; i < tokens.length; i++) {
            input.putScalar(new int[]{0, i}, tokens[i]);
        }

        return model.output(input).argMax().getInt();
    }
}
