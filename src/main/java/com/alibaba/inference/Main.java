package com.alibaba.inference;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Scanner;

/**
 * @author Lucien
 * @version 1.0.0
 */
public class Main {

    private static String modelPath = "model/model.h5";

    public static void main(String[] args) throws Exception {
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(modelPath);

        INDArray input = Nd4j.zeros(1, 64);
        INDArray output;
        String[] buffer;

        Scanner scanner = new Scanner(System.in);

        while (scanner.hasNext()) {
            buffer = scanner.next().split(",");

            for (int i = 0; i < 64; i++) {
                input.putScalar(new int[]{0, i}, Integer.parseInt(buffer[i]));
            }

            output = model.output(input);

            System.out.println(output.argMax());
        }
    }
}
