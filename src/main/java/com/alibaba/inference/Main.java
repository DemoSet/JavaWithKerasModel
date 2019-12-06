package com.alibaba.inference;

import com.alibaba.inference.model.Model;

import java.util.Scanner;

/**
 * @author Lucien
 * @version 1.0.1
 */
public class Main {

    public static void main(String[] args) throws Exception {

        Model model = new Model("model/model.h5");

        String[] buffer;
        int[] input = new int[64];

        Scanner scanner = new Scanner(System.in);

        while (scanner.hasNext()) {
            buffer = scanner.next().split(",");

            for (int i = 0; i < 64; i++) {
                input[i] = Integer.parseInt(buffer[i]);
            }

            System.out.println(model.predict(input));
        }
    }
}
