package ml.projectfour;

import helpers.Rand;
import ml.Matrix;
import ml.SupervisedLearner;

import java.util.*;

public class Bombing extends Ensemble {

    final int n;
    Matrix trainFeatures, trainLabels;
    Matrix predictFeatures, predictLabels;

    public Bombing(int n) {
        this.n = n;
    }

    /**
     * - Splits the training data into 2 parts
     * - Uses ensemble's train method to tain one half
     * - Evaluates n random weighing combinations and chooses the one that gives the lowest sum SSE
     * - The weights of the models are normalized
     */
    @Override
    public void train(Matrix features, Matrix labels) {
        int size = features.getNumRows(), mid = size / 2;

        trainFeatures = features.subMatrixRows(0, mid);
        trainLabels = labels.subMatrixRows(0, mid);
        predictFeatures = features.subMatrixRows(mid, size);
        predictLabels = labels.subMatrixRows(mid, size);

        super.train(trainFeatures, trainLabels);

        evaluateRandomCombinations();
    }

    /**
     * For categorical: chooses the label with the most total weight of predicted models
     * For continuous: the weights are normalized, so: prediction = sum[prediction_i * weight_i]
     */
    @Override
    public List<Double> predict(List<Double> in) {
        Matrix predictions = predictModels(in);
        List<Double> out = new ArrayList<Double>();
        for (int i = 0; i < predictions.getNumCols(); i++) {
            if (predictions.isCategorical(i)) {
                double colLabel = handleCategorical(predictions, i);
                out.add(colLabel);
            } else {
                double colMean = handleContinuous(predictions, i);
                out.add(colMean);
            }
        }
        return out;
    }

    private void evaluateRandomCombinations() {
        List<Double> bestWeights = null;
        double minSumError = Double.MAX_VALUE;

        for (int i = 0; i < n; i++) {
            double sumError = 0;
            weighModels();
            for (SupervisedLearner model : models) {
                sumError += model.getAccuracy(predictFeatures, predictLabels);
            }
            if (sumError < minSumError) {
                bestWeights = modelWeights;
                minSumError = sumError;
            }
        }
        modelWeights = bestWeights;
    }

    private void weighModels() {
        double sumOfWeights = 0;
        for (int i = 0; i < models.size(); i++) {
            double rand = Rand.drawFromStandardExp();
            modelWeights.set(i, rand);
            sumOfWeights += rand;
        }
        // Normalize weights
        for (int i = 0; i < models.size(); i++) {
            modelWeights.set(i, modelWeights.get(i) / sumOfWeights);
        }
    }

    private double handleCategorical(Matrix predictions, int col) {

        Map<Double, Double> valueTotalWeight = new HashMap<Double, Double>();
        for (int row = 0; row < predictions.getNumRows(); row++) {
            double value = predictions.getRow(row).get(col);
            double modelWeight = modelWeights.get(row);
            Double totalWeight = valueTotalWeight.get(value);
            totalWeight = totalWeight == null ? modelWeight : totalWeight + modelWeight;
            valueTotalWeight.put(value, totalWeight);
        }

        Set<Double> keys = valueTotalWeight.keySet();
        double max = Double.MIN_VALUE;
        double ans = Double.NEGATIVE_INFINITY;
        for (double key : keys) {
            double totalWeight = valueTotalWeight.get(key);
            if (totalWeight > max) {
                max = totalWeight;
                ans = key;
            }
        }

        return ans;
    }

    private double handleContinuous(Matrix predictions, int col) {
        double sum = 0;
        for (int row = 0; row < predictions.getNumRows(); row++) {
            sum += (predictions.getRow(row).get(col) * modelWeights.get(row));
        }
        return sum;
    }

}
