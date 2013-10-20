package ml.projectfour;

import ml.MLException;
import ml.Matrix;
import ml.SupervisedLearner;

import java.util.ArrayList;
import java.util.List;

public class Ensemble extends SupervisedLearner {

    Matrix features, labels;
    List<SupervisedLearner> models = new ArrayList<SupervisedLearner>();
    List<Double> modelWeights = new ArrayList<Double>();

    /**
     * Calls the train() method in each of the models
     */
    @Override
    public void train(Matrix features, Matrix labels) {

        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features size must be the same as labels size.");
        }
        this.features = features;
        this.labels = labels;

        for (SupervisedLearner model : models) {
            model.train(features, labels);
        }
    }

    /**
     * Uses the baseline approach to predict:
     * For categorical labels: most common value of all predictions
     * For continuous labels: mean of all predictions
     */
    @Override
    public List<Double> predict(List<Double> in) {
        Matrix predictions = predictModels(in);
        List<Double> out = new ArrayList<Double>();
        for (int i = 0; i < predictions.getNumCols(); i++) {
            if (predictions.isCategorical(i)) {
                out.add(predictions.mostCommonValue(i));
            } else {
                out.add(predictions.columnMean(i));
            }
        }
        return out;
    }

    public void addModel(SupervisedLearner model) {
        models.add(model);
        modelWeights.add(0.0);
    }

    /**
     * Puts predictions from each model into a matrix
     */
    protected Matrix predictModels(List<Double> in) {
        Matrix predictions = new Matrix(labels, true);
        for (SupervisedLearner model : models) {
            List<Double> prediction = model.predict(in);
            predictions.addRow(prediction);
        }
        return predictions;
    }

}
