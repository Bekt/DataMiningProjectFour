package ml.projectfour;

import ml.MLException;
import ml.Matrix;
import ml.SupervisedLearner;
import java.util.List;

import static helpers.Vector.sampleWithReplacement;

public class Bagging extends Ensemble {

    /**
     * Calls each model's train() with a sample with replacement
     */
    @Override
    public void train(Matrix features, Matrix labels) {
        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features size must be the same as labels size.");
        }
        this.features = features;
        this.labels = labels;

        for (SupervisedLearner model : models) {
            Matrix[] sample = sampleWithReplacement(features, labels, features.getNumRows());
            model.train(sample[0], sample[1]);
        }
    }

    @Override
    public List<Double> predict(List<Double> in) {
        return super.predict(in);
    }

}
