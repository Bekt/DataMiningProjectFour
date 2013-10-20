package ml.projectfour;

import ml.*;
import ml.projectone.BaselineLearner;
import ml.projectthree.DecisionTreeLearner;
import ml.projectthree.RandomDecisionTreeLearner;
import ml.projecttwo.InstanceBasedLearner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.sqrt;

/**
 * @author kanat
 *
 * Observations:
 * - For bombing ensemble with random trees, the results are almost identical with N=1 and N=50
 * - Random forest with 100 tress is almost identical to bagging ensemble with 100 random trees
 */
public class Main {

    static String[] learnerNames = {
            "Baseline",
            "1-nn",
            "5-nn",
            "Entropy-Reducing Tree",
            "Bagging ensemble (4 entropy reducing trees, and 1-nn)",
            "Random forest with 100 trees",
            "Bombing ensemble with 100 random trees, n=50"};

    public static void main(String[] args) throws Exception {

        if (args.length == 0) {
            throw new RuntimeException("Need data!");
        }

        List<Matrix> matrices = parseFiles(args);
        Matrix[][] data = splitMatrices(matrices);

        List<List<Double>> table = runLearners(data);

        for (List<Double> setResult : table) {
            System.out.println(setResult.toString().replaceAll("[\\[\\]]", ""));
        }
    }

    static List<Matrix> parseFiles(String[] files) throws IOException {
        List<Matrix> matrices = new ArrayList<Matrix>();
        for (String file : files) {
            Matrix matrix = ARFFParser.loadARFF(file);
            matrices.add(matrix);
        }
        return matrices;
    }

    static Matrix[][] splitMatrices(List<Matrix> matrices) {
        Matrix[][] data = new Matrix[matrices.size()][2];

        for (int i = 0; i < matrices.size(); i++) {
            Matrix matrix = matrices.get(i);
            int cols = matrix.getNumCols();
            data[i][0] = matrix.subMatrixCols(0, cols - 1);
            data[i][1] = matrix.subMatrixCols(cols - 1, cols);
        }
        return data;
    }

    static List<List<Double>> runLearners(Matrix[][] data) {
        List<Filter> learners = getLearners();
        List<List<Double>> table = new ArrayList<List<Double>>();

        for (int set = 0; set < data.length; set++) {
            List<Double> setResult = new ArrayList<Double>();
            table.add(setResult);
            for (int i = 0; i < learners.size(); i++) {
                Filter learner = learners.get(i);
                double rmse = getRMSE(learner, data[set][0], data[set][1]);
                setResult.add(rmse);
            }
        }
        return table;
    }

    static List<Filter> getLearners() {
        List<Filter> learners = new ArrayList<Filter>();

        learners.add(getFilter(new BaselineLearner()));
        learners.add(getFilter(new InstanceBasedLearner(1)));
        learners.add(getFilter(new InstanceBasedLearner(5)));
        learners.add(getFilter(new DecisionTreeLearner(1)));

        Bagging bagging = new Bagging();
        for (int i = 0; i < 4; i++) {
            bagging.addModel(getFilter(new DecisionTreeLearner(1)));
        }
        bagging.addModel(getFilter(new InstanceBasedLearner(1)));
        learners.add(getFilter(bagging));

        Ensemble randomForest = new Ensemble();
        for (int i = 0; i < 100; i++) {
            randomForest.addModel(getFilter(new RandomDecisionTreeLearner(1)));
        }
        learners.add(getFilter(randomForest));

        Bombing bombing = new Bombing(50);
        for (int i = 0; i < 100; i++) {
            bombing.addModel(getFilter(new RandomDecisionTreeLearner(1)));
        }
        learners.add(getFilter(bombing));

        return learners;
    }

    static Filter getFilter(SupervisedLearner learner) {
        return new Filter(learner, new Imputer(), true);
    }

    static double getRMSE(Filter learner, Matrix features, Matrix labels) {
        final int nFoldSize = 2;
        final int repetitions = 5;

        return sqrt(learner.repeatNFoldCrossValidation(features, labels, nFoldSize, repetitions));
    }

}
