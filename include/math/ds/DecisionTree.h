#pragma once

#include "../Matrix.h"
#include "../numerics/utils.h"
#include "Predictor.h"
#include <string>


enum ImpurityMeasure { ENTROPY = 0, GINI = 1 };

const char* measure_to_name(ImpurityMeasure measure) {
  switch(measure) {
    case ImpurityMeasure::ENTROPY: return "entropy";
    case ImpurityMeasure::GINI: return "gini";
    default: return "unknown";
  }
}
/*
 def entropy(s):
    counts = np.bincount(np.array(s, dtype=np.int64))
    percentages = counts / len(s)
    entropy = 0
    for pct in percentages:
        if pct > 0:
            entropy += pct*np.log2(pct)
    return -entropy
 */
double entropy(const Matrix<double>& in) {
  auto unique_values = unique(in);
  // TODO: Implement log2 to implement entropy... sadge
  return 0;
}

Matrix<double> count_bins(const Matrix<double>& in) {
  auto unique_values = sort(unique(in.rows() > in.columns() ? in : in.Transpose()));
  auto bins          = zeros(unique_values.rows(), 2);
  for(size_t i = 0; i < unique_values.rows(); ++i) {
    auto label                       = unique_values(i, 0);
    std::function<bool(double)> cond = [label](double xi) { return bool(xi == label); };
    bins(i, 0) = label;
    bins(i, 1)                       = where_true(where(cond, in, { { 1 } }, { { 0 } })).elements_total();
  }
  return bins;
}

/*

 def gini_impurity(s):
    counts = np.bincount(np.array(s, dtype=np.int64))
    percentages = counts / len(s)
    return 1 - (percentages**2).sum()
 */
double gini(const Matrix<double>& in) {
  auto bins = count_bins(in);
  bins = bins.GetSlice(0,bins.rows()-1,1,1);
  auto pct  = ((1.0 / (double)in.elements_total()) * bins).Apply([](double xi){return xi * xi;});
  return 1.0 - pct.sumElements();
}

enum DecisionNodeType { NONE = -1, LEAF = 0, DECISION = 1 };

/**
 * Binary Decision Tree Node.
 *
 * Avaibale in two types:
 * - LEAF: Leaf node holding final class label
 * - DECISION: Node holding two children that split a subset of the train data
 *
 */
class DecisionNode
{
public:
  // Leaf node
  double value = -1; // class label for leaf node

  // Decision node
  int feature            = -1;
  double threshold        = -1;
  double information_gain = -1;

  DecisionNode *left, *right;

  // general
  DecisionNodeType type = DecisionNodeType::NONE;

public:
  DecisionNode() { }

  DecisionNode(double val)
    : value(val) {
    type = DecisionNodeType::LEAF;
  }

  DecisionNode(int feat, double thresh, double ig, DecisionNode* left, DecisionNode* right)
    : feature(feat)
    , threshold(thresh)
    , information_gain(ig)
    , left(left)
    , right(right) {
    type = DecisionNodeType::DECISION;
  }
};


/**
 *
 */
struct DataSplit {
  int feature;
  double threshold;
  double information_gain;
  Matrix<double> left_split;
  Matrix<double> right_split;
} DataSplit;

/**
 * Decision (Bi-)Tree implementation
 * - The constructor allows providing a decision_method for the impurity measure that is used when selecting a good split
 * - The _information_gain(parent, left_child, right_child) calculates the information gain value of a split between a parent and two children
 * - The _best_split(X, y) function calculates the best splitting parameters for input features X and a target variable y
 *    - It does so by iterating over every column in X and every thresold value in every column to find the optimal
 * split using information gain
 * - The _build(X, y, depth) function recursively builds a decision tree until the data is _non splittable_
 * - The fit(X, y) function calls the _build() function and stores the built tree to the constructor
 * - The _predict(x) function traverses the tree to classify a single instance
 * - The predict(X) function applies the _predict() function to every instance in matrix X
 *
 */
class DecisionTree : public Predictor
{
  ImpurityMeasure _decision_method;
  int _max_depth             = 10;
  int _min_samples_per_split = 2;
  DecisionNode* base_node;

public:
  DecisionTree(
  int min_samples_per_split = 2, int max_depth = 10, ImpurityMeasure decision_method = ImpurityMeasure::GINI)
    : Predictor()
    , _decision_method(decision_method)
    , _min_samples_per_split(min_samples_per_split)
    , _max_depth(max_depth) { }


  double impurity(const Matrix<double>& x) {
    switch(_decision_method) {
      case ImpurityMeasure::ENTROPY: return entropy(x);
      case ImpurityMeasure::GINI: return gini(x);
      default: return -1;
    }
  }
  /**
   * Calculates the information gain of a set of values given a split of two children.
   *
   * Note: Pass your labels in here, not your input data!
   *
   * @param parent: super set of label values
   * @param left_child: left split of parent set
   * @param right_child: right split of parent set
   * @return information gain of given split
   */
  double
  information_gain(const Matrix<double>& parent, const Matrix<double>& left_child, const Matrix<double>& right_child) {
    double p_left  = (double)left_child.elements_total() / parent.elements_total();
    double p_right = (double)right_child.elements_total() / parent.elements_total();
    return impurity(parent) - (p_left * impurity(left_child) + p_right * impurity(right_child));
  }

  struct DataSplit best_split(const Matrix<double>& X, const Matrix<double>& y) {
    struct DataSplit best_split { };
    double best_info_gain = -1.;

    for(size_t idf = 0; idf < X.columns(); ++idf) {
      auto eV       = zerosV(X.columns());
      eV(idf, 0)    = 1;
      auto features = X * eV;
      for(size_t idx = 0; idx < features.rows(); ++idx) {
        auto df        = HorizontalConcat(X, y);
        auto threshold = features(idx, 0);

        std::function<bool(double)> cond = [threshold](double xi) { return bool(xi <= threshold); };
        auto split_indices               = where(cond, features, { { 1 } }, { { 0 } });
        auto left_indices                = where_true(split_indices);
        auto right_indices               = where_false(split_indices);
        if(left_indices.rows() > 0 && right_indices.rows() > 0) {
          auto left_split  = df.GetSlicesByIndex(left_indices);
          auto right_split = df.GetSlicesByIndex(right_indices);
          auto gain        = information_gain(
          y,
          left_split.GetSlice(0, left_split.rows() - 1, left_split.columns() - 1),
          right_split.GetSlice(0, right_split.rows() - 1, right_split.columns() - 1));
          if(gain > best_info_gain) {
            best_info_gain              = gain;
            best_split.information_gain = gain;
            best_split.feature          = idf;
            best_split.threshold        = threshold;
            best_split.left_split       = left_split;
            best_split.right_split      = right_split;
          }
        }
      }
    }
    return best_split;
  }

private:
  /**
   * Method to recursively build decision tree nodes.
   *
   * Uses pruning to decide to split the data further.
   * Class-Hyperparameters:
   * - max_depth
   * - min_samples_per_split
   *
   * @param X: input data of current node
   * @param y: related class labels
   * @return: A DecisionNode instance holding the next node.
   */
  DecisionNode* build_tree(const Matrix<double>& X, const Matrix<double>& y, int depth = 0) {
    DecisionNode* out = (DecisionNode*)malloc(sizeof(DecisionNode));
    if(X.rows() >= _min_samples_per_split && depth < _max_depth) {
      auto best = best_split(X, y);
      if(best.information_gain > 0) {
        auto left = build_tree(
        best.left_split.GetSlice(0, best.left_split.rows() - 1, 0, best.left_split.columns() - 2),
        best.left_split.GetSlice(0, best.left_split.rows() - 1, best.left_split.columns() - 1),
        depth + 1);
        auto right = build_tree(
        best.right_split.GetSlice(0, best.right_split.rows() - 1, 0, best.right_split.columns() - 2),
        best.right_split.GetSlice(0, best.right_split.rows() - 1, best.right_split.columns() - 1),
        depth + 1);

        out = new DecisionNode(best.feature, best.threshold, best.information_gain, left, right);
        return out;
      }
    }

    // found leaf node
    auto bins = count_bins(y);
    auto argmax_value = bins(argmax(bins.GetSlice(0, bins.rows() - 1, 1, 1)), 0);
    out = new DecisionNode(argmax_value);
    return out;
  }


  /**
   * Method to recursively traverse the tree to perform a prediction
   *
   * @param x: input value(s)
   * @param node: the current node to evaluate
   * @return: prediction for given node
   */
  double _predict(const Matrix<double>& x, DecisionNode* node) {
    if(node->type == DecisionNodeType::LEAF) { return node->value; }
    if(x(0, node->feature) <= node->threshold) { return _predict(x, node->left); }
    return _predict(x, node->right);
  }

  void render_node(std::ostream& ostr, DecisionNode* node, const char* side, int depth) const {
    ostr << std::string(depth, '\t') << side;
    if(node->type == DecisionNodeType::LEAF) {
      ostr << "label = " << node->value << std::endl;
    } else {
      ostr << "feature(" << node->feature << ") threshold(" << node->threshold << ")" << measure_to_name(_decision_method) << " = " << node->information_gain << std::endl;
      if(node->left) {
        render_node(
        ostr,
        node->left,
        ("L (x_" + std::to_string(node->feature) + " <= " + std::to_string(node->threshold) + "): ").c_str(),
        depth + 1);
      }
      if(node->right) {
        render_node(
        ostr,
        node->right,
        ("R (x_" + std::to_string(node->feature) + " > " + std::to_string(node->threshold) + "): ").c_str(),
        depth + 1);
      }
    }
  }

public:
  DecisionNode* GetRootNode() const { return base_node; }
  /**
   * Implements training algorithm
   * @param X: array-like with the shape: [n_samples, n_features]
   * @param y: array-like with shape: [n_samples, 1]
   * @return this
   */
  void fit(const Matrix<double>& X, const Matrix<double>& y) override {
    base_node = (DecisionNode*)malloc(sizeof(DecisionNode));
    base_node = build_tree(X, y, 0);
  }

  /**
   * Makes prediction for given input
   * @return
   */
  Matrix<double> predict(const Matrix<double>& X) override {
    auto out = zeros(X.rows(), 1);
    for(size_t i = 0; i < X.rows(); ++i) { out(i, 0) = _predict(X.GetSlice(i), base_node); }
    return out;
  }

  /**
   *
   *
   */
  Matrix<double> transform(const Matrix<double>& in) override { return in; };

  friend std::ostream& operator<<(std::ostream& ostr, const DecisionTree& tree) {
    ostr.precision(17);
    tree.render_node(ostr, tree.base_node, "root: ", 0);
    return ostr;
  }
};
