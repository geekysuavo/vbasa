
#pragma once

namespace vbasa {

class SubstitutionMatrix {
public:
  /* index_t: tuple used for pair indexing.
   * sequence_t: sequence of vocabulary members.
   */
  using index_t = std::tuple<unsigned int, unsigned int>;
  using sequence_t = std::vector<std::string>;

  /* SubstitutionMatrix(): constructor taking configuration values.
   *
   * arguments:
   *  @vocab: axis values on the matrix rows and columns.
   *  @init: initial matrix elements, otherwise zero.
   */
  SubstitutionMatrix(
    const sequence_t& vocab,
    std::optional<const Eigen::Ref<const Eigen::MatrixXd>> init = {},
    double scaling = 1.0
  ) {
    if (vocab.size() == 0)
      throw std::out_of_range("vocabulary must be non-empty");

    n = vocab.size();
    for (auto& v : vocab)
      labels.emplace_back(v);

    coeffs.resize(n, n);
    coeffs.setZero();

    if (init)
      set_coeffs(init.value());

    scale = scaling;
  }

  /* eval(): evaluate the matrix for a pair of sequences.
   *
   * arguments:
   *  @a: first sequence, maps to output rows.
   *  @b: second sequence, maps to output columns.
   *
   * returns:
   *  matrix of probability values.
   */
  auto eval(const sequence_t& a, const sequence_t& b) const {
    const unsigned int na = a.size();
    const unsigned int nb = b.size();

    Eigen::MatrixXd result(na, nb);

    for (unsigned int u = 0; u < na; u++) {
      const auto i = lookup(a[u]);

      for (unsigned int v = 0; v < nb; v++) {
        const auto j = lookup(b[v]);
        result(u, v) = get({i, j});
      }
    }

    result = (result.array() * scale).exp();
    Eigen::VectorXd sums = result.rowwise().sum().cwiseInverse();
    result = sums.asDiagonal() * result;

    return result;
  }

  /* get_item(): get a matrix element from either ints or strings.
   */
  double get_item(const py::tuple& key) const {
    return get(lookup(key));
  }

  /* set_item(): store a matrix element from either ints or strings.
   */
  void set_item(const py::tuple& key, double value) {
    set(lookup(key), value);
  }

  /* get_coeffs(): get the coefficient matrix.
   */
  auto get_coeffs() const {
    return coeffs;
  }

  /* set_coeffs(): set the coefficient matrix.
   */
  void set_coeffs(const Eigen::Ref<const Eigen::MatrixXd> M) {
    if (M.rows() != n || M.cols() != n)
      throw std::length_error("initialization matrix size mismatch");

    for (unsigned int i = 0; i < n; i++)
      for (unsigned int j = 0; j < n; j++)
        set({i, j}, M(i, j));
  }

  /* get_labels(): get the label vector.
   */
  auto get_labels() const {
    auto tup = py::tuple(n);
    for (unsigned int i = 0; i < n; i++)
      tup[i] = labels[i];

    return tup;
  }

  /* managed data:
   *  @n: matrix dimensionality.
   *  @labels: labels along each matrix dimension.
   *  @coeffs: log-probability values for label pairs.
   *  @scale: scale factor for probability calculation.
   */
  unsigned int n;
  std::vector<std::string> labels;
  Eigen::MatrixXd coeffs;
  double scale;

private:
  /* lookup(): parse a string into an integer index.
   */
  unsigned int lookup(const std::string_view& str) const {
    const auto it = std::find(labels.cbegin(), labels.cend(), str);
    if (it == labels.cend())
      throw std::out_of_range("string not found in vocabulary");

    return it - labels.cbegin();
  }

  /* lookup(): parse a tuple into a pair of integer indices.
   */
  index_t lookup(const py::tuple& key) const {
    /* (int, int) accessor. */
    try {
      const auto i = key[0].cast<unsigned int>();
      const auto j = key[1].cast<unsigned int>();

      return {i, j};
    }
    catch (py::cast_error& e) {}

    /* (str, str) accessor. */
    try {
      const auto is = key[0].cast<std::string_view>();
      const auto js = key[1].cast<std::string_view>();

      const auto i = lookup(is);
      const auto j = lookup(js);

      return {i, j};
    }
    catch (py::cast_error& e) {}

    throw py::cast_error("expected int or string indices");
  }

  /* get(): get a value from the matrix.
   *
   * arguments:
   *  @key: tuple holding the (i,j) indices.
   *
   * returns:
   *  value at the specified index.
   */
  double get(const index_t& key) const {
    const auto [i, j] = key;
    if (i >= n || j >= n)
      throw std::out_of_range("index out of bounds");

    return coeffs(i, j);
  }

  /* set(): store a value into the matrix.
   *
   * arguments:
   *  @key: tuple holding the (i,j) indices.
   *  @value: new value to store.
   */
  void set(const index_t& key, double value) {
    const auto [i, j] = key;
    if (i >= n || j >= n)
      throw std::out_of_range("index out of bounds");

    coeffs(i, j) = value;
    coeffs(j, i) = value;
  }
};

} /* namespace vbasa */

