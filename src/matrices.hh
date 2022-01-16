
#pragma once

namespace vbasa::matrices {

/* one_letter_codes(): FIXME
 */
std::vector<std::string> one_letter_codes() {
  return {{"A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
           "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"}};
}

/* pam250(): create a PAM250 substitution matrix instance.
 */
auto pam250() {
  auto vocab = one_letter_codes();
  Eigen::MatrixXd init(vocab.size(), vocab.size());

  /* FIXME: fill init */

  vbasa::SubstitutionMatrix mat(vocab, init);
  return py::cast(mat);
}

/* blosum62(): create a BLOSUM62 substitution matrix instance.
 */
auto blosum62() {
  auto vocab = one_letter_codes();
  Eigen::MatrixXd init(vocab.size(), vocab.size());

  /* FIXME: fill init */

  vbasa::SubstitutionMatrix mat(vocab, init);
  return py::cast(mat);
}

} /* namespace vbasa::matrices */

