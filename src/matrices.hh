
#pragma once

namespace vbasa::matrices {

/* one_letter_codes(): create a new vector of one-letter residue codes.
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

  init.triangularView<Eigen::StrictlyUpper>() =
    init.triangularView<Eigen::StrictlyLower>();

  vbasa::SubstitutionMatrix mat(vocab, init);
  return py::cast(mat);
}

/* blosum62(): create a BLOSUM62 substitution matrix instance.
 */
auto blosum62() {
  auto vocab = one_letter_codes();
  Eigen::MatrixXd init(vocab.size(), vocab.size());

  /* A:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
  init <<  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* R:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
          -1.0,  5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* N:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
          -2.0,  0.0,  6.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* D:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
          -2.0, -2.0,  1.0,  6.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* C:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0, -3.0, -3.0, -3.0,  9.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* Q:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
          -1.0,  1.0,  0.0,  0.0, -3.0,  5.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* E:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
          -1.0,  0.0,  0.0,  2.0, -4.0,  2.0,  5.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* G:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0, -2.0,  0.0, -1.0, -3.0, -2.0, -2.0,  6.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* H:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
          -2.0,  0.0,  1.0, -1.0, -3.0,  0.0,  0.0, -2.0,  8.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* I:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
          -1.0, -3.0, -3.0, -3.0, -1.0, -3.0, -3.0, -4.0, -3.0,  4.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* L:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* K:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* M:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* F:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* P:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* S:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* T:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* W:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* Y:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,

  /* FIXME */
  /* V:    A     R     N     D     C     Q     E     G     H     I
           L     K     M     F     P     S     T     W     Y     V */
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0;

  init.triangularView<Eigen::StrictlyUpper>() =
    init.triangularView<Eigen::StrictlyLower>();

  vbasa::SubstitutionMatrix mat(vocab, init);
  return py::cast(mat);
}

} /* namespace vbasa::matrices */

