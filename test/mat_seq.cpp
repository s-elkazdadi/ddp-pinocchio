#include "ddp/detail/mat_seq.hpp"
#include "ddp/detail/mat_seq_common.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <doctest/doctest.h>

DOCTEST_TEST_CASE("matrix sequence") {
  using namespace ddp;
  auto reg = indexing::vec_regular_indexer(0, 12, fix_index<3>{});
  auto reg2 = indexing::vec_regular_indexer(0, 12, dyn_index{2});
  auto reg3 = indexing::vec_regular_indexer(0, 12, fix_index<3>{});

  auto filtered = indexing::periodic_row_filter(reg, 3, 2);
  auto filtered2 = indexing::periodic_row_filter(reg3, 2, 1);

  auto concatenated = indexing::row_concat(filtered, reg2);

  auto idx = indexing::outer_prod(concatenated, filtered2);
  auto seq = detail::matrix_seq::mat_seq<double>(idx);
  auto* ptr = seq.data();

  auto it = begin(seq);

  DOCTEST_CHECK((**it).rows() == 2);
  DOCTEST_CHECK((**it).cols() == 0);

  ++it;
  DOCTEST_CHECK((**it).rows() == 2);
  DOCTEST_CHECK((**it).cols() == 3);

  (**it)(0, 0) = 1.0;
  (**it)(1, 0) = 2.0;
  (**it)(0, 1) = 3.0;

  ++it, ++it, ++it, ++it;
  DOCTEST_CHECK((**it).rows() == 5);
  DOCTEST_CHECK((**it).cols() == 3);

  (**it)(0, 0) = 4.0;
  (**it)(1, 0) = 5.0;
  (**it)(0, 1) = 6.0;

  DOCTEST_CHECK(ptr[0 + 2 * 0 + 0] == 1.0);
  DOCTEST_CHECK(ptr[0 + 2 * 0 + 1] == 2.0);
  DOCTEST_CHECK(ptr[0 + 2 * 1 + 0] == 3.0);

  DOCTEST_CHECK(ptr[12 + 5 * 0 + 0] == 4.0);
  DOCTEST_CHECK(ptr[12 + 5 * 0 + 1] == 5.0);
  DOCTEST_CHECK(ptr[12 + 5 * 1 + 0] == 6.0);
}
