#include "ddp/indexer.hpp"
#include <doctest/doctest.h>

DOCTEST_TEST_CASE("regular indexer") {
  using namespace ddp;
  auto idx = indexing::mat_regular_indexer(3, 12, dyn_index(3), fix_index<4>{});

  DOCTEST_CHECK(idx.index_begin() == 3);
  DOCTEST_CHECK(idx.index_end() == 12);

  auto it = begin(idx);
  ++it;
  --it;
  it++;
  it--;
  it += 4;
  it -= 2;
  auto r = *it;
  auto it1 = it++;
  auto it2 = it--;
  auto it3 = ++it;
  auto it4 = --it;

  DOCTEST_CHECK(it1 == it);
  DOCTEST_CHECK(it2 == (it + 1));
  DOCTEST_CHECK(it3 == (it + 1));
  DOCTEST_CHECK(it4 == it);

  DOCTEST_CHECK(it == it);
  DOCTEST_CHECK(it != (it + 1));
  DOCTEST_CHECK((it + 1) > it);
  DOCTEST_CHECK((it - 1) < it);
  DOCTEST_CHECK((it + 1) >= it);
  DOCTEST_CHECK((it - 1) <= it);
  DOCTEST_CHECK(it >= it);
  DOCTEST_CHECK(it <= it);
  DOCTEST_CHECK(r.current_index() == 5);
  DOCTEST_CHECK(r.rows().value() == 3);
  DOCTEST_CHECK(r.cols().value() == 4);
}

DOCTEST_TEST_CASE("compose indexers") {
  using namespace ddp;
  auto idx = indexing::vec_regular_indexer(0, 12, fix_index<3>{});
  auto idx2 = indexing::vec_regular_indexer(0, 12, fix_index<2>{});
  auto idx3 = indexing::vec_regular_indexer(0, 12, fix_index<3>{});

  auto filtered = indexing::periodic_row_filter(idx, 3, 2);
  auto filtered2 = indexing::periodic_row_filter(idx3, 2, 1);

  auto concatenated = indexing::row_concat(filtered, idx2);
  auto prod = indexing::outer_prod(concatenated, filtered2);

  auto it = begin(prod);

  static_assert(decltype((*it).max_rows())::value_at_compile_time == 5, "");
  static_assert(decltype((*it).max_cols())::value_at_compile_time == 3, "");

  DOCTEST_CHECK((*it).max_rows().value() == 5);
  DOCTEST_CHECK((*it).max_cols().value() == 3);

  DOCTEST_CHECK((*it).rows().value() == 2);
  DOCTEST_CHECK((*it).cols().value() == 0);
  ++it;
  DOCTEST_CHECK((*it).rows().value() == 2);
  DOCTEST_CHECK((*it).cols().value() == 3);
  ++it;
  DOCTEST_CHECK((*it).rows().value() == 5);
  DOCTEST_CHECK((*it).cols().value() == 0);
  ++it;
  DOCTEST_CHECK((*it).rows().value() == 2);
  DOCTEST_CHECK((*it).cols().value() == 3);
  ++it;
  DOCTEST_CHECK((*it).rows().value() == 2);
  DOCTEST_CHECK((*it).cols().value() == 0);
  ++it;
  DOCTEST_CHECK((*it).rows().value() == 5);
  DOCTEST_CHECK((*it).cols().value() == 3);

  DOCTEST_CHECK(prod.stride(0) == 0);
  DOCTEST_CHECK(prod.stride(1) == 6);
  DOCTEST_CHECK(prod.stride(2) == 0);
  DOCTEST_CHECK(prod.stride(3) == 6);
  DOCTEST_CHECK(prod.stride(4) == 0);
  DOCTEST_CHECK(prod.stride(5) == 15);
}
