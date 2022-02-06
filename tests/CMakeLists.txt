# Tests CMakeLists.txt

add_executable(test_base test_base.c)
target_link_libraries(test_base convGemm)

add_executable(test_back test_back.c)
target_link_libraries(test_back convGemm)

add_executable(test_trans test_trans.c)
target_link_libraries(test_trans convGemm)

add_custom_target(
        test.in
        COMMAND perl ${PROJECT_SOURCE_DIR}/tests/test.pl ${PROJECT_SOURCE_DIR}/tests/test.txt > test.in
)

add_custom_target(
        run_test_base
        COMMAND rm -f test_base.out
        COMMAND ${BASH} cat test.in | while read line\; do echo $\${line}\; for j in 1 2 3 4 5\; do ./test_base 6 $\${line} >> test_base.out || exit\; done\; done
        DEPENDS test_base test.in
)

add_custom_target(
        run_test_trans
        COMMAND rm -f test_trans.out
        COMMAND ${BASH} cat test.in | while read line\; do echo $\${line}\; for j in 1 2 3 4 5\; do ./test_trans 6 $\${line} >> test_trans.out || exit\; done\; done
        DEPENDS test_trans test.in
)

add_custom_target(
        run_test_back
        COMMAND rm -f test_back.out
        COMMAND ${BASH} cat test.in | while read line\; do echo $\${line}\; for j in 1 2 3 4 5\; do ./test_back 6 $\${line} >> test_back.out || exit\; done\; done
        DEPENDS test_back test.in
)

add_custom_target(
        run_all_tests
        DEPENDS run_test_base run_test_trans run_test_back
)
