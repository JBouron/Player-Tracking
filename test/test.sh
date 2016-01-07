#!/bin/bash

# Test the basic functionality of the program.
beg=`date`
echo Begin test : "$beg"
./Bachelor_Project --test > test_results.out
echo Test finished : `date`

# Compare the results with the expected output.
d=`diff test/expected.out test_results.out`
if [ "$d" == "" ]
then
	echo Test Succeded !
else
	echo Test Failed ! Contact us.
fi

# Delete the results.
rm test_results.out