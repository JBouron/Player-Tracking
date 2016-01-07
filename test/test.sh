#!/bin/bash

# Test the basic functionality of the program.
../Bachelor_Project --test > test_results.out

# Compare the results with the expected output.
d=`diff expected.out test_results.out`
if [ "$d" == "" ]
then
	echo Test Succeded !
else
	echo Test Failed ! Contact us.
fi

# Delete the results.
rm test_results.out