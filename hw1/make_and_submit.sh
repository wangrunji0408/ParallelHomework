#!bin/sh
./make_floats 100000000 >testcase/1e8
rm -f result_1e8
job_id=$(echo $(qsub job_1e8.sh) | awk -F '[. ]' '{print $1}')
while [ ! -f "result_1e8" ]; do
	sleep 0.2
done
sleep 0.1
./check result_1e8
rm testcase/1e8 result_1e8
