#Arguments:
#$1 <ont_number>: "multi_ont" or "single_ont"

#python3 src/NORM/relation_extraction.py # Uncomment if there is no relation files in ./tmp dir

python3 src/NORM/pre_process_norm.py $1 

cd src/NORM

java ppr_for_ned_all $1

cd ../../

python3 src/NORM/post_process_norm.py $1
