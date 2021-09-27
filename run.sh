# creating list of arguments
args=""
for ITEM in "$@"
do
    args="$args $ITEM" 
done

python polynomial_fitting/main.py $args