for k in $( seq 1 50 )
do
    # python example_main.py -dataset='sparse_linear'
    # python example_main.py -dataset='financial'
    # python example_main.py -dataset='mushroom'
    # python example_main.py -dataset='jester'
    # python example_main.py -dataset='statlog'
    python example_main.py -dataset='adult'
    # python example_main.py -dataset='covertype'
    # python example_main.py -dataset='census'
done
