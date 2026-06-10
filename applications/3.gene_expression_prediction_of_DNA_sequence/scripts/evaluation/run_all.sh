for chr in Chr{01..12}; do
    echo "$chr"
    ./scripts/evaluation/run_demo_all.sh $chr
done
