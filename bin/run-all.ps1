$env:PYTHONPATH="$PWD/src"

echo "Generating fake data..."
jupyter nbconvert --to notebook --execute notebooks/01-fake-data-generation.ipynb.ipynb --log-level=WARN --stdout > $null
echo "Fake data saved to output directory"

echo "Preparing data for modelling..."
jupyter nbconvert --to notebook --execute notebooks/02-preparing-data-for-modelling.ipynb --log-level=WARN --stdout > $null
echo "Preprocessed data saved to output directory"

echo "Running a random forest model..."
jupyter nbconvert --to notebook --execute notebooks/03-modelling.ipynb --log-level=WARN --stdout > $null
echo "Model outputs saved to output directory"