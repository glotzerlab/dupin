for version in 3.9 3.10 3.11 3.12 3.13; do
    uv pip compile --python-version $version ../../../requirements/requirements-build.txt ../../../requirements/requirements-testing.txt ../../../requirements/requirements.txt > requirements$version.txt
done
