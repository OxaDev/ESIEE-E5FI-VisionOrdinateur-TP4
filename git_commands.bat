CD %0\..\
git add .
git commit -m "commit Auto"
git for-each-ref --format="%(refname)" --sort='authordate' refs/replace | xargs git push origin master