for filename in *; do
  echo ____ PROGRAM ENDED SUCCESSFULLY WITH STATUS 0 AT \(%[^_]+%\)____ >> $filename
done
