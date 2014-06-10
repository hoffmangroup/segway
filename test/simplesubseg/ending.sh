#!/bin/bash

ENDING=6755a364ee8a11e390d1ec55f9c205f4

for files in *.6755a364ee8a11e390d1ec55f9c205f4
do
 mv "$files" "${files%.$ENDING}.(%[0-9a-f]{32}%)"
done
