#!/usr/bin/env bash
set -euo pipefail
# Script for building tar.gz archive for deployment
# author Oleg.Shpynov os@jetbrains.com

VERSION=1.3
BUILD=development
GA=""

for ARGUMENT in "$@"; do
    KEY=$(echo "${ARGUMENT}" | cut -f1 -d=)
    VALUE=$(echo "${ARGUMENT}" | cut -f2 -d=)

    case "$KEY" in
            build)      BUILD=${VALUE} ;;
            ga)         GA=${VALUE} ;;
            *)          echo "Unknown KEY ${KEY} = ${VALUE}"; exit 1;;
    esac
done


VERSION_BUILD="${VERSION}.${BUILD}"
FULL_VERSION="${VERSION_BUILD} built on $(date +%F)"
echo "Full version $FULL_VERSION"
PTV=pubtrends-${VERSION_BUILD}

echo "Copy sources"
mkdir -p "${PTV}"
cp resources/docker/main/Dockerfile "${PTV}"/
cp config.properties "${PTV}"/
cp *.yml "${PTV}"/
cp scripts/init.sh "${PTV}"/
cp scripts/nlp.sh "${PTV}"/
cp README.md "${PTV}"/
cp LICENSE.txt "${PTV}"/
rsync -aiz pysrc "${PTV}"/ --exclude=**/.git* --exclude=**/__pycache__*

echo "Update config VERSION"
# Don't use sed -i for BSD compatibility
sed -E "s/VERSION[^\n]*/VERSION = '${FULL_VERSION}'/g" pysrc/version.py > "${PTV}"/pysrc/version.py

echo "Setup GA"
if [[ ! -z "${GA}" ]]; then
  GA_SCRIPT="<!-- Global site tag (gtag.js) - Google Analytics -->\
<script async src='https://www.googletagmanager.com/gtag/js?id=$GA'></script>\
<script>\
  window.dataLayer = window.dataLayer || [];\
  function gtag(){dataLayer.push(arguments);}\
  gtag('js', new Date());\
  gtag('config', '$GA');\
</script>"
  echo "Google Analytics script: $GA_SCRIPT"
  # Ignore init.html file and admin/ files
  for F in $(find "${PTV}"/pysrc -name "*.html" | grep -v init.html | grep -v admin/); do
    echo "Save version to $F"
    sed "s#{{ version }}#${FULL_VERSION}#" $F > $F.new
    mv $F.new $F
    if [[ -z "$(echo $F | grep 'admin')" ]]; then
      echo "Adding Google Analytics tracker to $F"
      sed "s#<head>#<head>${GA_SCRIPT}#" $F > $F.new
      mv $F.new $F
    fi
  done
fi

echo "Create distributive dist/${PTV}.tar.gz"
if [[ -d dist ]]; then
  rm -r dist
fi
mkdir -p dist
tar -zcf "dist/${PTV}.tar.gz" "${PTV}"

echo "Cleanup"
rm -r "${PTV}"

echo "Move jar to dist if exists"
if [[ -f build/libs/pubtrends-dev.jar ]]; then
  mv build/libs/pubtrends-dev.jar "dist/${PTV}.jar"
fi

echo "Done"