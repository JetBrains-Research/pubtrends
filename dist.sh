#!/usr/bin/env bash
# Script for building tar.gz archive for deployment
# author Oleg.Shpynov os@jetbrains.com

VERSION=0.19
BUILD=development

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
cp environment.yml "${PTV}"/
cp Dockerfile "${PTV}"/
cp config.properties "${PTV}"/
cp docker-compose.yml "${PTV}"/
cp gunicorn.sh "${PTV}"/
cp -r pysrc "${PTV}"/
cp -r pubtrends-review "${PTV}"/

echo "Update config VERSION"
sed -E "s/VERSION[^\n]*/VERSION = '${FULL_VERSION}'/g" -i "${PTV}"/pysrc/version.py

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

  for F in $(find "${PTV}"/pysrc -name "*.html"); do
    echo "Save version to $F"
    sed "s#{{ version }}#${FULL_VERSION}#" -i $F
    if [[ -z "$(echo $F | grep 'admin')" ]]; then
      echo "Adding Google Analytics tracker to $F"
      echo "$GA_SCRIPT"
      sed "s#<head>#<head>${GA_SCRIPT}#" -i $F
    fi
  done
fi

echo "Create folder for logs"
mkdir "${PTV}"/logs
chmod a+rwx "${PTV}"/logs

echo "Create distributive dist/${PTV}.tar.gz"
rm -r dist
mkdir -p dist
tar -zcf "dist/${PTV}.tar.gz" "${PTV}"

echo "Cleanup"
rm -r "${PTV}"

echo "Move jar to dist if exists"
if [[ -f build/libs/pubtrends-dev.jar ]]; then
  mv build/libs/pubtrends-dev.jar "dist/${PTV}.jar"
fi