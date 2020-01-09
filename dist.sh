#!/usr/bin/env bash
# Script for building tar.gz archive for deployment
# author Oleg.Shpynov os@jetbrains.com

version=0.2
build=development

for ARGUMENT in "$@"; do
    KEY=$(echo "${ARGUMENT}" | cut -f1 -d=)
    VALUE=$(echo "${ARGUMENT}" | cut -f2 -d=)

    case "$KEY" in
            version)    version=${VALUE} ;;
            build)      build=${VALUE} ;;
            *)          echo "Unknown KEY ${KEY} = ${VALUE}"; exit 1;;
    esac
done


VERSION_BUILD="${version}.${build}"
FULL_VERSION="${VERSION_BUILD} built on $(date)"

# Update config version
mkdir -p pubtrends
cp config.properties pubtrends/
sed -E "s/version = [^\n]*/version = ${FULL_VERSION}/g"     -i pubtrends/config.properties
cp docker-compose.yml pubtrends/
cp -r models pubtrends/

# Create distributive tar.gz
rm -r dist
mkdir -p dist
tar -zcvf "dist/pubtrends-${VERSION_BUILD}.tar.gz" pubtrends
# Cleanup
rm -f pubtrends

# Move jar to dist if exists
if [[ -f build/libs/pubtrends-dev.jar ]]; then
  mv build/libs/pubtrends-dev.jar "dist/pubtrends-${VERSION_BUILD}.jar"
fi